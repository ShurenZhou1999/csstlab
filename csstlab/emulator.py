import sys, os, warnings
import numpy as np
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from .base import BaseEmulator_GP
from .emu_simu import Emulator_simu
from .emu_loop import Emulator_loop
from .emu_error import estimate_emulation_error

warnings.simplefilter('always', UserWarning)
warnings.formatwarning = \
    lambda message, category, filename, lineno, line=None  : \
            f"{category.__name__}: {message}\n"



class Emulator(BaseEmulator_GP):
    r'''
    Hybrid Lagrangian Bias Expansion emulator. 
    The emulation range is :
    --------------------------------
    k-scale : 
        [0.001, 1.05] h/Mpc
    z-scale : 
        [0, 3]
    Cosmological parameters :
        \Omega_b : 0.04 - 0.06
        \Omega_m : 0.24 - 0.40
        h        : 0.6 - 0.8
        n_s      : 0.92 - 1.00
        10^9 A_s : 1.7 - 2.5
        w_0      : -1.3 - -0.7
        w_a      : -0.5 - 0.5
        M_\nu    : 0 - 0.3 eV
    --------------------------------
    '''
    def __init__(self, remake=False, ):
        r'''
        remake : bool, default is False
            If True, re-calculate the ratio for the simulation emulator, 
            and re-calculate the principal components decomposition and the Gaussian Process training for the loop-result emulator
        '''
        super().__init__()
        self.__PathData = os.path.dirname(__file__) + "/data/"
        self.__FileSimu = self.__PathData + "GP_simu.npy"
        self.__FileLoop = self.__PathData + "GP_loop.npy"
        self.__FileLin  = self.__PathData + "GP_lin.npy"
        self.__kmin = 0.001     ## minimum emulation k-bin in unit of [h/Mpc]
        self.__kmax = 1.05      ## maximum emulation k-bin in unit of [h/Mpc]
        self.__kmax_lin = 0.2
        self.__Nz = 12
        self.Nloop_samples = 14000       ## The number of samples to train the theory emulator; the parameters like PC numbers are fine-tuned to this sample size.
        self.EmuSimu = Emulator_simu(kmax=self.__kmax)
        self.EmuLoop = Emulator_loop(kmax=self.__kmax, opt_PCs = 1, )
        self.EmuLin  = Emulator_loop(kmax=self.__kmax_lin, opt_PCs = 2, )   # linear scale of 1-loop power spectrum
        self.__set_emulators(remake=remake)

        self.__has_set_k_and_z = False
        self.__to_k_mask()
        self.__set_intepolation()
        self.__set_emulationError()


    def __set_emulators(self, remake=False, ):
        if os.path.exists( self.__FileSimu ) and \
           os.path.exists( self.__FileLoop ) and \
           os.path.exists( self.__FileLin  ) and \
            not remake :
            self._load_emulator()
        else:
            self._train_emulator()
        self.__z = self.redshifts      # `z` and `redshifts` refer to the same array
        self.__k = self.EmuSimu.k      # `k` bin measured from the simulation
        self.Nk = self.__k.shape[0]
        self.__klin = self.EmuLin.k    # `k`-bin of linear scale, where the samples are replaced by 1-loop Pk
    

    def _train_emulator(self, ):
        print("Emulator message :: Training begins. Several minutes are required.")  ## 
        try: self.__load_raw_results( )
        except FileNotFoundError:
            raise FileNotFoundError(
                    "\n  The theoretical 1-loop power spectrum may not included in the `./data` folder due to large size. "
                  + "\n  You may generate them and then remake the emulator. \n" )
        paramsNorm = self.NormalizeParam(self.ext_Params)
        self.EmuLoop._train_emulator( paramsNorm, self.__k, self.ext_Pkij_T, 
                                    to_save=True, filename=self.__FileLoop, )
        #self.EmuLoop._load_emulator( self.__FileLoop )   ## TEST
        self.EmuLin ._train_emulator( paramsNorm, self.__klin, self.Pkij_lin, 
                                     to_save=True, filename=self.__FileLin, )
        #self.EmuLin ._load_emulator( self.__FileLin  )   ## TEST

        paramsNorm = self.NormalizeParam(self.Parameters)
        paramsNorm_and_z = self.To_Abscissas_MultiCosmo(paramsNorm)
        ## Instead of use accurate theoretical results, we use the output of trained theory-emulator as templates to eliminate the model inaccuracy. 
        pk_T = np.array([ self.EmuLoop( iparam ) for iparam in paramsNorm ])   #self.Pkij_T
        pk_D = self.Pkij[..., self.__k<self.__kmax ]
        karr = self.__k[self.__k<self.__kmax]
        self.EmuSimu._train_emulator( paramsNorm_and_z, karr, pk_D, pk_T, 
                                    to_save=True, filename=self.__FileSimu, )
        
        del self.Parameters, self.Pkij, self.Pkij_T
        del self.ext_Params, self.ext_Pkij_T
        del self.Pkij_lin

    
    def _load_emulator(self, ):
        self.redshifts = np.array(self.Redshift)
        self.EmuSimu._load_emulator( self.__FileSimu )
        self.EmuLoop._load_emulator( self.__FileLoop )
        self.EmuLin ._load_emulator( self.__FileLin  )


    def __load_raw_results(self, ):
        Dload = np.load( self.__PathData + "HLPT_simulation_Pkij.npy", allow_pickle=True, )[()]
        self.Parameters = Dload["Parameters"]
        self.redshifts = Dload["redshifts"]   # attention the redshift is ordered by  z=3 -> z=0 
        self.Pkij = Dload["Pkij"]
        self.Pkij_T = None   ## Dload["Pkij_T"]
        self.__k = Dload["k"]
        self.raw_kedges = Dload["kedges"]

        Nloop_samples = self.Nloop_samples
        Eload = np.load( self.__PathData + "HLPT_1loop_calculation.npy", allow_pickle=True, )[()]
        self.ext_Params = Eload["Parameters"][:Nloop_samples]
        self.ext_Pkij_T = Eload["Pkij_T"][:Nloop_samples]

        Lload = np.load( self.__PathData + "HLPT_1loop_calculation_linear-scale.npy", allow_pickle=True, )[()]
        self.__klin  = Lload["k"]
        self.Pkij_lin = Lload["Pkij"][:Nloop_samples]
    


    def __set_emulationError(self, ):
        '''
        the estimated error of the basis spectra. 
        Given shape as : 
            (z,  i, j, m, n,  k_1, k_2)
        '''
        self.__emuErr_stat, self.__emuErr_syst = estimate_emulation_error(self.__PathData)

        elements = [ ]
        for i in range(6):
            for j in range(i+1):
                for m in range(6):
                    for n in range(m+1):
                        elements.append( 
                            tuple(sorted([
                                tuple(sorted([i, j])), 
                                tuple(sorted([m, n])), 
                            ]) )
                        )
        self.__ee_ij = set(elements)

        for ee in [self.__emuErr_stat, self.__emuErr_syst, ]:
            for (i, j, kInd) in self.__index_stack:
                ee[:, i, j, ..., :kInd, :] = 0.0
                ee[:, j, i, ..., :kInd, :] = 0.0
                ee[:, ..., i, j, :, :kInd] = 0.0
                ee[:, ..., j, i, :, :kInd] = 0.0
        



    def __to_k_mask(self, k_array = None, z_array=None, ):
        r'''
        In the ouput results, we mask the low-k comonpents of `P_{1\delta^3}` and `P_{\delta\delta^3}`, where there are almost noise in this region. 
        '''
        if k_array is None : k_array = self.__k
        if z_array is None : z_array = self.__z
        self.__Mask_k = np.ones(( len(z_array), 6, 6, len(k_array), ), dtype='int32', )
        for (i, j, kmax) in [
            (0, 5, 0.35), # (0, 5),   The theory and simulation result are inconsistent in all region. 
            (1, 5, 0.20), # (1, 5), 
            (4, 4, 0.0015), # (4, 4), 
        ]:
            self.__Mask_k[:, i, j][..., k_array<kmax ] = 0
            self.__Mask_k[:, j, i] = self.__Mask_k[:, i, j]
        ## Given single redshift, return 1-D array instead of 2-D array
        ## see inner function :: self.set_k_and_z
        if len(z_array)==1 :
            self.__Mask_k = np.squeeze(self.__Mask_k, axis=0)
    

    def __set_intepolation(self, ):
        r'''
        set the connection between linear-scale 1-loop Pk and the non-linear-scale simulation Pk
        '''
        ## 
        #self.__intp_kdrop = 1      ## drop the first few k-bin in data
        #self.__k_stack = np.hstack([ self.__klin, self.__k[self.__intp_kdrop:], ])
        self.__intp_kdrop = self._empty_list
        self.__k_stack    = self._empty_list
        kdrop0 = np.sum( self.__klin < self.__k[0] ) - 2
        k_stack = np.hstack([ self.__klin[:kdrop0], self.__k[0:], ])
        for l, (i, j) in enumerate(self._index):
            self.__intp_kdrop[i][j] = [kdrop0, 0, ]
            self.__k_stack[i][j] = k_stack
        
        self.__index_stack = [
            (0, 2, 5), 
            (0, 3, 4), 
            (1, 2, 5), 
            (1, 3, 6), 
            (2, 4, 6), 
            (3, 4, 4), 
            (4, 4, 4), 
            (2, 5, 2), 
            (3, 5, 2), 

            (0, 5, 29), 
            (1, 5, 21), 
            (4, 5, 32), 
        ]
        for (i, j, kInd) in self.__index_stack:
            kdrop0 = np.sum( self.__klin < self.__k[kInd] ) - 3
            self.__intp_kdrop[i][j] = [kdrop0, kInd, ]
            self.__k_stack[i][j] = np.hstack([ self.__klin[:kdrop0], self.__k[kInd:], ])



    def set_k_and_z_default(self, ):
        self.set_k_and_z( self.__k, self.__z, )
    

    def set_k_and_z(self, k, z):
        r'''
        Set the k-bin and z-bin for the interpolation. 
        ----------
        k, z : 1D arrays or scalar
            The k-bin and z-bin for the interpolation. 
            The k-bin should be in the range of [0.001, 1.05] h/Mpc, and the z-bin should be in the range of [0, 3].
        ----------
        '''
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)

        if np.max(k) > self.__kmax or np.min(k) < self.__kmin :
            warnings.warn( f"\nThe wavenumber `k` is out of the range of the emulator ({self.__kmin} < k < {self.__kmax} h/Mpc). \n"
                          +f"Note that for those k < {self.__kmin} or k > {self.__kmax}, the emulator will be extrapolated. ")
        if np.max(z) > 3 or np.min(z) < 0 :
            warnings.warn("\nThe redshift `z` is out of the range of the emulator (0 < z < 3). \n"
                          +f"Note that for those z < 0 or z > 3, the emulator will be extrapolated. ")
        if np.any( np.diff(k) <= 0 ):
            raise Warning("The `k` array should be in strictly ascending order")
        if np.any( np.diff(np.sort(z)) <= 0 ):
            raise Warning("The `z` array should be in strictly ascending order")

        if len(z)==1 : 
            self.__intp_zsort = 0       ## Given single redshift, return 1-D array instead of 2-D array
            self.__set_Nz = 1
        else : 
            self.__intp_zsort = np.argsort(z)     ## The redshift is ordered by `z=0 -> z=3` as the interpolation function requires.
            self.__set_Nz = len(z)
        self.__set_Nk = len(k)
        
        self.__has_set_k_and_z = True
        self.__set_k = k
        self.__set_z = z[self.__intp_zsort]
        self.__to_k_mask(k, z)
        self.__set_pkij = np.zeros(( len(z), 6, 6, len(k), ), dtype='float64', )
        self.__set_pkij = np.squeeze(self.__set_pkij)     ## shape as (6, 6, Nk) if `z` is a scalar

    
    def unset_k_and_z(self, ):
        r'''
        Recover to the default k-bin and z-bin. Not interpolation for the output spectra. 
        '''
        self.__has_set_k_and_z = False
        self.__to_k_mask()
        self.__intp_zsort = None
        self.__set_k , self.__set_z  = None, None
    

    @property
    def k(self, ):
        '''
        The k-bin of the emulator output
        '''
        if self.__has_set_k_and_z : 
            return self.__set_k.copy()
        else : 
            return self.__k.copy()
    
    @property
    def z(self, ):
        '''
        The z-bin of the emulator output
        '''
        if self.__has_set_k_and_z : 
            return self.__set_z.copy()
        else : 
            return self.__z[::-1].copy()
    

    def release__Mask(self, ) :
        r'''
        This function release the k-region mask which set the unreliable k-region as zeros.
        The calling of `set_k_and_z` and `unset_k_and_z` will reset the mask.
        '''
        self.__Mask_k = np.ones(1)


    
    def __call__(self, Param, ):
        r'''
        Array of Cosmological parameters 
            ( Omega_b, Omega_m, h, n_s, 10^9 As, w0, w_a, M_nu, )
        
        Parameters
        ----------
        Param : 1D array with shape (8)

        If k and z are set, return P_ij array, with shape (21, Nz, Nk), where 21 is the number of P_ij components.
        Otherwise, the (k, z) bins are the default setting in training the emulator.
        '''
        ParamNorm = self.NormalizeParam(Param)
        if np.any( np.abs(ParamNorm) > 1):
            warnings.warn("The input Cosmological parameters are out of the range of the emulator. ")
        ParamNorm_and_z = self.To_Abscissas(ParamNorm)
        pk_T = self.EmuLoop( ParamNorm, )
        pk_D = self.EmuSimu( ParamNorm_and_z, pk_T, ) 
        
        if self.__has_set_k_and_z :
            pk_lin = self.EmuLin( ParamNorm, )
            for l, (i, j) in enumerate(self._index):
                kdrop0, kdrop1 = self.__intp_kdrop[i][j]
                data_pk = np.hstack([ pk_lin[:, i, j, :kdrop0], pk_D[:, i, j, kdrop1: ], ])
                self.__set_pkij[..., i, j, :] = \
                RectBivariateSpline(
                    self.__z[::-1], self.__k_stack[i][j], data_pk[::-1],   ## `z` axis should be descending
                    kx=3, ky=3,  
                )(  self.__set_z, self.__set_k, grid=True,  )[self.__intp_zsort]
                self.__set_pkij[..., j, i, :] = self.__set_pkij[..., i, j, :]
            return  self.__set_pkij *self.__Mask_k
        else:
            #pk_D = [ pk_D[:, i, j] for (i, j) in self._index ]
            return  pk_D *self.__Mask_k
    


    

    ## -----------------------------------------------------------------------------
    ## emulator error
    ## -----------------------------------------------------------------------------


    def error(self, etype='tot', Param=None, norm=True, ):
        '''
        ----------
        etype : 'tot' | 'syst' | 'stat'
        Param : cosomological parameters, optional
        norm  : bool, default is True
            If True, return the normalized error matrix, $\rho_{ijmn} = Cov_{ijmn} /(P_{ij}P_{mn})$
            when 'norm=True', `Param` should be given.
        ----------
        return the Covariance matrix of the basis spectra error, with shape as
            (Nz, 6, 6, 6, 6, Nk, Nk)
        '''
        if not self.__has_set_k_and_z:
            self.set_k_and_z_default()
        
        if etype == 'tot':
            ee = self.__emuErr_stat + self.__emuErr_syst
        elif etype == 'syst':
            ee = self.__emuErr_syst
        elif etype == 'stat':
            ee = self.__emuErr_stat
        else:
            raise ValueError(f"Emulator error type `{etype}` is not supported. ")
        
        set_kz = np.meshgrid(self.__set_z, self.__set_k, self.__set_k, indexing='ij', )
        set_kz = np.array([set_kz[0], set_kz[1], set_kz[2], ]).T
        err_out = np.zeros(( self.__set_Nz, 6, 6, 6, 6, self.__set_Nk, self.__set_Nk, ), dtype='float64', )
        _k = self.__k.copy()
        _k[ 0] -= 1e-5
        _k[-1] += 1e-2

        for [[i, j], [m, n]] in self.__ee_ij:
            val_interp = RegularGridInterpolator( (self.__z[::-1], _k, _k), values=ee[:, i, j, m, n][::-1],
                        method="nearest", fill_value=0, bounds_error=False, 
                   )( set_kz ).T
            val_interp_T = val_interp.transpose(0, 2, 1) 
            err_out[:, i, j, m, n] = val_interp
            if m!=n:
                err_out[:, i, j, n, m] = val_interp
            if i!=j:
                err_out[:, j, i, m, n] = val_interp
                if m!=n:
                    err_out[:, j, i, n, m] = val_interp
            if i!=m or j!=n:
                err_out[:, m, n, i, j] = val_interp_T
                if m!=n:
                    err_out[:, n, m, i, j] = val_interp_T
                if i!=j:
                    err_out[:, m, n, j, i] = val_interp_T
                    if m!=n:
                        err_out[:, n, m, j, i] = val_interp_T
        
        err_out = np.squeeze(err_out)
        if norm:
            return err_out
        elif Param is None:
            raise ValueError("When `norm=False`, the `Param` should be given to calculate the error matrix. ")
        
        Pk_ij = self.__call__(Param)
        if Pk_ij.ndim==3:
            return err_out * Pk_ij.reshape(6, 6, 1, 1, -1, 1) * Pk_ij.reshape(1, 1, 6, 6, 1, -1)
        else:
            nz = Pk_ij.shape[0]
            return err_out * Pk_ij.reshape(nz, 6, 6, 1, 1, -1, 1) * Pk_ij.reshape(nz, 1, 1, 6, 6, 1, -1)
            

    

        
    ## -----------------------------------------------------------------------------
    ## differentiation methods
    ## -----------------------------------------------------------------------------
    
    def Pk_ij( self, 
            Omega_b = None, 
            Omega_m = None, 
            h = None, 
            n_s= None, 
            As1e9 = None, 
            w_0 = None, 
            w_a= None, 
            M_nu = None, 
    ):
        r'''
        wrapper function for the `self.__call__` method to accept the parameters separately 
        '''
        return self.__call__( np.array([
            Omega_b, Omega_m, h, n_s, As1e9, w_0, w_a, M_nu,
        ]) )
    
    
    def Pk_ij_FiniteDiff_Dparam(self, 
            Omega_b = None, 
            Omega_m = None, 
            h = None, 
            n_s= None, 
            As1e9 = None, 
            w_0 = None,
            w_a= None, 
            M_nu = None, 
            Iparam : int = None, 
            eps = 0.05
        ):
        r'''
        Finite difference implementation of the 1-st derivative of the power spectrum with respect to the cosmological parameters.
        '''
        Param = np.array([
            Omega_b, Omega_m, h, n_s, As1e9, w_0, w_a, M_nu,
        ])
        ones = np.ones_like(Param) 
        ones[Iparam] = 1 + eps
        pk_i1 = self.__call__( Param *ones, )
        ones[Iparam] = 1 + 2*eps
        pk_i2 = self.__call__( Param *ones, )
        ones[Iparam] = 1 - eps
        pk_j1 = self.__call__( Param *ones, )
        ones[Iparam] = 1 - 2*eps
        pk_j2 = self.__call__( Param *ones, )
        pk_deri = ( -pk_i2 + 8*pk_i1 - 8*pk_j1 + pk_j2 ) / (12*eps*Param[Iparam])
        return pk_deri 
    

    def Pk_ij_FiniteDiff_DparamDparam(self, 
            Omega_b = None, 
            Omega_m = None, 
            h = None, 
            n_s= None, 
            As1e9 = None, 
            w_0 = None,
            w_a= None, 
            M_nu = None, 
            Iparam1 : int = None, 
            Iparam2 : int = None, 
            eps = 0.05
        ):
        r'''
        Finite difference implementation of 2-nd derivative of the power spectrum with respect to the cosmological parameters.
        '''
        Param = np.array([
            Omega_b, Omega_m, h, n_s, As1e9, w_0, w_a, M_nu,
        ])
        if Iparam2 is None or Iparam1==Iparam2 :
            ## 4-th order accurate 
            ones = np.ones_like(Param)
            ones[Iparam1] = 1 + eps
            pk_i1 = self.__call__( Param *ones, )
            ones[Iparam1] = 1 + 2*eps
            pk_i2 = self.__call__( Param *ones, )
            ones[Iparam1] = 1 - eps
            pk_j1 = self.__call__( Param *ones, )
            ones[Iparam1] = 1 - 2*eps
            pk_j2 = self.__call__( Param *ones, )
            pk_i0 = self.__call__( Param, )
            pk_deri = ( -pk_i2 + 16*pk_i1 - 30*pk_i0 + 16*pk_j1 - pk_j2 ) / (12*eps*eps*Param[Iparam1]*Param[Iparam1])
        else:
            if Iparam1 > Iparam2 :
                Iparam1, Iparam2 = Iparam2, Iparam1
            ## second order accurate ; formula with 4-th accuracy is too long ... ... 
            ones = np.ones_like(Param)
            ones[Iparam1] = 1 + eps ; ones[Iparam2] = 1 + eps
            pk_i1 = self.__call__( Param *ones, )
            ones[Iparam1] = 1 + eps ; ones[Iparam2] = 1 - eps
            pk_i2 = self.__call__( Param *ones, )
            ones[Iparam1] = 1 - eps ; ones[Iparam2] = 1 + eps
            pk_j1 = self.__call__( Param *ones, )
            ones[Iparam1] = 1 - eps ; ones[Iparam2] = 1 - eps
            pk_j2 = self.__call__( Param *ones, )
            pk_deri = ( pk_i1 - pk_i2 - pk_j1 + pk_j2 ) / (4*eps*eps*Param[Iparam1]*Param[Iparam2])
        return pk_deri
    
    



    
    def __repr__(self, ):
        return "Hybrid Lagrangian Bias Expansion Emulator\n" \
            + f"  k-range : [{self.__kmin}, {self.__kmax}] h/Mpc\n" \
            + f"  z-range : [{0}, {3}] \n" \
            + f"\n" \
            + f"  Cosmological parameters list should be given in following ordering and ranges : \n" \
            + f"    Omega_b  : {0.04} - {0.06}\n" \
            + f"    Omega_m  : {0.24} - {0.40}\n" \
            + f"    h        : {0.6} - {0.8}\n" \
            + f"    n_s      : {0.92} - {1.00}\n" \
            + f"    10^9 A_s : {1.7} - {2.5}\n" \
            + f"    w_0      : {-1.3} - {-0.7}\n" \
            + f"    w_a      : {-0.5} - {0.5}\n" \
            + f"    M_nu     : {0} - {0.3} eV\n" 
    



