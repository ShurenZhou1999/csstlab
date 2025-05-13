import sys, os, warnings
import numpy as np
from scipy.interpolate import RectBivariateSpline
from .base import BaseEmulator_GP
from .emu_simu import Emulator_simu
from .emu_loop import Emulator_loop

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
        '''
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
    


    def __to_k_mask(self, k_array = None, z_array=None, ):
        '''
        In the ouput results, we mask the low-k comonpents of `P_{1\delta^3}` and `P_{\delta\delta^3}`, where there are almost noise in this region. 
        '''
        if k_array is None : k_array = self.__k
        if z_array is None : z_array = self.__z
        self.__Mask_k = np.ones(( 21, len(z_array), len(k_array), ), dtype='int32', )
        ## Given single redshift, return 1-D array instead of 2-D array
        ## see inner function :: self.set_k_and_z
        if len(z_array)==1 :
            self.__Mask_k = np.squeeze(self.__Mask_k, axis=1)
        for (l, kmax) in [
            ( 5, 0.35), # (0, 5),   The theory and simulation result are inconsistent in all region. 
            (10, 0.20), # (1, 5), 
            (18, 0.0015), # (4, 4), 
        ]:
            self.__Mask_k[l][..., k_array<kmax ] = 0
    

    def __set_intepolation(self, ):
        '''
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
        
        for (i, j, kInd) in [ 
            (0, 2, 4), 
            (0, 3, 4), 
            (1, 2, 5), 
            (1, 3, 5), 
            (2, 4, 4), 
            (3, 4, 4), 
            (4, 4, 4), 
            (2, 5, 2), 
            (3, 5, 2), 

            (1, 5, 20), 
            (4, 5, 32), 
        ]: 
            kdrop0 = np.sum( self.__klin < self.__k[kInd] ) - 3
            self.__intp_kdrop[i][j] = [kdrop0, kInd, ]
            self.__k_stack[i][j] = np.hstack([ self.__klin[:kdrop0], self.__k[kInd:], ])



    def set_k_and_z(self, k, z):
        '''
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

        if len(z)==1 : self.__intp_zsort = 0       ## Given single redshift, return 1-D array instead of 2-D array
        else         : self.__intp_zsort = np.argsort(z)     ## The redshift is ordered by `z=0 -> z=3` as the interpolation function requires.
        
        self.__has_set_k_and_z = True
        self.__set_k = k
        self.__set_z = z[self.__intp_zsort]
        self.__to_k_mask(k, z) 
        self.__set_pkij = np.zeros(( 21, len(z), len(k), ), dtype='float64', )
        self.__set_pkij = np.squeeze(self.__set_pkij)     ## shape as (21, Nk) if `z` is a scalar

    
    def unset_k_and_z(self, ):
        '''
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
            return self.__z.copy()
    

    def release__Mask(self, ) :
        '''
        This function release the k-region mask which set the unreliable k-region as zeros.
        The calling of `set_k_and_z` and `unset_k_and_z` will reset the mask.
        '''
        self.__Mask_k = np.ones(1)


    
    def __call__(self, Param, ):
        '''
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
                self.__set_pkij[l] = \
                RectBivariateSpline(
                    self.__z[::-1], self.__k_stack[i][j], data_pk[::-1],   ## `z` axis should be descending
                    kx=3, ky=3,  
                )(  self.__set_z, self.__set_k, grid=True,  )[self.__intp_zsort]
            return  self.__set_pkij *self.__Mask_k
        else:
            pk_D = [ pk_D[:, i, j] for (i, j) in self._index ]
            return  pk_D *self.__Mask_k
    

    @property
    def EFTofLSS_Model(self,):
        return EFTofLSS_Model








class EFTofLSS_Model:
    def __init__(self, ):
        pass

    @staticmethod
    def CombinePkij( k, pks, 
                    b_1, b_2, b_s2, b_n2, b_3=None, ):
        '''
        combine the Lagrangian basis power spectrum to the biased tracer power spectrum
        not include the shot noise term in the auto-power spectrum
        '''
        P_cross = pks[0] + b_1 *pks[1] + b_2 *pks[2] + b_s2 *pks[3] + b_n2 *pks[4]
        P_auto =    ( pks[0] + 2*b_1 *pks[1] + 2*b_2 *pks[2] + 2*b_s2 *pks[3] + 2*b_n2 *pks[4] ) + \
                b_1*(            b_1 *pks[6] + 2*b_2 *pks[7] + 2*b_s2 *pks[8] + 2*b_n2 *pks[9] ) + \
                b_2*(                            b_2 *pks[11]+ 2*b_s2 *pks[12]+ 2*b_n2 *pks[13]) + \
                b_s2*(                                           b_s2 *pks[15]+ 2*b_n2 *pks[16]) + \
                b_n2*(                                                            b_n2 *pks[18]) 
        if b_3 is not None:
            P_cross += b_3 * pks[5] 
            P_auto  += 2* b_3 *( pks[5] + b_1 *pks[10] + b_2 *pks[14] + b_s2 *pks[17] + b_n2 *pks[19] ) \
                        + b_3*b_3 *pks[20]
        return P_auto, P_cross
    

    @staticmethod
    def CombinePkij_replace_nabla2( k, pks, b_1, b_2, b_s2, b_n2, b_3=None, ):
        '''
        same as `CombinePkij`, but replace the $\nabla^2\delta$ with $-k^2\delta$
        '''
        b_n2 = - k**2 *b_n2
        P_cross = pks[0] + b_1 *pks[1] + b_2 *pks[2] + b_s2 *pks[3] + b_n2 *pks[0]
        P_auto =    ( pks[0] + 2*b_1 *pks[1] + 2*b_2 *pks[2] + 2*b_s2 *pks[3] + 2*b_n2 *pks[0] ) + \
                b_1*(            b_1 *pks[6] + 2*b_2 *pks[7] + 2*b_s2 *pks[8] + 2*b_n2 *pks[1] ) + \
                b_2*(                            b_2 *pks[11]+ 2*b_s2 *pks[12]+ 2*b_n2 *pks[2]) + \
                b_s2*(                                           b_s2 *pks[15]+ 2*b_n2 *pks[3]) + \
                b_n2*(                                                            b_n2 *pks[0]) 
        if b_3 is not None:
            P_cross += b_3 * pks[5] 
            P_auto  += 2* b_3 *( pks[5] + b_1 *pks[10] + b_2 *pks[14] + b_s2 *pks[17] + b_n2 *pks[5] ) \
                        + b_3*b_3 *pks[20]
        return P_auto, P_cross
    
    
    @staticmethod
    def CombinePkij_plus_1shotnoise( k, pks, shotnoise, biasList, ):
        alpha, bs = biasList[:1], biasList[1:]
        pk_hh, pk_hm = EFTofLSS_Model.Pkij_to_biasPk( pks, *bs )
        pk_hh += alpha[0] *shotnoise
        return pk_hh, pk_hm
    
    
    @staticmethod
    def CombinePkij_plus_2shotnoise( k, pks, shotnoise, biasList, ):
        alpha, bs = biasList[:2], biasList[2:]
        pk_hh, pk_hm = EFTofLSS_Model.Pkij_to_biasPk( pks, *bs )
        pk_hh += (alpha[0] + alpha[0]*k**2) *shotnoise
        return pk_hh, pk_hm
    
    

    @staticmethod
    def CombinePkij_gradient( pks, b_1, b_2, b_s2, b_n2, b_3=None, ):
        '''
        Gradient of the biased tracer power spectrum with respect to the bias parameters
        '''
        gridP_cross = [ pks[1], pks[2], pks[3], pks[4], ]
        gridP_auto = [ 
            2* (pks[1] + b_1 *pks[6] + b_2 *pks[7] + b_s2 *pks[8] + b_n2 *pks[9] ) , 
            2* (pks[2] + b_1 *pks[7] + b_2 *pks[11]+ b_s2 *pks[12]+ b_n2 *pks[13]) ,
            2* (pks[3] + b_1 *pks[8] + b_2 *pks[12]+ b_s2 *pks[15]+ b_n2 *pks[16]) ,
            2* (pks[4] + b_1 *pks[9] + b_2 *pks[13]+ b_s2 *pks[16]+ b_n2 *pks[18]) ,
        ]
        if b_3 is not None:
            gridP_cross.append( pks[5] )
            gridP_auto.append( 
                2*( pks[5] + b_1 *pks[10] + b_2 *pks[14] + b_s2 *pks[17] + b_n2 *pks[19] + b_3 *pks[20] )
            )
            gridP_auto[0] += 2*b_3 *pks[10]
            gridP_auto[1] += 2*b_3 *pks[14]
            gridP_auto[2] += 2*b_3 *pks[17]
            gridP_auto[3] += 2*b_3 *pks[19]
        return gridP_auto, gridP_cross
