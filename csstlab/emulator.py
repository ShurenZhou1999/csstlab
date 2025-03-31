import sys, os, warnings
import numpy as np
from scipy.interpolate import interpn
from .base import BaseEmulator_GP
from .emu_simu import Emulator_simu
from .emu_loop import Emulator_loop



class Emulator(BaseEmulator_GP):
    def __init__(self, remake=False, ):
        '''
        remake : bool, default is False
            If True, remake the ratio for the simulation emulator, 
            and remake the principal components decomposition and the Gaussian Process training for the loop-result emulator
        '''
        super().__init__()
        self.__PathData = os.path.dirname(__file__) + "/data/"
        self.__FileSimu = self.__PathData + "GP_simu.npy"
        self.__FileLoop = self.__PathData + "GP_loop.npy"
        self.__FileLin  = self.__PathData + "GP_lin.npy"
        self.__kmax = 1.05
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
        self.z = self.redshifts      # `z` and `redshifts` refer to the same array
        self.k = self.EmuSimu.k      # `k` bin measured from the simulation
        self.Nk = self.k.shape[0]
        self.klin = self.EmuLin.k    # `k`-bin of linear scale, where the samples are replaced by 1-loop Pk
    

    def _train_emulator(self, ):
        print("Emulator message :: Training begins. Several minutes are required.")  ## 
        try: self.__load_raw_results( )
        except FileNotFoundError:
            raise FileNotFoundError(
                    "\n  The theoretical 1-loop power spectrum may not included in the `./data` folder due to large size. "
                  + "\n  You may generate them and then remake the emulator. \n" )
        paramsNorm = self.NormalizeParam(self.ext_Params)
        #self.EmuLoop._train_emulator( paramsNorm, self.k, self.ext_Pkij_T, 
        #                            to_save=True, filename=self.__FileLoop, )
        self.EmuLoop._load_emulator( self.__FileLoop )   ## TEST
        self.EmuLin ._train_emulator( paramsNorm, self.klin, self.Pkij_lin, 
                                     to_save=True, filename=self.__FileLin, )
        #self.EmuLin ._load_emulator( self.__FileLin  )   ## TEST

        paramsNorm = self.NormalizeParam(self.Parameters)
        paramsNorm_and_z = self.To_Abscissas_MultiCosmo(paramsNorm)
        ## Instead of use accurate theoretical results, we use the output of trained theory-emulator as templates to eliminate the model inaccuracy. 
        pk_T = np.array([ self.EmuLoop( iparam ) for iparam in paramsNorm ])   #self.Pkij_T
        pk_D = self.Pkij[..., self.k<self.__kmax ]
        karr = self.k[self.k<self.__kmax]
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
        self.k = Dload["k"]
        self.raw_kedges = Dload["kedges"]

        Nloop_samples = self.Nloop_samples
        Eload = np.load( self.__PathData + "HLPT_1loop_calculation.npy", allow_pickle=True, )[()]
        self.ext_Params = Eload["Parameters"][:Nloop_samples]
        self.ext_Pkij_T = Eload["Pkij_T"][:Nloop_samples]

        Lload = np.load( self.__PathData + "HLPT_1loop_calculation_linear-scale.npy", allow_pickle=True, )[()]
        self.klin  = Lload["k"]
        self.Pkij_lin = Lload["Pkij"][:Nloop_samples]
    


    def __to_k_mask(self, k_array = None, z_array=None, ):
        '''
        In the ouput results, we mask the low-k comonpents of `P_{1\delta^3}` and `P_{\delta\delta^3}`, where there are almost noise in this region. 
        '''
        if k_array is None : k_array = self.k
        if z_array is None : z_array = self.z
        if len(z_array) == 1:
            ## Given single redshift, return 1-D array instead of 2-D array
            ## see inner function :: self.set_k_and_z
            self.__Mask_k = np.ones(( 21, len(k_array), ), dtype='int32', )
        else:
            self.__Mask_k = np.ones(( 21, len(z_array), len(k_array), ), dtype='int32', )
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
        #self.__k_stack = np.hstack([ self.klin, self.k[self.__intp_kdrop:], ])
        self.__intp_kdrop  = self._empty_list
        self.__k_stack     = self._empty_list
        self.__intp_method = self._empty_list
        kdrop0 = np.sum( self.klin < self.k[0] ) - 2
        k_stack = np.hstack([ self.klin[:kdrop0], self.k[0:], ])
        for l, (i, j) in enumerate(self._index):
            self.__intp_kdrop[i][j] = [kdrop0, 0, ]
            self.__k_stack[i][j] = k_stack
            if i==j or (i, j)==(0, 1) or (i, j)==(2, 3) :
                self.__intp_method[i][j] = 'cubic'
            else:
                self.__intp_method[i][j] = 'slinear'
        
        
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
            kdrop0 = np.sum( self.klin < self.k[kInd] ) - 3
            self.__intp_kdrop[i][j] = [kdrop0, kInd, ]
            self.__k_stack[i][j] = np.hstack([ self.klin[:kdrop0], self.k[kInd:], ])



    def set_k_and_z(self, k, z):
        #if np.max(k) > self.__kmax or np.min(k) < self.k[0] :
        #    warnings.warn("The wavenumber `k` is out of the range of the emulator. Note that these range will be extrapolated. ")
        if np.max(z) > 3 or np.min(z) < 0 :
            warnings.warn("The redshift `z` is out of the range of the emulator. Note that these range will be extrapolated. ")

        if len(z)==1 : self.__intp_zcut = 0       ## Given single redshift, return 1-D array instead of 2-D array
        else         : self.__intp_zcut = ...

        self.__SamplingPoint_k_z = np.array( np.meshgrid( 
                            # self.NormalizeTime(z), 
                            z, k , indexing="ij"  )   ).T
        self.__has_set_k_and_z = True
        self.__set_k = k
        self.__set_z = z
        self.__to_k_mask(k, z) 

    
    def unset_k_and_z(self, ):
        '''
        recover to the default k-bin and z-bin, without interpolation
        '''
        self.__has_set_k_and_z = False
        self.__to_k_mask()
        self.__intp_zcut = None
        self.__SamplingPoint_k_z = None
    

    def set_intepolation_method(self, method='cubic'):
        '''
        Using lower order interpolation method, such as `cubic`, can accelate the calculation.
        ------------
        method : str, default is 'cubic',
        '''
        #if method not in [ "cubic", "quintic" ]:
        #    raise ValueError("We do not recommand the method %s for interpolation. " % method)
        return None
        self.__intp_method = method
    

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
        Otherwise, the Nz and Nk is the default value in training the emulator.
        '''
        ParamNorm = self.NormalizeParam(Param)
        if np.any( np.abs(ParamNorm) > 1):
            warnings.warn("The input Cosmological parameters are out of the range of the emulator. ")
        ParamNorm_and_z = self.To_Abscissas(ParamNorm)
        pk_T = self.EmuLoop( ParamNorm, )
        pk_D = self.EmuSimu( ParamNorm_and_z, pk_T, ) 
        
        if self.__has_set_k_and_z :
            pk_lin = self.EmuLin( ParamNorm, )
            pks_out = [ ]
            for l, (i, j) in enumerate(self._index):
                kdrop0, kdrop1 = self.__intp_kdrop[i][j]
                data_k  = self.__k_stack[i][j]
                data_pk = np.hstack([ pk_lin[:, i, j, :kdrop0], pk_D[:, i, j, kdrop1: ], ])
                ipk_intp = \
                interpn( (self.z, data_k), data_pk ,       ## redshifts -> TimeNormalized
                        xi = self.__SamplingPoint_k_z , 
                        method = self.__intp_method[i][j],     ## The `quintic` method is accurate for P(k) in low-k .
                        bounds_error = False, 
                        fill_value = None, 
                    ).T[self.__intp_zcut]        ## transform the array to match the shape and order of (Nz, Nk) 
                pks_out.append( ipk_intp )
            return pks_out *self.__Mask_k
        else:
            pk_D = [ pk_D[:, i, j] for (i, j) in self._index ]
            return pk_D *self.__Mask_k
    

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
