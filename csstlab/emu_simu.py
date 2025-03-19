import numpy as np
from scipy.signal import savgol_filter
from .base import *
from .GaussianProcess import GaussianProcessRegressor, ConstantKernel, RBF



class Emulator_simu(BaseEmulator_GP):
    '''
    Parameters
    ----------
    '''
    def __init__(self, kmax=1.05, ):
        super().__init__()
        self.GP_kernel = ConstantKernel(2) * RBF(5)
        self.RescaleFunc = self._empty_list
        self.__kmax = kmax
        self.Nz = 12
    

    def __set_rescaleDamping(self, k):
        '''
        [ fine tunning parameters based on LOO test ]
        '''
        
        self.CrossZeros_DampFunc = [ 
            #(3, 0),  (3, 1), 
            (3, 2), 
            (4, 2), (4, 3),
            (5, 3), (5, 4), 
        ]
        self.CrossZeros_DampFunc  += [(j, i) for (i, j) in self.CrossZeros_DampFunc ]

        Damping_r2_1 = lambda paras, x : paras[0] * np.exp( - (x-paras[1]) )
        Damping_r2_0 = lambda paras, x : paras[0] * np.exp( - (x-paras[1])**2 )
        #damp_30 = Damping_r2([ 1, 3 ], False, )(karr)
        #damp_31 = Damping_r2([ 2.583e-01,  1.259e+00], False, )(karr)
        self.damp_32 = Damping_r2_1([ 0.02, 4, ], k)
        self.damp_42 = Damping_r2_0([ 8, 2, ], k)
        self.damp_53 = Damping_r2_0([ 2, 1.5, ], k)
        self.damp_54 = Damping_r2_0([ 2, 1.5, ], k)
    
    
    
    def __offset(self, i, j, pk_T, ):
        '''
        [ fine tunning parameters based on LOO test ]

        ------------
        ii, jj : int
        pk_T   : Ndarray, shape (*, 6, 6, Nk)
        ------------
        '''
        if (i, j) in self.CrossZeros_DampFunc:
            pk_T_i = pk_T[..., i, i, :].reshape(-1, self.Nk)
            pk_T_j = pk_T[..., j, j, :].reshape(-1, self.Nk)
            offset = np.sqrt(pk_T_i*pk_T_j)
            if (i, j)==(2, 4) : offset *= self.damp_42
            if (i, j)==(3, 5) : offset *= self.damp_53
            if (i, j)==(4, 5) : offset *= self.damp_54
        else:
            offset = 0
        return offset

    def __rescale(self, i, j, pk_T, pk_D,  ):
        pk_t = pk_T[..., i, j, :].reshape(-1, self.Nk)
        pk_d = pk_D[..., i, j, :].reshape(-1, self.Nk)
        offset = self.__offset(i, j, pk_T, )
        ratio = (pk_d+offset) /(pk_t+offset)
        ratio = np.abs(ratio)
        return np.log( ratio )
    
    def __rescale_inv(self, i, j, pk_T, ratio=None, ):
        pk_t = pk_T[..., i, j, :].reshape(-1, self.Nk)
        offset = self.__offset(i, j, pk_T, )
        return np.exp(ratio) * (pk_t+offset) - offset
        
    
    
    def __smooth(self, k, pk_D, pk_T, ):
        '''
        [ fine tunning parameters based on LOO test ]
        '''
        k_cut = 1     ## the index of first k-bin, to avoid large fluctuation
        k_max = 0.8   ## the maximum k range, to avoid the unstable small scale of `pk_T`
        kdrop1 = 5    # 
        kdrop2 = 11
        List_kcut_H1 = [ (0, 2), (1, 2), (0, 3), 
                         (0, 4), (1, 4),  (4, 4), ]    ## for these basis, the smoothing seems biased for first few k-bins
        List_kcut_H2 = [ (3, 1) ]
        List_NotSmooth = [ # (0, 0), (0, 1), (1, 1),   ## NOT smoothing beacuse of the BAO wiggles
                           (3, 2), (4, 2), ]     ## do NOT smooth these basis spectra with crossing-zero
        List_kcut_H1   = List_kcut_H1   + [ (j, i) for (i, j) in List_kcut_H1 ]
        List_kcut_H2   = List_kcut_H2   + [ (j, i) for (i, j) in List_kcut_H2 ]
        List_NotSmooth = List_NotSmooth + [ (j, i) for (i, j) in List_NotSmooth ]

        ratio = pk_D /pk_T
        ratio[ np.isnan(ratio) ] = 0
        sratio = savgol_filter( ratio[..., k_cut:], window_length=7, polyorder=2, mode="constant", cval=1, )
        spk_D = sratio *pk_T[..., k_cut:]
        pk_out = pk_D.copy()
        pk_out[..., k_cut:][..., k[k_cut:]<k_max] = spk_D[..., k[k_cut:]<k_max]
        for i in range(6):
            for j in range(6):
                if (i, j) in List_kcut_H1: 
                    pk_out[..., i, j, :kdrop1] = pk_D[..., i, j, :kdrop1]
                if (i, j) in List_kcut_H2: 
                    pk_out[..., i, j, :kdrop2] = pk_D[..., i, j, :kdrop2]
                if (i, j) in List_NotSmooth : 
                    pk_out[..., i, j, :] = pk_D[..., i, j, :]
        return pk_out
    
    
    
    def _train_emulator(self, Params, k, pk_D, pk_T, 
                 to_save=False, filename=None, 
        ):
        '''
        Parameters
        ----------
        Params : Ndarray, shape ( Ncosmo*(Nparams+1),  Nz )
        pk_D   : Ndarray, shape (Ncosmo, Nz, Pk_ii, Pk_jj, Nk)
        pk_T   : Ndarray, shape (Ncosmo, Nz, Pk_ii, Pk_jj, Nk)
        '''
        if k.shape[0] != pk_D.shape[-1] or k.shape[0] != pk_T.shape[-1] :
            raise ValueError("The k-bin ranges of `k`, `pk_D` and `pk_T` should be the same. ")
        if to_save : List_pk_ratio = self._empty_list
        
        self.krange = k < self.__kmax
        k, pk_D, pk_T = k[self.krange], pk_D[..., self.krange], pk_T[..., self.krange]
        self.__set_rescaleDamping(k)
        self.k, self.Nk = k, np.sum(self.krange)
        self.N_training_samples = pk_D.shape[0]

        pk_D = self.__smooth( k, pk_D, pk_T, )
        for ipk, jpk in self._index:
            #pk_d = pk_D[..., ipk, jpk, :].reshape(-1, Nk)
            #pk_t = pk_T[..., ipk, jpk, :].reshape(-1, Nk)
            pk_ratio = self.__rescale(ipk, jpk, pk_T, pk_D=pk_D,  )
            pk_ratio[ np.isnan(pk_ratio) ] = 0
            gaussP = GaussianProcessRegressor( Params, pk_ratio, 
                        self.GP_kernel, alpha=1e-12, normalize_y=True, )
            self.GPs[ipk][jpk] = gaussP
            
            if to_save : List_pk_ratio[ipk][jpk] = pk_ratio
        
        if to_save : 
            self.save( filename, {
                "k" : k, 
                "Params" : Params, 
                "pk_ratio" : List_pk_ratio,
            })
    
    
    def _load_emulator(self, filename):
        dataload = self.load(filename)
        self.N_training_samples = dataload["pk_ratio"][0][0].shape[0]
        k, Params, pk_ratio = dataload["k"], dataload["Params"], dataload["pk_ratio"]
        self.Nk, self.k = k.shape[0], k
        self.__set_rescaleDamping(k)
        self._set_GPs(Params, pk_ratio)
    
    
    def __call__(self, Param, pk_T, ):
        '''
        Parameters
        ----------
        Params : Ndarray, shape ( Nparams+1 = 9,  Nz ). 
            Normalized cosmological parameters with range [-1, 1]
        pk_T   : Ndarray, shape (Nz, Pk_ii, Pk_jj, Nk). 
            Theoretical templates for prediction
        '''
        y_pred = np.zeros(( self.Nz , 6, 6, self.Nk, ))
        for ipk, jpk in self._index:
            r_pred = self.GPs[ipk][jpk].predict( Param )
            r_pred[ np.isnan(r_pred) ] = 0
            y_pred[:, ipk, jpk ] = self.__rescale_inv(ipk, jpk, pk_T, r_pred )
            y_pred[:, jpk, ipk ] = y_pred[:, ipk, jpk ]
        return y_pred
    

