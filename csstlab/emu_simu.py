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
        self.PCA_Sim = None
        self.RescaleFunc = self._empty_list
        self.__kmax = kmax
        self.__N_PCs = 12
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
        #self.damp_32 = Damping_r2_1([ 0.02, 4, ], k)
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
            offset = np.sqrt( pk_T[..., i, i, :] *pk_T[..., j, j, :] )
            if (i, j)==(2, 4) : offset *= self.damp_42
            if (i, j)==(3, 5) : offset *= self.damp_53
            if (i, j)==(4, 5) : offset *= self.damp_54
        else:
            offset = 0
        return offset

    def __rescale(self, i, j, pk_T, pk_D,  ):
        pk_t = pk_T[..., i, j, :]#.reshape(-1, self.Nk)
        pk_d = pk_D[..., i, j, :]#.reshape(-1, self.Nk)
        offset = self.__offset(i, j, pk_T, )
        ratio = (pk_d+offset) /(pk_t+offset)
        ratio = np.abs(ratio)
        return np.log( ratio )
    
    def __rescale_inv(self, i, j, pk_T, ratio=None, ):
        pk_t = pk_T[..., i, j, :]#.reshape(-1, self.Nk)
        offset = self.__offset(i, j, pk_T, )
        return np.exp(ratio) * (pk_t+offset) - offset
        
    
    
    def __smooth(self, k, pk_D, pk_T, ):
        '''
        [ fine tunning parameters based on LOO test ]
        '''
        k_cut = 0     ## the index of first k-bin, to avoid large fluctuation
        k_max = 0.8   ## the maximum k range, to avoid the unstable small scale of `pk_T`
        kdrop1 = 5    # 
        kdrop2 = 11
        List_kcut_H1 = [ (0, 2), (1, 2), (0, 3), 
                         (0, 4), (1, 4),  (4, 4), ]    ## for these basis, the smoothing seems biased for first few k-bins
        List_kcut_H2 = [ (3, 1) ]
        List_NotSmooth = [ (0, 0), (0, 1), (1, 1),   ## smoothing introduce additional noise !
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
        
        self.krange = k < self.__kmax
        k, pk_D, pk_T = k[self.krange], pk_D[..., self.krange], pk_T[..., self.krange]
        self.__set_rescaleDamping(k)
        self.k, self.Nk = k, np.sum(self.krange)
        self.N_training_samples = pk_D.shape[0]
        ## mask the extreme noisy region in the simulation data
        for (i, j, kmax) in [ 
            (0, 5, 0.20), 
            (1, 5, 0.10),  
        ]: 
            win_filter_lowk = 0.5 + 0.5* np.tanh( -(kmax-self.k)*50 )
            pk_D[..., i, j, :,] *= win_filter_lowk
            pk_D[..., j, i, :,] *= win_filter_lowk

        List_pk_ratio = [ ]
        pk_D = self.__smooth( k, pk_D, pk_T, )
        for ipk, jpk in self._index:
            pk_ratio = self.__rescale(ipk, jpk, pk_T, pk_D=pk_D,  )
            pk_ratio[ np.isnan(pk_ratio) ] = 0
            List_pk_ratio.append( pk_ratio )
        List_pk_ratio = np.array(List_pk_ratio)

        self.PCA_Sim = PrincipalComponentAnalysis__simu( k, List_pk_ratio, kmax=self.__kmax , N_PCs=self.__N_PCs, )
        self._set_GPs( Params, self.PCA_Sim.Array_Aij, )
        self.k = k[self.PCA_Sim.krange]

        #    gaussP = GaussianProcessRegressor( Params, pk_ratio, 
        #                self.GP_kernel, alpha=1e-12, normalize_y=True, )
        #    self.GPs[ipk][jpk] = gaussP
            
        #    if to_save : List_pk_ratio[ipk][jpk] = pk_ratio
        if to_save : 
            Rij, Aij = self.PCA_Sim.read_PCs()
            self.save( filename, {
                "k" : self.k ,   # note that we only use k < 1.05
                "Params" : Params, 
                "Rij" : Rij,
                "Aij" : Aij, 
                "N_training_samples" : Aij.shape[-1],
                ## we have saved the Aij in the GP-model, thus we do not need to save the training samples again.
                #"pk_ratio" : List_pk_ratio,
            })
            self._save_GPs( filename )
    
    
    def _load_emulator(self, filename):
        Dload = self.load(filename)
        k, self.Params = Dload["k"], Dload["Params"] 
        self.k, self.Nk = k, k.shape[0]
        self.N_training_samples = Dload["N_training_samples"]
        
        self.PCA_Sim = PrincipalComponentAnalysis__simu( k, None, kmax=self.__kmax , N_PCs=self.__N_PCs, )
        self.PCA_Sim.set_PCs( Dload["Rij"], Dload["Aij"] )
        self.__set_rescaleDamping(k)
        self._load_GPs( filename )
    
    
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
            coeffs = self.GPs[ipk][jpk].predict( Param )
            r_pred = self.PCA_Sim( ipk, jpk, coeffs )
            r_pred[ np.isnan(r_pred) ] = 0
            y_pred[:, ipk, jpk ] = self.__rescale_inv(ipk, jpk, pk_T, r_pred )
            y_pred[:, jpk, ipk ] = y_pred[:, ipk, jpk ]
        return y_pred
    





class PrincipalComponentAnalysis__simu:
    def __init__( self, k, Pk_T=None, 
                kmax = 1.05, 
                onlyPCA = True, 
                N_PCs = 12, 
        ):
        self.__N_PCs = N_PCs
        self.onlyPCA = onlyPCA
        self.k = k
        self.kmax = kmax
        self.krange = k < self.kmax
        self.Nk = np.sum(self.krange)
        self.Nz = 12
        self.__index = [ (i, j) for i in range(6) for j in range(i, 6) ]
        
        if Pk_T is not None:
            self.__make_PCA( Pk_T,)
    
    @property
    def __empty_list(self, ):
        return [ [None for i in range(6)] for j in range(6) ]
    
    def __get_N_PCs(self, i, j, ):
        if (i, j) in [ (0, 5), (5, 0), ] : return 45    # maximum number of PCs (k-bins)
        if (i, j) in [ (1, 5), (5, 1), ] : return 45
        return self.__N_PCs
    

    def __make_PCA(self, Pk_T, ):
        if self.k.shape[0] != Pk_T.shape[-1] :
            raise ValueError("ND-array with wrong shape" )
        krange = self.krange
        self.Array_Aij = self.__empty_list
        self.Array_Rij = self.__empty_list
        if not self.onlyPCA : 
            self.Pk_T = Pk_T
            self.Array_Prec = self.__empty_list
        
        for icount, (ipk, jpk) in enumerate(self.__index):
            matT = Pk_T[icount][..., krange].reshape(-1, self.Nk)
            eigvecL, eigval, eigvecR = np.linalg.svd( matT , full_matrices=False )
            Aij = eigvecL*eigval
            N_PCs = self.__get_N_PCs( ipk, jpk, )
            Aij, eigvecR = Aij[:, :N_PCs] , eigvecR[:N_PCs]
            
            self.Array_Aij[ipk][jpk] = Aij
            self.Array_Rij[ipk][jpk] = eigvecR
            
            if not self.onlyPCA : 
                self.Array_Prec[ipk][jpk] = Aij @ eigvecR
    

    def read_PCs(self, ):
        '''
        return the principal components and the reference power spectrum
        Note that the `Aij` is used for GP training
        '''
        Rij = [ self.Array_Rij[i][j]     for (i, j) in self.__index ]
        Aij = [ self.Array_Aij[i][j].T   for (i, j) in self.__index ]
        return np.vstack(Rij), np.vstack(Aij)
    
    
    def set_PCs(self, Rij, Aij=None, ):
        '''
        Rij : 3D array, shape ( N_{Pij}*N_{PCs}, Nz, Nk ) = (..., 540)
        '''
        self.Array_Rij  = self.__empty_list
        i0, i1 = 0, 0
        for l, (i, j) in enumerate(self.__index):
            i0, i1 = i1, i1 + self.__get_N_PCs(i, j)
            self.Array_Rij[i][j] = Rij[i0:i1]
        if i1 != Rij.shape[0] : 
            raise ValueError("The loaded Principal Components are not consistent with the current setting.")

        if Aij is not None :
            i0, i1 = 0, 0
            self.Array_Aij = self.__empty_list
            for l, (i, j) in enumerate(self.__index):
                i0, i1 = i1, i1 + self.__get_N_PCs(i, j)
                self.Array_Aij[i][j] = Aij[i0:i1].T
    
    
    def __call__(self, i, j, a_ij):
        '''
        Parameters
        ----------
        i, j : int, range [0, 6).
        a_ij : coefficients of Principal Components
        ----------
        '''
        if i > j : i, j = j, i
        return a_ij @ self.Array_Rij[i][j]


