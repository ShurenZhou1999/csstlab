import numpy as np
from .base import BaseEmulator_GP
from .GaussianProcess import DotProduct, Matern, \
                            ConstantKernel, RBF



class Emulator_loop(BaseEmulator_GP):
    '''
    Emulator for the 1-loop power spectrum, ( P_{11}, P_{1\delta}, ... )
    '''
    def __init__(self, kmax=1.05, N_PCs=None, opt_PCs = 1, ):
        super().__init__()
        self.GP_kernel = DotProduct(0.1)*Matern(3)
        self.PCA_T = None
        self.__kmax = kmax
        if opt_PCs == 1 :
            ## For the loop-emulator 
            N_PCs = [
                [ 20  , 20  , 20  , 40  , 20  , 20 , ], 
                [ None, 20  , 20  , 40  , 20  , 20 , ], 
                [ None, None, 20  , 40  , 40  , 40 , ], 
                [ None, None, None, 20  , 20  , 40 , ], 
                [ None, None, None, None, 20  , 40 , ], 
                [ None, None, None, None, None, 20 , ], 
            ]
        elif opt_PCs == 2 :
            # for the `Pk_ij` values in linear region
            N_PCs = [
                [ 12  , 12  , 30  , 30  , 30  , 12 , ], 
                [ None, 12  , 12  , 20  , 12  , 12 , ], 
                [ None, None, 12  , 12  , 20  , 12 , ], 
                [ None, None, None, 12  , 12  , 20 , ], 
                [ None, None, None, None, 12  , 12 , ], 
                [ None, None, None, None, None, 12 , ], 
            ]
        else :
            N_PCs = None
        self.__N_PCs = N_PCs
    
    
    def _train_emulator(self, Params, k, pk_T, 
                 to_save=False, filename=None, 
        ):
        '''
        Parameters
        ----------
        Params : Ndarray, shape ( Ncosmo,  8, ).
            Normalized cosmological parameters.
        k      : 1D array, shape (Nk, )
        pk_T   : Ndarray, shape ( Ncosmo, Nz, Pk_ii, Pk_jj, Nk)
        '''
        if Params.shape[0] != pk_T.shape[0] :
            raise ValueError("The number of cosmological parameters and the number of power spectra should be the same. ")
        if k.shape[0] != pk_T.shape[-1] :
            raise ValueError("The k-bin ranges of `k` and `pk_T` should be the same. ")
        
        self.PCA_T = PrincipalComponentAnalysis__loop( k, pk_T, kmax=self.__kmax , N_PCs=self.__N_PCs, )
        self._set_GPs( Params, self.PCA_T.Array_Aij, )
        self.k = k[self.PCA_T.krange]
        
        self.Nk = self.k.shape[0]
        self.N_training_samples = pk_T.shape[0]

        if to_save : 
            Rij, PijMean, Aij = self.PCA_T.read_PCs()
            self.save( filename, {
                "k" : self.k ,   # note that we only use k < 1.05
                "Params" : Params, 
                "Rij" : Rij,
                "PijMean" : PijMean,
                "Aij" : None , # Aij, 
                "N_training_samples" : Aij.shape[-1],
                ## we have saved the Aij in the GP-model, thus we do not need to save the training samples again.
            })
            self._save_GPs( filename )
    
    
    def _load_emulator(self, filename):
        Dload = self.load(filename)
        k, Params = Dload["k"], Dload["Params"] 
        self.k, self.Nk = k, k.shape[0]
        self.N_training_samples = Dload["N_training_samples"]
        
        self.PCA_T = PrincipalComponentAnalysis__loop( k, None, kmax=self.__kmax , N_PCs=self.__N_PCs, )
        self.PCA_T.set_PCs( Dload["Rij"], Dload["PijMean"], None )
        self._load_GPs( filename )
        #Aij = self.PCA_T.set_PCs( Dload["Rij"], Dload["PijMean"], Dload["Aij"] )
        #self._set_GPs(Params, Aij)
    
    
    def __call__(self, Param, ):
        '''
        Parameters
        ----------
        Params : Ndarray, shape ( Nparams, 8, ).
            Normalized cosmological parameters.
        ----------
        '''
        y_pred = np.zeros(( self.Nz, 6, 6, self.Nk, ))
        for i, j in self._index:
            coeffs = self.GPs[i][j].predict( Param )
            y_pred[:, i, j, :] = self.PCA_T( i, j, coeffs ) 
            y_pred[:, j, i, :] = y_pred[:, i, j, :]
        return y_pred
    






    

class PrincipalComponentAnalysis__loop:
    def __init__( self, k, Pk_T=None, 
                 kmax=1.05, 
                 onlyPCA=True, 
                 N_PCs=None, 
        ):
        '''
        Principal Components Analysis for range k < 1.05
        Parameters
        ----------
        k : 1D array, shape(Nk)
        Pk : Nd array, shape (*, 6, 6, Nk)
        N_PCs : int. 
            Number of Principal Components to keep. 
            If None, the default values are used.
        ----------
        '''
        self.__N_PCs = [
            [ 20  , 20  , 20  , 40  , 20  , 20 , ], 
            [ None, 20  , 20  , 40  , 20  , 20 , ], 
            [ None, None, 20  , 40  , 40  , 40 , ], 
            [ None, None, None, 20  , 20  , 40 , ], 
            [ None, None, None, None, 20  , 40 , ], 
            [ None, None, None, None, None, 20 , ], 
        ]
        self.onlyPCA = onlyPCA
        self.k = k
        self.kmax = kmax
        self.krange = k < self.kmax
        self.Nk = np.sum(self.krange)
        self.Nz = 12
        self.__index = [ (i, j) for i in range(6) for j in range(i, 6) ]
        if N_PCs is not None :
            if isinstance(N_PCs, int):
                for (i, j) in self.__index :
                    self.__N_PCs[i][j] = N_PCs
            elif isinstance(N_PCs, list):
                self.__N_PCs = N_PCs
        
        if Pk_T is not None:
            self.__make_PCA( Pk_T,)


    def get_N_PCs(self, i, j, ):
        if i > j : i, j = j, i
        return self.__N_PCs[i][j]
    
    @property
    def __empty_list(self, ):
        return [ [None for i in range(6)] for j in range(6) ]
    
    
    def __make_PCA(self, Pk_T, ):
        if self.k.shape[0] != Pk_T.shape[-1] :
            raise ValueError("ND-array with wrong shape" )
        krange = self.krange
        Nk, Nz = self.Nk, self.Nz
        self.Array_PijMean = self.__empty_list
        self.Array_Aij = self.__empty_list
        self.Array_Rij = self.__empty_list
        if not self.onlyPCA : 
            self.Pk_T = Pk_T
            self.Array_Prec = self.__empty_list
        
        for icount, (ipk, jpk) in enumerate(self.__index):
            pk_select = Pk_T[:, :, icount][..., krange]
            pk_select_mean = np.mean(pk_select, axis=0)
            matT = (pk_select/pk_select_mean).reshape(-1, Nk*Nz) - 1

            eigvecL, eigval, eigvecR = np.linalg.svd( matT , full_matrices=False )
            Aij = eigvecL*eigval
            n_PC = self.get_N_PCs(ipk, jpk)
            Aij, eigvecR = Aij[:, :n_PC] , eigvecR[:n_PC]
            
            self.Array_PijMean[ipk][jpk] = pk_select_mean
            self.Array_Aij[ipk][jpk] = Aij
            self.Array_Rij[ipk][jpk] = eigvecR
            
            if not self.onlyPCA : 
                pk_rec = Aij @ eigvecR
                pk_rec = pk_select_mean * ( pk_rec.reshape(pk_select.shape) + 1 )
                self.Array_Prec[ipk][jpk] = pk_rec
    
    
    def read_PCs(self, ):
        '''
        return the principal components and the reference power spectrum
        Note that the `Aij` is used for GP training
        '''
        Rij     = [ self.Array_Rij[i][j]     for (i, j) in self.__index ]
        PijMean = [ self.Array_PijMean[i][j] for (i, j) in self.__index ]
        Aij     = [ self.Array_Aij[i][j].T   for (i, j) in self.__index ]
        return np.vstack(Rij), np.array(PijMean), np.vstack(Aij)
    
    
    def set_PCs(self, Rij, PijMean, Aij=None, ):
        '''
        Rij : 3D array, shape ( N_{Pij}*N_{PCs}, Nk*Nz ) = (..., 540)
        PijMean : 3D array, shape ( N_{Pij}, Nz, Nk ) = (21, 12, 45)
        '''
        self.Array_Rij     = self.__empty_list
        self.Array_PijMean = self.__empty_list
        i0, i1 = 0, 0
        for l, (i, j) in enumerate(self.__index):
            i0, i1 = i1, i1 + self.get_N_PCs(i, j)
            self.Array_Rij[i][j]     = Rij[i0:i1]
            self.Array_PijMean[i][j] = PijMean[l]
        if i1 != Rij.shape[0] : 
            raise ValueError("The loaded Principal Components are not consistent with the current setting.")

        if Aij is not None :
            i0, i1 = 0, 0
            Array_Aij = self.__empty_list
            for l, (i, j) in enumerate(self.__index):
                i0, i1 = i1, i1 + self.get_N_PCs(i, j)
                Array_Aij[i][j] = Aij[i0:i1].T
            return Array_Aij
    
    
    def __call__(self, i, j, a_ij):
        '''
        Parameters
        ----------
        i, j : int, range [0, 6).
        a_ij : coefficients of Principal Components
        ----------
        '''
        if i > j : i, j = j, i
        pca = ( a_ij @ self.Array_Rij[i][j] ).reshape(self.Nz, self.Nk)
        pk_out = self.Array_PijMean[i][j] * ( pca + 1 )
        return pk_out











## -------------------------------------------------------------------------- 
## not longer used
## -------------------------------------------------------------------------- 



class Emulator_lin(BaseEmulator_GP):
    '''
    Emulator for linear power spectrum, P_{lin} in large scale (k<0.01)
    '''
    def __init__(self, ):
        super().__init__()
        self.GP_kernel = ConstantKernel(2)*RBF(5)
        self.__GP = None
        self.PCA_T = None
    
    def _train_emulator(self, Params, k, pk_T, 
                 to_save=False, filename=None, ):
        self.PCA_T = Pk_lin_PCA( k, pk_T, )
        self.__GP = GaussianProcessRegressor( Params, self.PCA_T.Array_Aij, 
                    self.GP_kernel, alpha=1e-12, normalize_y=True, )
        self.k = k
        self.Nk = self.k.shape[0]
        self.N_training_samples = pk_T.shape[0]
        if to_save:
            Rij, PijMean, Aij = self.PCA_T.read_PCs()
            self.save( filename, {
                "k" : self.k ,   # note that we only use k < 1.05
                "Params" : Params, 
                "Rij" : Rij,
                "PijMean" : PijMean,
                "Aij" : Aij, 
            })
            if filename[-4:] == ".npy" : filename = filename[:-4]
            self.__GP.save( filename + "_GPmodel.npy" )
        
    def _load_emulator(self, filename):
        if filename[-4:] != ".npy" : filename += ".npy"
        Dload = self.load(filename)
        k = Dload["k"]
        self.k, self.Nk = k, k.shape[0]
        self.N_training_samples = Dload["Aij"].shape[-1]
        
        self.PCA_T = Pk_lin_PCA( k, None )
        self.PCA_T.set_PCs( Dload["Rij"], Dload["PijMean"], None )

        filename = filename[:-4]
        self.__GP = GaussianProcessRegressor( kernel=self.GP_kernel, )
        self.__GP.load( filename + "_GPmodel.npy"  )
    

    def __call__(self, Param, ):
        '''
        Params : Ndarray, shape ( 8, ). Normalized cosmological parameters.
        '''
        coeffs = self.__GP.predict( Param )
        return self.PCA_T( coeffs ) 






class Pk_lin_PCA:
    def __init__( self, k, Pk_T=None, ):
        '''
        Principal Components Analysis for linear power spectrum
        Parameters
        ----------
        k : 1D array, shape(Nk)
        Pk : Nd array, shape (*, Nk)
        ----------
        '''
        self.__N_PCs = 15
        self.k = k
        self.Nz = 12
        self.Nk = k.shape[0]
        
        if Pk_T is not None:
            self.__make_PCA( Pk_T,)

    
    def __make_PCA(self, Pk_T, ):
        if self.k.shape[0] != Pk_T.shape[-1] :
            raise ValueError("ND-array with wrong shape" )
        Nk, Nz = self.Nk, self.Nz
        pk_mean = np.mean(Pk_T, axis=0)
        matT = (Pk_T/pk_mean).reshape(-1, Nk*Nz) - 1
        eigvecL, eigval, eigvecR = np.linalg.svd( matT , full_matrices=False )
        self.Array_PijMean = pk_mean
        self.Array_Aij = (eigvecL*eigval)[:, :self.__N_PCs]
        self.Array_Rij = eigvecR[:self.__N_PCs]
    
    def read_PCs(self, ):
        return self.Array_Rij, self.Array_PijMean, self.Array_Aij
    
    def set_PCs(self, Rij, PijMean, Aij=None, ):
        '''
        Rij : 3D array, shape ( N_{Pij}*N_{PCs}, Nk*Nz ) = (..., 540)
        PijMean : 3D array, shape ( N_{Pij}, Nz, Nk ) = (21, 12, 45)
        '''
        self.Array_Rij     = Rij
        self.Array_PijMean = PijMean
        if Aij is not None :
            self.Array_Aij = Aij
            return Aij
    
    def __call__(self, a_ij):
        '''
        a_ij : coefficients of Principal Components
        '''
        pca = ( a_ij @ self.Array_Rij ).reshape(self.Nz, self.Nk)
        pk_out = self.Array_PijMean * ( pca + 1 )
        return pk_out
