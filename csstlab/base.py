import sys, os
import numpy as np
from abc import ABC, abstractmethod
from .GaussianProcess import GaussianProcessRegressor




class BaseEmulator_GP(ABC):
    def __init__(self, ):
        self.GP_kernel = None
        self.GPs = self._empty_list
        self._index =  [ (i, j) for i in range(6) for j in range(i, 6) ]
        self.__set_parameters()

    @property
    def _empty_list(self, ):
        return [ [None for i in range(6)] for j in range(6) ]
    

    # -------------------------------------------------------------------------------------------------
    # Utils for Gaussian Process fitting
    # -------------------------------------------------------------------------------------------------
    
    @abstractmethod
    def _train_emulator(self, **kwargs):
        '''
        train the emulator
        '''
        pass
    
    @abstractmethod
    def _load_emulator(self, **kwargs):
        '''
        load all the emulator data
        '''
        pass
    

    def save(self, filename, data, ) :
        '''
        save a single Numpy file
        '''
        path0 = os.path.dirname(filename) 
        if not os.path.exists(path0) : os.makedirs(path0)
        np.save( filename , np.array(data) )
    

    def load(self, filename=None, ):
        '''
        load a single Numpy file
        '''
        if not os.path.exists(filename) : 
            raise ValueError( f"The file {filename} does not exist. ")
        return np.load( filename , allow_pickle=True)[()]
    

    def _set_GPs(self, X, Ys):
        '''
        train the Gaussian Process models with the input `X` and `Ys`
            single `X` for multi-targets `Ys`
        '''
        for (i, j) in self._index : \
        self.GPs[i][j] = GaussianProcessRegressor( X, Ys[i][j], self.GP_kernel, alpha=1e-10, normalize_y=True, )
    

    def _save_GPs(self, filename, ):
        '''
        save all the trained Gaussian Process models to Numpy files
        '''
        if filename[-4:] == ".npy" : filename = filename[:-4]
        for (i, j) in self._index : \
        self.GPs[i][j].save( filename + f"_GPmodel_ij-{i}{j}.npy" )
    

    def _load_GPs(self, filename, ):
        '''
        load all the trained Gaussian Process models from Numpy files
        '''
        if filename[-4:] == ".npy" : filename = filename[:-4]
        for (i, j) in self._index :
            path_ij = filename + f"_GPmodel_ij-{i}{j}.npy" 
            if not os.path.exists(path_ij) :
                raise ValueError( f"The Gaussian Process model file {path_ij} does not exist. ")
            self.GPs[i][j] = GaussianProcessRegressor( kernel=self.GP_kernel, )
            self.GPs[i][j].load( path_ij )
        


    # -------------------------------------------------------------------------------------------------
    # Parameters rescaling
    # -------------------------------------------------------------------------------------------------
    
    def __set_parameters(self, ):
        ## The lower & upper bound of Cosmological parameters
        self.ParamRange = np.array([
            [ 0.04, 0.06 ], 
            [ 0.24, 0.40 ], 
            [ 0.6, 0.8 ], 
            [ 0.92, 1.00 ], 
            [ 1.7, 2.5 ], 
            [ -1.3, -0.7],
            [ -0.5, 0.5 ], 
            [ 0, 0.3],
        ])
        self.label_Params = [
            "$\Omega_b$",
            "$\Omega_m$",
            "$h$",
            "$n_s$",
            "$10^9 A_s$",
            "$w_0$",
            "$w_a$",
            "$M_\nu$",
        ]
        ## basis Lagrangian fields
        self.label_fields = [
            "1", 
            r"$\delta$", 
            r"$\delta^2$", 
            r"$s^2$", 
            r"$\nabla^2\delta$", 
            r"$\delta^3$", 
        ]
        ## the redshift that the snapshot stored in simulations
        self.Nz = 12
        self.Redshift = np.array([ 3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.8, 0.5, 0.25, 0.1, 0.0, ])
        self.TimeNormalized = self.NormalizeTime(self.Redshift)    # normalized time variable

    
    def NormalizeParam(self, param ):
        '''
        Normalize the cosmological parameters to [-1, 1], 
            ( Omega_b, Omega_m, h, n_s, 10^9 As, w0, w_a, M_nu, )
        --------------------------------
        param :: 8 cosmologcial parameters, not include the redshift
                1D array with shape (8)
                2D array with shape (Ncosmo, 8)
        --------------------------------
        '''
        ## attention the last dimension is casted
        return (np.array(param) - self.ParamRange[:, 0] ) /(self.ParamRange[:, 1]-self.ParamRange[:, 0]) *2 - 1


    def NormalizeParam_inv(self, paramNorm ):
        '''
        Inverse the normalization of cosmological parameters from [-1, 1] to physical scale
        see `NormalizeParam` for the details
        '''
        ## attention the last dimension is casted
        return ( np.array(paramNorm) + 1 ) *0.5 *(self.ParamRange[:, 1]-self.ParamRange[:, 0]) + self.ParamRange[:, 0]
    

    def NormalizeTime(self, z ):
        '''
        Normalize the redshift from physical scale 0 < z < 3 to [-1, 1]. 
        '''
        return 1 /(np.array(z) +1) *8./3. - 5./3.
    

    def NormalizeTime_inv(self, TimeNorm ):
        '''
        Inverse the normalization of redshift from [-1, 1] to physical scale
        see `NormalizeTime` for the details
        '''
        return 1 /(np.array(TimeNorm) + 5./3.) *3./8. -1
    


    def To_Abscissas(self, ParamsNormalied ):
        '''
        Convert the Array of cosmological parameters to the normalized abscissas
        --------------------------------
        ParamsNormalied : 1D array, shape (8). Normalized cosmological parameters.
        --------------------------------
        return : 
            2D array, shape (Nz, N_Abscissas=9)
        '''
        return np.vstack([
            np.tile( ParamsNormalied.reshape(8, 1), self.Nz, ) , 
            self.TimeNormalized
        ]).T
    

    def To_Abscissas_MultiCosmo(self, ParamsNormalied ):
        '''
        Convert the Array of cosmological parameters to the normalized abscissas
        Simplified version of `To_Abscissas` is provided for the fast response when called. 

        --------------------------------
        ParamsNormalied : 1D/2D array, shape (8) or (Ncosmo, 8). Normalized cosmological parameters.
        --------------------------------
        return : 
            2D array, shape (N_samples, N_Abscissas) = (Ncosmo*Nz, 9)
        '''
        ParamsNormalied = np.array(ParamsNormalied).reshape(-1, 8)
        Ncosmo = ParamsNormalied.shape[0]
        return np.vstack([
            np.tile( ParamsNormalied.T.reshape(8, Ncosmo, 1), self.Nz, ).reshape(8, -1) , 
            np.tile( self.TimeNormalized.reshape(-1, 1), Ncosmo ).T.reshape(-1) ,
        ]).T
    

    
    def Check_Range_Params(self, params ):
        '''
        Check if the 8 cosmological parameters are in the range of the training set
        '''
        params = np.array(params)
        if params.shape[-1] != 8 :
            raise ValueError("The shape of `params` should be (*, 8). ")
        return np.all( (params >= self.ParamRange[:, 0]) & (params <= self.ParamRange[:, 1]) )
    

    def Check_Range_z(self, z ):
        if np.any(z < 0) or np.any(z > 3) : return False
        return True
    
    
    




    

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------





class Selection_TheoreticalTemplate:
    def __init__(self):
        pass

    @staticmethod
    def Combine_TheoreticalTempate_shape66( Pk_cleft, Pk_kecl, ):
        '''
        We make use of the theoretical templates from `CLEFT` and `KECLEAN`. 
        The choices of which templates to be used for the specific `P_{ij}` are given by test. 

        Parameters
        ----------
        Pk_cleft, Pk_kecl : Ndarray, shape (... , 6, 6, Nk) 
        ----------
        '''
        if Pk_cleft.shape[-2] != 6  or Pk_kecl.shape[-2] != 6  \
        or Pk_cleft.shape[-3] != 6  or Pk_kecl.shape[-3] != 6  :
            raise ValueError("The shape of `Pk_cleft` and `Pk_kecl` should be (*, 6, 6, Nk). ")
        pk_out = Pk_cleft.copy()
        for i, j in [
            [2, 2], 
            [3, 2], 
            [4, 2], [4, 3], [4, 4],  
            [5, 3], [5, 4], [5, 5], 
        ]:
            pk_out[..., i, j, :] = Pk_kecl[..., i, j, :]
            pk_out[..., j, i, :] = pk_out [..., i, j, :]
        return pk_out


    @staticmethod
    def Combine_TheoreticalTempate_shape21( Pk_cleft, Pk_kecl, ):
        '''
        We make use of the theoretical templates from `CLEFT` and `KECLEFT`. 
        The choices of which templates to be used for the specific `P_{ij}` are given by test. 

        Parameters
        ----------
        Pk_cleft, Pk_kecl : Ndarray, shape (... , 21, Nk) 
        ----------
        '''
        if Pk_cleft.shape[-2] != 21  or Pk_kecl.shape[-2] != 21  :
            raise ValueError("The shape of `Pk_cleft` and `Pk_kecl` should be (*, 21 Nk). ")
        IndexUsingCL = [
            (2, 2), 
            (3, 2), 
            (4, 2), (4, 3), (4, 4),  
            (5, 3), (5, 4), (5, 5), 
        ]
        pk_out = Pk_cleft.copy()
        icount = -1
        for i in range(6):
            for j in range(i, 6):
                icount += 1
                if (i, j) in IndexUsingCL or (j, i) in IndexUsingCL:
                    pk_out[:, :, icount, :] = Pk_cleft[:, :, icount, :]
        return pk_out
        

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
