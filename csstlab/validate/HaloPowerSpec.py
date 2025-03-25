import sys, os
import numpy as np
from scipy.optimize import minimize


## 
##  Gaussian covariance matrix for fitting halo power spectrum
##

def GaussianCovariance( pk_hh, pk_hm, pk_mm, Nk, ):
    hhhh = np.diag(pk_hh**2 /Nk)
    hhhm = np.diag(pk_hh*pk_hm /Nk)
    hhmm = np.diag(pk_hh*pk_mm /Nk)
    hmhm = np.diag(pk_hm**2 /Nk)
    return np.vstack([
            np.hstack([ 2*hhhh , 2*hhhm     , ]), 
            np.hstack([ 2*hhhm , hhmm + hmhm, ]), 
        ])


     
## 
## Halo power spectrum and the estimated covariance
## 
    

class HaloPowerSpec:
    def __init__(self, 
                PathHalo = None, 
                PathParam = None, 
                ):
        if PathHalo is None:
            PathHalo = "/Users/zhoushuren/_Projects/_24_HLPT/Data/ExtentHaloPowerSpec/Halo_PowerSpectrum.npy"
        if PathParam is None:
            PathParam = "/Users/zhoushuren/_Projects/_24_HLPT/Data//EmulatorDataSet/AllParameters.npy"
        HaloPower = np.load( PathHalo , allow_pickle=True, )[()]
        self.Cosmo0, self.Cosmo1 = 83, 129
        self.Cosmo0 -= 1    ## begin from index 1
        self.Cosmo1 -= 1
        AllParams = np.load( PathParam, allow_pickle=True)[()]["Param"]

        L = 1000
        dV_k = (2*np.pi/L)**3
        kedges = HaloPower["c0000"]["k_edges"]
        k = 0.5*(kedges[1:]+kedges[:-1])
        dk = kedges[1:] - kedges[:-1]
        Nk_cells = 4*np.pi *dk *k**2 /dV_k /2.
        
        self.HaloPower = HaloPower
        self.__params = AllParams
        self.V = L**3
        self.k = k
        self.kedges = kedges
        self.Nk_cells = Nk_cells
        self.tag = None
        self.IndexK = None
    
    def set_kmax(self, IndexK):
        self.IndexK = IndexK
        self.Nk_set = self.Nk_cells[:IndexK]
        return self.k[:IndexK]
    
    def __cosmoTag(self, icosmo):
        if icosmo > 0 : tag = "c%04d"%(icosmo+self.Cosmo0)
        else          : tag = "c0000"
        return tag
    
    
    def get_params(self, icosmo):
        if icosmo > 0 :
            return self.__params[ icosmo+self.Cosmo0 ]
        return self.__params[0]
    
    
    def Nhalo(self, icosmo, IndexZ=None, IndexMass=None):
        tag = self.__cosmoTag(icosmo)
        nhalo = self.HaloPower[tag]["Nhalo"]
        if IndexZ is None : return nhalo
        nhalo = nhalo[IndexZ]
        if IndexMass is None : return nhalo
        return nhalo[IndexMass]
    
    
    def __call__(self, icosmo, z, mass, ):
        HaloPower = self.HaloPower
        tag = self.__cosmoTag(icosmo)
        kmax = self.IndexK
        karr = self.k[:kmax]
        
        pk_shot = self.V / HaloPower[tag]["Nhalo"][z, mass]
        pk_hh = HaloPower[tag]["Pk_hh"][z, mass, :kmax]
        pk_hm = HaloPower[tag]["Pk_hm"][z, mass, :kmax]
        pk_mm = HaloPower[tag]["Pk_mm"][z, :kmax]
        ##cov = self.fpm_cov(z, karr, pk_hh, pk_hm, pk_mm, pk_shot, )
        cov = GaussianCovariance( pk_hh, pk_hm, pk_mm, self.Nk_set, )

        return karr, pk_hh, pk_hm, pk_shot, cov,







class LossFunction:
    def __init__(self, _k, 
                 _biasPk,       # stacked atuo and cross power spectrum of biased tracer
                 _Cov_hhhm,     # stacked covariance matrix 
                 _Pk_shot,      # shot noise of auto-P
                 _Pkij_list,    # basis Pk_{ij}
        ):
        self._k = _k
        self._biasPk = _biasPk

        vals, vecs = np.linalg.eigh(_Cov_hhhm)
        vals_inv = 1/vals
        vals_inv[ vals < vals.max()*1e-15 ] = 0      ## also remove negative eigenvalues
        cov_inv = vecs @ np.diag(vals_inv) @ vecs.T 
        self._Cov_inv = cov_inv
        #self._Cov_inv = np.linalg.pinv(_Cov_hhhm, rcond=1e-5 )

        self._Pk_shot = _Pk_shot
        self._Pkij_list = _Pkij_list

        Nd = _biasPk.shape[0]
        self.one_zeros = np.zeros(Nd)
        self.one_zeros[:Nd//2] = 1

        self.__alphas = 1
        self.__auto_has_ksq_shotnoise = False
        self.__auto_has_ksq_Pmm = False
        self.__cross_has_ksq_Pmm = False
        self.__klaw = None
    

    def set_auto_ksq_shotnoise(self, klaw=2, ):
        if self.__alphas == 2:
            raise ValueError("more than 2 alpha-parameter are given")
        self.__auto_has_ksq_shotnoise = True
        self.__klaw = klaw
        self.__alphas = 2
        self.__k_stack = np.hstack([self._k, self._k])
    
    def set_auto_ksq_Pmm(self, klaw=2, ):
        if self.__alphas == 2:
            raise ValueError("more than 2 alpha-parameter are given")
        self.__auto_has_ksq_Pmm = True
        self.__klaw = klaw
        self.__alphas = 2
        self.__k_stack = np.hstack([self._k, self._k])
    
    def set_cross_ksq_Pmm(self, klaw=2):
        if self.__alphas == 2:
            raise ValueError("more than 2 alpha-parameter are given")
        self.__cross_has_ksq_Pmm = True
        self.__klaw = klaw
        self.__alphas = 2


        
    def __call__(self, bias):
        '''
        if 5(6) parameters are given:
            bias = [alpha0, (alpha1), b1, b2, bs2, bn2, ]
        if 6(7) parameters are given:
            bias = [alpha0, (alpha1), b1, b2, bs2 , bn2, b3, ]
        '''
        alpha, bs = bias[:self.__alphas], bias[self.__alphas:]
        pk_auto, pk_cross = self.sum_Pkij( self._k, self._Pkij_list, *bs )
        pk_auto += alpha[0] *self._Pk_shot  # shot noise
        if self.__auto_has_ksq_shotnoise:
            pk_auto += alpha[1] *self._k**self.__klaw *self._Pk_shot
        if self.__auto_has_ksq_Pmm:
            pk_auto += alpha[1] *self._k**self.__klaw *self._Pkij_list[0]
        if self.__cross_has_ksq_Pmm:
            pk_cross += alpha[1] *self._k**self.__klaw *self._Pkij_list[0]
        
        pk_delta = self._biasPk - np.hstack([pk_auto, pk_cross,])
        val = pk_delta @ self._Cov_inv @ pk_delta
        return val
    

    def D(self, bias):
        '''
        loss function gradient
        '''
        alpha, bs = bias[:self.__alphas], bias[self.__alphas:]
        pk_auto, pk_cross = self.sum_Pkij( self._k, self._Pkij_list, *bs )
        pk_delta = self._biasPk - np.hstack([pk_auto, pk_cross,])
        dfdb_auto, dfdb_cross = self.sum_Pkij_D( self._Pkij_list, *bs )
        temp_ = self._Cov_inv @ pk_delta
        dloss = [ self._Pk_shot *self.one_zeros @temp_ ]
        if self.__alphas == 2:
            dloss.append( self._Pk_shot *(self.one_zeros *self.__k_stack**2) @temp_ )
            
        for i in range(len(bs)):
            dloss.append( 
                np.hstack([dfdb_auto[i], dfdb_cross[i],]) @temp_ 
            )
        return -2* np.array(dloss)
    
    def set_func(self, func, func_D=None, ):
        self.sum_Pkij = func
        self.sum_Pkij_D = func_D
        # Emulator.EFTofLSS_Model.Pkij_to_biasPk




def SolveEquation( _lossfunc, Nparams=5, Ntry=2):
    '''
    Multi-method and multi-try to find the global solution
    '''
    funcval = 1e10
    sol = None
    np.random.seed(0)
    for method in [ "Nelder-Mead", "CG", "Powell", "BFGS"]:
        for itry in range(Ntry):
            x0 = np.random.uniform(-2, 2, Nparams)
            retu = minimize( _lossfunc, x0 , #= (1,)*Nparams , 
                        #jac="cs" if method!="Powell" else None , #lossfunc.D 
                        method=method, tol=1e-15, 
                        options={"maxiter":6000}, 
                        )
            if funcval > retu["fun"]:
                funcval = retu["fun"] 
                sol = retu["x"]
    return funcval, sol


