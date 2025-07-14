import sys, os
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interpn
from scipy.signal import savgol_filter


class FastPM_Covariance:
    def __init__(self, PathHalo=None, ):
        if PathHalo is None:
            PathHalo = "/Users/zhoushuren/_Projects/_24_HLPT/Data/ExtentHaloPowerSpec/"
        FastPMPower = np.load( PathHalo + "FastPM_PowerSpectrum.npy", allow_pickle=True, )[()]

        Pk_phaseA = FastPMPower["Pk_phaseA"]
        Pk_phaseB = FastPMPower["Pk_phaseB"]
        FPM_k  = FastPMPower["k"]
        FPM_kmodes  = FastPMPower["Nmodes"]
        Pk_stack = np.hstack([Pk_phaseA, Pk_phaseB])
        
        ## 
        ## merge 6 bins into 1 bin to reduce the noise
        ## 
        Nz = 12
        Nsamples = 50
        kend = 475      # make sure it matched with `nbin`
        nbin = 6
        modes_wei = np.sum(FPM_kmodes[1:kend].reshape(-1, nbin), axis=-1, )
        Pk_wei = Pk_stack[:, :, 1:kend] *FPM_kmodes[1:kend]
        k_wei  = FPM_k[1:kend] *FPM_kmodes[1:kend]

        Pk_wei = np.sum( Pk_wei.reshape(Nz, Nsamples, -1, nbin), axis=-1, ) / modes_wei 
        k_wei  = np.sum( k_wei .reshape(-1, nbin), axis=-1, ) / modes_wei
        
        ## 
        ## smooth the cross-coefficients
        ## 

        Nk = k_wei.shape[0]
        CorrMat_raw = np.zeros((Nz, Nk, Nk, ))
        CorrMat_smo = np.zeros((Nz, Nk, Nk, ))

        for IndexZ  in range(Nz):
            CovMat = np.cov( Pk_wei[IndexZ].T )
            CovMat[ CovMat<0 ] = 0
            CovDiag = CovMat.diagonal()
            mat_ccc = CovMat /np.sqrt(CovDiag.reshape(-1, 1) *CovDiag.reshape(1, -1) )

            iarr = np.arange( Nk )
            smat_ccc = mat_ccc.copy()
            smat_ccc[iarr[1:], iarr[1:]] = np.diag(smat_ccc, k=-1)
            smat_ccc[0, 0] = smat_ccc[0, 1]
            for iax in [0, 1, ]:
                smat_ccc = savgol_filter(smat_ccc, window_length = 13 , polyorder=1, axis=iax )

            CorrMat_raw[IndexZ] =  mat_ccc
            CorrMat_smo[IndexZ] = smat_ccc
        
        self.CorrMat_raw = CorrMat_raw
        self.CorrMat     = CorrMat_smo
        self.k_wei  = k_wei
        self.Pk_wei = Pk_wei
        self.__list_mat_r2 = None
        
    
    
    def fit(self, IndexZ, k, ):
        '''
        Parameters
        ----------
        IndexZ : int, [0, 12).
        k : Nd-array
        
        return 
        ------
        The matter power spectrum cross-coefficient at given k bins. 
        '''
            
        smat_ccc = self.CorrMat[IndexZ]
        nk = k.shape[0]
        
        mat_fit = interpn( 
            (self.k_wei, self.k_wei), 
            smat_ccc ,
            np.array( np.meshgrid(k, k, indexing="ij", ) ).reshape(2, -1).T, 
            method='nearest', 
            bounds_error=False, 
            fill_value = None, 
        ).reshape(nk, nk)
        
        mat_fit[ mat_fit<0 ] = 0
        iarr = np.arange(nk)
        mat_fit[iarr, iarr] = 1
        
        return mat_fit**2
    
    
    def set_k(self, k, ):
        self.__list_mat_r2 = [ self.fit(IndexZ, k) for IndexZ in range(12) ]
        self.Idiag = np.arange(k.shape[0])
    
    
    def __call__(self, IndexZ, 
                 k, pk_hh, pk_hm, pk_mm, shot, Nk):
        '''
        return Covariance for vector [P_hh, P_hm]. 
        The diagonal part is given by Gaussian covariance, 
            [ 2 * P_hh^2    , 2 * P_hh*P_mh   ]
            [ 2 * P_hh*P_hm , P_hh*P_mm + P_hm^2 ]
        The off-diagonal part is re-scaled by the FastPM matter power spectrum cross-coefficients, where the shot noise is not cosidered.
        Note that the number of k-cells is not divided in the output covariance. 
        '''
        if self.__list_mat_r2 is None:
            mat_r2 = self.fit(IndexZ, k)
            iarr = np.arange(k.shape[0])
        else:
            mat_r2 = self.__list_mat_r2[IndexZ]
            iarr = self.Idiag
        ## TEST ##
        ##mat_r2 = np.eye(k.shape[0])
        ##########
        pk_hh_s = pk_hh - shot
        if np.any(pk_hh_s<0):
            ival = pk_hh_s.min()
            pk_hh_s -= ival
            shot += ival
        hhhh = mat_r2 *pk_hh_s.reshape(-1, 1) *pk_hh_s.reshape(1, -1)
        hhhm = mat_r2 *pk_hh_s.reshape(-1, 1) *pk_hm.reshape(1, -1)
        hhmm = mat_r2 *pk_hh_s.reshape(-1, 1) *pk_mm.reshape(1, -1)
        hmhm = mat_r2 *pk_hm.reshape(-1, 1) *pk_hm.reshape(1, -1)
        
        hhhh[iarr, iarr] += 2*pk_hh_s*shot + shot**2 
        hhhm[iarr, iarr] += pk_hm*shot 
        hhmm[iarr, iarr] += pk_mm*shot 
        Nk_stack = np.hstack([ Nk, Nk, ])

        return np.vstack([
                np.hstack([ 2*hhhh , 2*hhhm.T   , ]), 
                np.hstack([ 2*hhhm , hhmm + hmhm, ]), 
            ]) /np.sqrt( Nk_stack.reshape(-1, 1) *Nk_stack.reshape(1, -1) )




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
        
        self.fpm_cov = FastPM_Covariance()      ##########
        self.HaloPower = HaloPower
        self.__params = AllParams
        self.V = L**3
        self.k = k
        self.kedges = kedges
        self.Nk_cells = Nk_cells
        self.tag = None
        self.IndexK = None
        self.__f_cov = 1
    
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
    
     
    def set_cov(self, opt="fpm", ):
        if opt == "fpm":
            self.__f_cov = 1
        elif opt == "gaussian":
            self.__f_cov = 2
        else:
            raise ValueError("Unknown covariance option: %s"%opt)
    

    def __call__(self, icosmo, z, mass, ):
        HaloPower = self.HaloPower
        tag = self.__cosmoTag(icosmo)
        kmax = self.IndexK
        karr = self.k[:kmax]
        
        pk_shot = self.V / HaloPower[tag]["Nhalo"][z, mass]
        pk_hh = HaloPower[tag]["Pk_hh"][z, mass, :kmax]
        pk_hm = HaloPower[tag]["Pk_hm"][z, mass, :kmax]
        pk_mm = HaloPower[tag]["Pk_mm"][z, :kmax]
        if self.__f_cov==1:
            cov = self.fpm_cov(z, karr, pk_hh, pk_hm, pk_mm, pk_shot, self.Nk_set )        #################
        elif self.__f_cov==2:
            cov = GaussianCovariance( pk_hh, pk_hm, pk_mm, self.Nk_set, )
        else:
            raise ValueError("Unknown covariance option: %s"%self.__f_cov)

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
        vals_inv[ vals < vals.max()*1e-10 ] = 0      ## also remove negative eigenvalues
        cov_inv = vecs @ np.diag(vals_inv) @ vecs.T 
        self._Cov     = _Cov_hhhm
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
        self.__cross_has_shotnoise  = False
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
    
    def set_cross_shotnoise(self):
        if self.__alphas == 2:
            raise ValueError("more than 2 alpha-parameter are given")
        self.__cross_has_shotnoise = True
        self.__alphas = 2
        
    
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
        pk_auto, pk_cross = self.sum_Pkij( self._k, self._Pkij_list, bs )
        pk_auto += alpha[0] *self._Pk_shot  # shot noise
        if self.__auto_has_ksq_shotnoise:
            pk_auto += alpha[1] *self._k**self.__klaw *self._Pk_shot
        if self.__auto_has_ksq_Pmm:
            pk_auto += alpha[1] *self._k**self.__klaw *self._Pkij_list[0]
        if self.__cross_has_shotnoise:
            pk_cross += alpha[1]   
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
        pk_auto, pk_cross = self.sum_Pkij( self._k, self._Pkij_list, bs )
        pk_delta = self._biasPk - np.hstack([pk_auto, pk_cross,])
        dfdb_auto, dfdb_cross = self.sum_Pkij_D( self._Pkij_list, bs )
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
    Attempt to find the global solution with multi-method and multi-try
    '''
    funcval = 1e10
    sol = None
    np.random.seed(0)
    for method in [ "Nelder-Mead", "CG", "Powell", "BFGS"]:
        for itry in range(Ntry):
            x0 = np.random.uniform(-2, 2, Nparams)
            retu = minimize( _lossfunc, x0 , #= (1,)*Nparams , 
                        #jac="cs" if method!="Powell" else None , #lossfunc.D 
                        method=method, tol=1e-10, 
                        options={"maxiter":20000}, 
                        )
            if funcval > retu["fun"]:
                funcval = retu["fun"] 
                sol = retu["x"]
    return funcval, sol


