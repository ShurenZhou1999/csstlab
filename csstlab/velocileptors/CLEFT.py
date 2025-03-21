## This is Shi-Fan Chen's code at
##   https://github.com/sfschen/velocileptors/tree/master/velocileptors/LPT/cleft_fftw.py
##   https://github.com/sfschen/ZeNBu/blob/main/ZeNBu/zenbu.py


import os, numpy as np
from scipy.interpolate import interp1d
from .Utils import QFuncFFT, loginterp
from .Utils import SphericalBesselTransform_fftw as SphericalBesselTransform
from .Math import *


class CLEFT:
    '''
    Class to calculate power spectra up to one loop.
    Based on Chirag's code
    https://github.com/sfschen/velocileptors/blob/master/LPT/cleft_fftw.py
    
    '''

    def __init__(self, k, P_rescale, P_disp=None, 
                 jn=25, N = 4000, threads=None, extrap_min = -5, extrap_max = 3, 
                 import_wisdom=False, wisdom_file='wisdom.npy', 
                 force_high_ell=False,  
                 ):
        '''
        Parameters
        ----------
        k         : wavenumbers
        P_rescale : power spectrum to generate the inital density field
        P_disp    : power spectrum to generate the displacement field
            In principle, `P_disp` should be the same as `P_rescale`. Howover, the simulation may use slightly differet power spectrum in high redshift, so that the evolution result in low redshift matches the observation. 
            Default is `P_disp = P_rescale`
        force_high_ell : calculate the spherical bessel transforms without check the divergence, ignoring the warning. 
        '''

        
        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        self.kint = np.logspace( extrap_min, extrap_max,self.N)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        
        self.one_loop = True

        if P_disp is None : P_disp = P_rescale
        self.setup_powerspectrum(k, P_rescale, P_disp)
        
        self.pktable = None
        self.num_power_components = 21
        self.jn = jn
        self.jn_final_1loop = None    # final value of `jn`` truncated in calculation
        self.jn_final_Zel   = None
        
        if threads is None : threads = os.cpu_count()
        self.threads = threads
        #self.threads = int( os.getenv("OMP_NUM_THREADS","1") )
    
        self.import_wisdom = import_wisdom
        self.wisdom_file = wisdom_file
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads, 
                                            force_high_ell=force_high_ell,
                                            import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        

    def setup_powerspectrum(self, k, P_rescale, P_disp, ):
        # Updates the power spectrum and various q functions. Can continually compute for new cosmologies without reloading FFTW
        self.k = k
        self.P_r = P_rescale
        self.P_d = P_disp
        self.Pint_r  = loginterp(k, self.P_r, )(self.kint)
        self.Pint_d  = loginterp(k, self.P_d, )(self.kint)

        
        # This sets up terms up to one looop in the combination (symmetry factors) they appear in pk
        self.qf = QFuncFFT(self.kint, self.Pint_r, self.Pint_d, 
                           qv=self.qint, oneloop=self.one_loop,  )
        
        # linear terms
        self.Xlin = self.qf.Xlin
        self.Ylin = self.qf.Ylin
        
        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint
        
        self.Ulin = - self.qf.xi1m1
        self.corlin = self.qf.xi00

        # one loop terms: here we add in all the symmetry factors
        self.Xloop = 2 * self.qf.Xloop13 + self.qf.Xloop22; self.sigmaloop = self.Xloop[-1]
        self.Yloop = 2 * self.qf.Yloop13 + self.qf.Yloop22

        self.Vloop = 3 * (2 * self.qf.V1loop112 + self.qf.V3loop112) # this multiplies mu in the pk integral
        self.Tloop     = 3 * self.qf.Tloop112 # and this multiplies mu^3

        self.X10 = 2 * self.qf.X10loop12
        self.Y10 = 2 * self.qf.Y10loop12
        self.sigma10 = (self.X10 + self.Y10)[-1]

        self.U3 = self.qf.U3
        self.U11 = self.qf.U11
        self.U20 = self.qf.U20
        self.Us2 = self.qf.Us2
        
        # shear terms
        self.Xs2 = self.qf.Xs2
        self.Ys2 = self.qf.Ys2; self.sigmas2 = (self.Xs2 + self.Ys2)[-1]
        self.Xs4 = self.qf.Xs4
        self.Ys4 = self.qf.Ys4
        self.V = self.qf.V
        self.zeta = self.qf.zeta
        self.chi = self.qf.chi


    def p_integrals(self, k, ):
        '''
        Compute P(k) for a single k as a vector of all bias contributions.
        
        '''
        k1  = k 
        ksq = k1**2; kcu = k1**3; 
        EXP_THRESHOLD = 700
        phase = -0.5*ksq * (self.XYlin - self.sigma)
        phase[ phase > EXP_THRESHOLD ] = EXP_THRESHOLD
        expon    = np.exp(phase)
        suppress = np.exp(-0.5 * ksq *self.sigma)
        qf = self.qf
        
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components, self.N)  )
        
        #  'l'  in the summation 'n' in (B.3) of Vlah et al. 2019
        for l in range(self.jn):
            # l-dep functions
            lm1 = l-1
            mu1fac = (l>0)/(k1 * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = (1. - 2.*lm1/ksq/self.Ylin) * mu1fac # mu3 terms start at j1 so l -> l-1
            
            # (1, 1)
            bias_integrands[0,:] = 1. - 0.5 *ksq *(self.Xloop + mu2fac*self.Yloop) + kcu *(self.Vloop *mu1fac + self.Tloop *mu3fac) /6.
            # (1, \delta)
            bias_integrands[1,:] =  -k1 *mu1fac *(self.Ulin+self.U3) - 0.5* ksq*(self.X10 + self.Y10*mu2fac)
            # (\delta, \delta)
            bias_integrands[2,:] = qf.xi00 - ksq*mu2fac*self.Ulin**2 - mu1fac*k1*self.U11
            # (1, \delta^2)
            bias_integrands[3,:] = - ksq *mu2fac *self.Ulin**2 - mu1fac*k1*self.U20 
            # (\delta, \delta^2)
            bias_integrands[4,:] = -2 *k1 *mu1fac *self.Ulin * qf.xi00  
            # (\delta^2, \delta^2)
            bias_integrands[5,:] = 2 *qf.xi00**2 
            
            # (1, s^2)
            bias_integrands[6,:] = - 0.5*ksq *(self.Xs2 + mu2fac*self.Ys2) - k1 *mu1fac *self.Us2   
            # (\delta, s^2)
            bias_integrands[7,:] = - k1*mu1fac*self.V  
            # (\delta^2, s^2)
            bias_integrands[8,:] = self.chi
            # (s^2, s^2)
            bias_integrands[9,:] = self.zeta 


            ## (\nabla, 1)
            bias_integrands[10,:] = - k1 *mu1fac *qf.xi11 - ksq*(qf.A10_X_i2 + qf.A10_Y_i2*mu2fac)
            ## (\nabla, \delta)
            bias_integrands[11,:] = - qf.rxi02 + ksq *mu2fac *qf.xi1m1 *qf.xi11 - k1*mu1fac*qf.U11_i2   # minus sign in `-k^2` is included in `U11_*`
            ## (\nabla, \delta^2)
            bias_integrands[12,:] = - 2* k1 *mu1fac *qf.xi1m1 *qf.rxi02
            ## (\nabla, s^2)
            bias_integrands[13,:] =  k1 *mu1fac *(8./15.*qf.xi1m1 -0.8*qf.xi3m1) *qf.rxi22 
            ## (\nabla, \nabla)
            bias_integrands[14,:] =  qf.rxi04  - ksq *mu2fac *qf.xi11**2 - k1*mu1fac*qf.U11_i4

            
            ## (\delta^3, 1)
            bias_integrands[15,:] = - 1./6. *kcu *mu3fac *qf.xi1m1**3 + 0.5* ksq *mu2fac *qf.U20 *qf.xi1m1
            ## (\delta^3, \delta)
            bias_integrands[16,:] = - 0.5 *ksq *mu2fac *qf.xi00 *qf.xi1m1**2 - 0.5 *k1 *mu1fac *qf.U20 *qf.xi00
            ## (\delta^3, \delta^2)
            bias_integrands[17,:] = 0.5 *k1 *mu1fac *qf.xi1m1 *qf.xi00**2  
            ## (\delta^3, s^2)
            bias_integrands[18,:] = 2./3. *k1 *mu1fac *qf.xi1m1 *qf.rxi20**2   
            ## (\delta^3, \nabla)
            bias_integrands[19,:] = 0.5 *ksq *mu2fac *qf.rxi02 *qf.xi1m1**2 ##+ 0.5 *k *mu1fac *qf.U20 *qf.xi02 
            ## (\delta^3, \delta^3)
            bias_integrands[20,:] = 1./6. *qf.xi00**3
            
            #bias_integrands[-1,:] = 1 # this is the counterterm, minus a factor of k2
            
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            if bias_ffts is None: break
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
        #ret += ret[0] * zero_lags
        if l+1 != self.jn: self.jn_final_1loop = l-1
        else:              self.jn_final_1loop = l

        return 4*suppress*np.pi*ret
    


    def p_integrals_Zeldovich(self, k):
        '''
        function from: 
        https://github.com/sfschen/ZeNBu/blob/main/ZeNBu/zenbu.py
        '''
        ksq = k**2; kcu = k**3; 
        k4 = k**4 ; k5 = k**5 ; k6 = k**6
        EXP_THRESHOLD = 700
        phase = -0.5*ksq * (self.XYlin - self.sigma)
        phase[ phase > EXP_THRESHOLD ] = EXP_THRESHOLD
        expon    = np.exp(phase)
        exponm1  = np.expm1(phase)
        suppress = np.exp(-0.5 * ksq *self.sigma)
        qf = self.qf
        
        ret = np.zeros(self.num_power_components)
        bias_integrands = np.zeros((self.num_power_components,self.N))
        
        for l in range(self.jn):
            # l-dep functions
            lm1 = l-1
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = (1. - 2.*lm1/ksq/self.Ylin) * mu1fac # mu3 terms start at j1 so l -> l-1
            mu4fac = 1 - 4*l/ksq/self.Ylin + 4*l*(l-1)/(ksq*self.Ylin)**2
            mu5fac = (1 - 4*lm1/ksq/self.Ylin + 4*lm1*(lm1-1)/(ksq*self.Ylin)**2) * mu1fac
            mu6fac = 1 - 6*l/ksq/self.Ylin + 12*l*(l-1)/(ksq*self.Ylin)**2 - 8*l *(l-1) *(l-2) /(ksq*self.Ylin)**3
            
            # (1,1)
            bias_integrands[0,:] = 1
            # (1, b1)
            bias_integrands[1,:] = - k *self.Ulin *mu1fac
            # (b1, b1)
            bias_integrands[2,:] = self.corlin - ksq*mu2fac*self.Ulin**2
            # (1,b2)
            bias_integrands[3,:] = - ksq *mu2fac *self.Ulin**2
            # (b1,b2)
            bias_integrands[4,:] = -2 *k *self.Ulin *self.corlin *mu1fac + kcu*self.Ulin**3  *mu3fac
            # (b2,b2)
            bias_integrands[5,:] = 2 * self.corlin**2 - 4*ksq*self.Ulin**2*self.corlin*mu2fac + ksq**2*self.Ulin**4*mu4fac
            # (1,bs)
            bias_integrands[6,:] = -0.5 * ksq * (self.Xs2 + mu2fac*self.Ys2)
            # (b1,bs)
            bias_integrands[7,:] = -k*self.V*mu1fac + 0.5*kcu*self.Ulin*(self.Xs2*mu1fac + self.Ys2*mu3fac)
            # (b2,bs)
            bias_integrands[8,:] = self.chi - 2*ksq*self.Ulin*self.V*mu2fac \
                                      + 0.5*ksq**2 *self.Ulin**2 *(self.Xs2*mu2fac + self.Ys2*mu4fac)
            # (bs,bs)
            bias_integrands[9,:] = self.zeta - 4*ksq*(self.Xs4 + mu2fac*self.Ys4) \
                                    + 0.25*k4 * (self.Xs2**2 + 2*self.Xs2*self.Ys2*mu2fac + self.Ys2**2*mu4fac)
            
            # ---------------------------------------------------
            ## (\nabla, 1)
            bias_integrands[10,:] = - k *mu1fac *qf.xi11
            ## (\nabla, \delta)
            bias_integrands[11,:] = - qf.rxi02  + ksq *mu2fac *qf.xi1m1 *qf.xi11
            ## (\nabla, \delta^2)
            bias_integrands[12,:] = - 2* k *mu1fac *qf.xi1m1 *qf.rxi02 + kcu *mu3fac *qf.xi11 *qf.xi1m1**2
            ## (\nabla, s^2)
            bias_integrands[13,:] =  0.5*kcu*qf.xi11*(self.Xs2*mu1fac + self.Ys2*mu3fac) \
                                - 4./3. *k *mu1fac *qf.rxi22 *(0.6*qf.xi3m1 -0.4*qf.xi1m1)
            ## (\nabla, \nabla)
            bias_integrands[14,:] =  qf.rxi04  - ksq *mu2fac *qf.xi11**2

            
            ## (\delta^3, 1)
            bias_integrands[15,:] = - 1./6. *kcu *mu3fac *qf.xi1m1**3 
            ## (\delta^3, \delta)
            bias_integrands[16,:] = - 0.5 *ksq *mu2fac *qf.xi00 *qf.xi1m1**2 + 1./6. *k4 *mu4fac *qf.xi1m1**4
            ## (\delta^3, \delta^2)
            bias_integrands[17,:] = 0.5 *k*mu1fac *qf.xi1m1 *qf.xi00**2 - 0.5*kcu*mu3fac *qf.xi00 *qf.xi1m1**3  \
                                    + 1./12. *k5 *mu5fac *qf.xi1m1**5       #  Up to \delta^{12}
            ## (\delta^3, s^2)
            bias_integrands[18,:] = 2./3. *k*mu1fac *qf.xi1m1 *qf.rxi20**2   \
                                - 2./3. *kcu *mu3fac *qf.xi1m1**2 *qf.rxi20 *(0.6*qf.xi3m1 -0.4*qf.xi1m1)   \
                                + 1./12. *k5 *qf.xi1m1**3 *(self.Xs2*mu3fac + self.Ys2*mu5fac)   #  Up to \delta^{12}
            ## (\delta^3, \nabla)
            bias_integrands[19,:] = 0.5 *ksq *mu2fac *qf.rxi02 *qf.xi1m1**2 - 1./6. *k4 *mu4fac *qf.xi1m1**3 *qf.xi11 
            ## (\delta^3, \delta^3)
            bias_integrands[20,:] = 1./6. *qf.xi00**3  - 0.5 *ksq *mu2fac *qf.xi00**2 *qf.xi1m1**2    \
                                    + 0.25 *k4 *mu4fac *qf.xi00 *qf.xi1m1**4  - 1./36. *k6 *mu6fac *qf.xi1m1**6   #  Up to \delta^{12}

            # ---------------------------------------------------
            #bias_integrands[-1,:] = 1 # this is the counterterm, minus a factor of k2


            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            if bias_ffts is None: break
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)

        if l+1 != self.jn: self.jn_final_Zel = l-1
        else:              self.jn_final_Zel = l
        
        return 4*suppress*np.pi*ret
    



    def get_basis_terms(self, kmin = 1e-3, kmax = 5, nk = 200, karr=None ):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        ----------------------------------------------
        In the original formula, there are 2 kinds of difference factors :
            (1) the cross-correlation terms are defined as 
                $ 2 b_x b_y P_xy$    ->    $b_x b_y P_xy$
            (2) the $\delta^2$ field is defined as 
                $ F = b_1 \delta + {1\over 2}b_2 \delta^2 ... $
        We change the both definitions here. And note that we have include the factor of 2 in our cross-correlation definition for both new $b_\nabla$ and $b_3$ terms.
        '''

        if karr is None: kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:            kv = karr
        nk = len(kv)
        self.pktable = np.zeros([nk, self.num_power_components+1])   # one column for ks
        self.pktable[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable[foo, 1:] = self.p_integrals(kv[foo])

        x = self.pktable[:, 1:]
        # (1,   \delta,   \delta^2,   s^2,    \nabla^2\delta,   \delta^3 )
        return self.pktable[:, 0], [ 
            [ x[:, 0], x[:, 1], x[:, 3], x[:, 6], x[:, 10],  6*x[:, 15], ],
            [ None,    x[:, 2], x[:, 4], x[:, 7], x[:, 11],  6*x[:, 16], ],
            [ None,    None   , x[:, 5], x[:, 8], x[:, 12], 12*x[:, 17], ], 
            [ None,    None   , None   , x[:, 9], x[:, 13],  6*x[:, 18], ], 
            [ None,    None   , None   , None   , x[:, 14],  6*x[:, 19], ], 
            [ None,    None   , None   , None   , None    , 36*x[:, 20], ], 
        ]
    

    
    def get_basis_terms_Zeldovich(self, kmin = 1e-3, kmax = 5, nk = 200, karr=None, ):
        if karr is None: kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:            kv = karr
        nk = len(kv)
        self.pktable_Zeldovich = np.zeros([nk, self.num_power_components+1]) # one column for ks
        self.pktable_Zeldovich[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable_Zeldovich[foo, 1:] = self.p_integrals_Zeldovich(kv[foo])
        
        x = self.pktable_Zeldovich[:, 1:]
        # (1,   \delta,   \delta^2,   s^2,    \nabla^2\delta,   \delta^3 )
        return self.pktable_Zeldovich[:, 0], [ 
            [ x[:, 0], x[:, 1], x[:, 3], x[:, 6], x[:, 10],  6*x[:, 15], ],
            [ None,    x[:, 2], x[:, 4], x[:, 7], x[:, 11],  6*x[:, 16], ],
            [ None,    None   , x[:, 5], x[:, 8], x[:, 12], 12*x[:, 17], ], 
            [ None,    None   , None   , x[:, 9], x[:, 13],  6*x[:, 18], ], 
            [ None,    None   , None   , None   , x[:, 14],  6*x[:, 19], ], 
            [ None,    None   , None   , None   , None    , 36*x[:, 20], ], 
        ]
    


    ## ---------------------------------------------------------------------------------
    ## ---------------------------------------------------------------------------------

    
    def matter_power_sepctrum(self, kmin = 1e-3, kmax = 5, nk = 300, karr=None ):
        '''
        1LPT matter power spectrum
        '''
        if karr is None: kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:            kv = karr
        nk = len(kv)
        pk_mm = np.zeros((nk))
        self.sph_mm = SphericalBesselTransform(self.qint, L=self.jn, ncol=1, threads=self.threads, force_high_ell=False,)

        for foo in range(nk):
            pk_mm[foo] = self.__matter_power_sepctrum__at_k(kv[foo])
        return kv, pk_mm
    

    def __matter_power_sepctrum__at_k(self, k):
        ksq = k**2
        EXP_THRESHOLD = 700
        phase = -0.5*ksq * (self.XYlin - self.sigma)
        phase[ phase > EXP_THRESHOLD ] = EXP_THRESHOLD
        expon    = np.exp(phase)
        suppress = np.exp(-0.5 * ksq *self.sigma)
        
        ret = np.zeros(1)
        bias_integrands = np.zeros((1,self.N))
        
        for l in range(self.jn):
            bias_integrands[0,:] = 1

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_mm.sph(l, bias_integrands)
            if bias_ffts is None: break
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
        
        return 4*suppress*np.pi*ret