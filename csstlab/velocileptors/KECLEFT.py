## This is Shi-Fan Chen's code at
##   https://github.com/sfschen/velocileptors/tree/master/velocileptors/LPT/cleft_fftw.py


import os, numpy as np
from scipy.interpolate import interp1d
from .Utils import QFuncFFT, loginterp
from .Utils import SphericalBesselTransform_fftw as SphericalBesselTransform
from .Math import *



class KECLEFT:
    def __init__(self, k, P_rescale, P_disp=None, 
                 N = 6000, threads=None, extrap_min = -5, extrap_max = 3, 
                 import_wisdom=False, wisdom_file='wisdom.npy', 
                 force_high_ell=False, 
                 ):

        
        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        self.kint = np.logspace( extrap_min, extrap_max,self.N)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)

        if P_disp is None : P_disp = P_rescale
        self.setup_powerspectrum(k, P_rescale, P_disp)
        
        self.pktable = None
        self.num_power_components = 21
        self.jn = 4
        
        if threads is None : threads = os.cpu_count()
        self.threads = threads
        #self.threads = int( os.getenv("OMP_NUM_THREADS","1") )
    
        self.import_wisdom = import_wisdom
        self.wisdom_file = wisdom_file
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads, 
                                            force_high_ell=force_high_ell,
                                            import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        self.integral_mu = Integral_mun_Pell()
        

    def setup_powerspectrum(self, k, P_rescale, P_disp, ):
        # Updates the power spectrum and various q functions. Can continually compute for new cosmologies without reloading FFTW
        self.k = k
        self.P_r = P_rescale
        self.P_d = P_disp
        self.Pint_r  = loginterp(k, self.P_r, )(self.kint)
        self.Pint_d  = loginterp(k, self.P_d, )(self.kint)

        
        # This sets up terms up to one looop in the combination (symmetry factors) they appear in pk
        self.qf = QFuncFFT(self.kint, self.Pint_r, self.Pint_d, 
                           qv=self.qint, oneloop=True,  )
        
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
        self.Tloop = 3 * self.qf.Tloop112 # and this multiplies mu^3

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

        ##self.Ub3 = self.qf.Ub3
        ##self.theta = self.qf.theta



    def p_integrals(self, k, ):
        '''
        Compute P(k) for a single k as a vector of all bias contributions.
        
        '''
        k1  = k 
        ksq = k1**2; kcu = k1**3; k4 = k1**4
        k5  = k1**5; k6  = k1**6
        qf = self.qf
        
        ret = np.zeros(self.num_power_components)
        bias_integrands = np.zeros( (self.num_power_components, self.N)  )
        
        for l in range(self.jn):
            # l-dep functions
            mu0fac = self.integral_mu(l, 0)
            mu1fac = self.integral_mu(l, 1)
            mu2fac = self.integral_mu(l, 2)
            mu3fac = self.integral_mu(l, 3)
            mu4fac = self.integral_mu(l, 4)
            mu5fac = self.integral_mu(l, 5)
            mu6fac = self.integral_mu(l, 6)
            # leading order terms of `\mu^n \exp(-1/2 k_ik_j A_{ij})`
            mu0fac_Exp = mu0fac - 0.5 *ksq *( mu0fac*qf.Xlin + mu2fac*qf.Ylin) 
            mu1fac_Exp = mu1fac - 0.5 *ksq *( mu1fac*qf.Xlin + mu3fac*qf.Ylin) 
            mu2fac_Exp = mu2fac - 0.5 *ksq *( mu2fac*qf.Xlin + mu4fac*qf.Ylin) 
            mu3fac_Exp = mu3fac - 0.5 *ksq *( mu3fac*qf.Xlin + mu5fac*qf.Ylin)

            # (1, 1)
            bias_integrands[0,:] =  mu0fac_Exp - 0.5 *ksq *( mu0fac*self.Xloop + mu2fac*self.Yloop )  \
                                + kcu *(self.Vloop *mu1fac + self.Tloop *mu3fac) /6.  \
                                + 1./8. *k4 *( mu0fac*qf.Xlin**2 + 2*mu2fac*qf.Xlin*qf.Ylin + mu4fac*qf.Ylin**2 )    \
                                ## Further terms introduce the significant unstable behavior
                                # - 1./48.*k6 *( mu0fac*qf.Xlin**3 + 3* mu2fac*qf.Xlin**2*qf.Ylin + 3* mu4fac*qf.Xlin*qf.Ylin**2 + mu6fac*qf.Ylin**3 )
            # (1, \delta)
            bias_integrands[1,:] =  - k1 *mu1fac *(self.Ulin+self.U3) - 0.5* ksq*(mu0fac*self.X10 + self.Y10*mu2fac)  \
                                    - k1 *(mu1fac_Exp -mu1fac) *self.Ulin
            # (\delta, \delta)
            bias_integrands[2,:] = mu0fac_Exp *qf.xi00  - ksq*mu2fac*self.Ulin**2 - mu1fac*k1*self.U11
            # (1, \delta^2)
            bias_integrands[3,:] = - ksq *mu2fac *self.Ulin**2 - mu1fac*k1*self.U20   ## additional terms (in `mu1fac_Exp`)  -->  seems not improvement
            # (\delta, \delta^2)
            bias_integrands[4,:] = - 2 *k1 *mu1fac *self.Ulin * qf.xi00    \
                                    - kcu *mu3fac *qf.xi1m1**3     ## additional terms (in this line and `mu1fac_Exp`)
            # (\delta^2, \delta^2)
            bias_integrands[5,:] = 2 *qf.xi00**2 *mu0fac_Exp    ## additional terms (in `mu0fac_Exp`) ; more terms from higher order `\delta` do NOT help
            
            # (1, s^2)
            bias_integrands[6,:] = - 0.5*ksq *(mu0fac_Exp *self.Xs2 + mu2fac_Exp*self.Ys2) - k1 *mu1fac *self.Us2    # additional terms (in `mu0fac_Exp` and `mu2fac_Exp`)
            # (\delta, s^2)
            bias_integrands[7,:] = - k1 *mu1fac *self.V  \
                                    + 0.5* kcu*self.Ulin*(self.Xs2*mu1fac + self.Ys2*mu3fac)  # additional terms (in `mu1fac_Exp` and `mu3fac_Exp`)
            # (\delta^2, s^2)
            bias_integrands[8,:] = self.chi *mu0fac     ## plus the additional terms (in `mu0fac`) does NOT help
            # (s^2, s^2)
            bias_integrands[9,:] = self.zeta *mu0fac_Exp     ## additional terms (in `mu0fac_Exp`)


            ## (\nabla, 1)
            bias_integrands[10,:] = - k1 *mu1fac_Exp *qf.xi11 - ksq*(mu0fac*qf.A10_X_i2 + mu2fac*qf.A10_Y_i2)  ## additional terms (in `mu1fac_Exp`) helps
            ## (\nabla, \delta)
            bias_integrands[11,:] = - mu0fac_Exp *qf.rxi02 + ksq *mu2fac *qf.xi1m1 *qf.xi11 - k1*mu1fac*qf.U11_i2   ##  minus sign in `-k^2` is included in `U11_*` ; additional terms (in `mu0fac_Exp`) helps
            ## (\nabla, \delta^2)
            bias_integrands[12,:] = - 2* k1 *mu1fac *qf.xi1m1 *qf.rxi02
            ## (\nabla, s^2)
            bias_integrands[13,:] =  k1 *mu1fac *(8./15.*qf.xi1m1 -0.8*qf.xi3m1) *qf.rxi22     ##  more terms from higher order `\delta` do NOT help
            ## (\nabla, \nabla)
            bias_integrands[14,:] =  mu0fac_Exp *qf.rxi04  - ksq *mu2fac *qf.xi11**2 - k1*mu1fac*qf.U11_i4

            
            ## (\delta^3, 1)
            bias_integrands[15,:] = - 1./6. *kcu *mu3fac *qf.xi1m1**3 + 0.5* ksq *mu2fac *qf.U20 *qf.xi1m1
            ## (\delta^3, \delta)
            bias_integrands[16,:] = - 0.5 *ksq *mu2fac *qf.xi00 *qf.xi1m1**2 - 0.5 *k1 *mu1fac *qf.U20 *qf.xi00
            ## (\delta^3, \delta^2)
            bias_integrands[17,:] = 0.5 *k1 *mu1fac *qf.xi1m1 *qf.xi00**2  
            ## (\delta^3, s^2)
            bias_integrands[18,:] = 2./3. *k1 *mu1fac *qf.xi1m1 *qf.rxi20**2    \
                                    - 2./3. *kcu *mu3fac *qf.xi1m1**2 *qf.rxi20 *(0.6*qf.xi3m1 -0.4*qf.xi1m1)   \
                                    + 1./12. *k5 *qf.xi1m1**3 *(self.Xs2*mu3fac + self.Ys2*mu5fac)  
            ## (\delta^3, \nabla)
            bias_integrands[19,:] = 0.5 *ksq *mu2fac *qf.rxi02 *qf.xi1m1**2   \
                                    # + 0.5 *k *mu1fac *qf.U20 *qf.xi02   - 1./6. *k4 *mu4fac *qf.xi1m1**3 *qf.xi11
            ## (\delta^3, \delta^3)
            bias_integrands[20,:] = 1./6. *qf.xi00**3 *mu0fac_Exp  \
                                    ## the following additive terms do not work 
                                    #- 0.5 *ksq *mu2fac *qf.xi00**2 *qf.xi1m1**2    \  
                                    #+ 0.25 *k4 *mu4fac *qf.xi00 *qf.xi1m1**4  - 1./36. *k6 *mu6fac *qf.xi1m1**6 
            

            if l >= 0 : bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            if bias_ffts is None: break
            ret +=  interp1d(ktemps, bias_ffts)(k)

        return 4*np.pi *ret 
    


    def get_basis_terms(self, kmin = 1e-3, kmax = 5, nk = 200, karr=None ):

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


    def export_wisdom(self, wisdom_file='./wisdom.npy'):
        self.sph.export_wisdom(wisdom_file=wisdom_file)
