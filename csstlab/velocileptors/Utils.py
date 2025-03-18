## This is Shi-Fan Chen's code at
##   https://github.com/sfschen/velocileptors/tree/master/velocileptors/LPT


import numpy as np
import pyfftw
import pickle
from scipy.special import loggamma
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
import inspect

ALMOST_ZEROS_VALUE = 1e-300
ALMOST_INF_VALUE = 1e300



# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------



class loginterp:
    def __init__(self, x, y):
        '''
        unknown numerical issues arise when direct interpolating the function, 
        thus we force the value outside the range to be zero.
        '''
        NonZero = np.where( y>1e-15 )
        self.x = x[NonZero]
        self.y = y[NonZero]
        logx = np.log(self.x)
        logy = np.log(self.y)
        self.logx_min = logx.min()
        self.logx_max = logx.max()
        self.LogFunc = interpolate( logx, logy, k = 5, ext="zeros", )
    
    def __call__(self, x):
        logx = np.log(x)
        y = np.exp(self.LogFunc(logx))
        y[ (logx<self.logx_min) | (logx>self.logx_max) ] = 0
        return y




def loginterp__(x, y, yint = None, side = "both", lorder = 9, rorder = 9, lp = 1, rp = -2,
              ldx = 1e-6, rdx = 1e-6,\
              interp_min = -12, interp_max = 12, Nint = 10**5, verbose=False, option='B'):
    '''
    Extrapolate function by evaluating a log-index of left & right side.
    
    From Chirag Modi's CLEFT code at
    https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
    
    The warning for divergent power laws on both ends is turned off. To turn back on uncomment lines 26-33.
    '''
    
    if yint is None:
        yint = interpolate(x, y, k = 5, ext="zeros", )
    if side == "both":
        side = "lr"
    
    # Make sure there is no zero crossing between the edge points
    # If so assume there can't be another crossing nearby
    
    if np.sign(y[lp]) == np.sign(y[lp-1]) and np.sign(y[lp]) == np.sign(y[lp+1]):
        l = lp
    else:
        l = lp + 2
        
    if np.sign(y[rp]) == np.sign(y[rp-1]) and np.sign(y[rp]) == np.sign(y[rp+1]):
        r = rp
    else:
        r = rp - 2
    
    lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
    rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
    #if np.abs(y[l])<ALMOST_ZEROS_VALUE: 
    #    lneff, rneff = 0, 0
    #else :
    #    lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder) *x[l]/y[l]
    #    rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder) *x[r]/y[r]
    
    #print(lneff, rneff)
    
    # uncomment if you like warnings.
    #if verbose:
    #    if lneff < 0:
    #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
    #        print('WARNING: Runaway index on left side, bad interpolation. Left index = %0.3e at %0.3e'%(lneff, x[l]))
    #    if rneff > 0:
    #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
    #        print('WARNING: Runaway index on right side, bad interpolation. Reft index = %0.3e at %0.3e'%(rneff, x[r]))

    if option == 'A':
    
        xl = np.logspace(interp_min, np.log10(x[l]), Nint)
        xr = np.logspace(np.log10(x[r]), interp_max, Nint)
        yl = y[l]*(xl/x[l])**lneff
        yr = y[r]*(xr/x[r])**rneff
        #print(xr/x[r])

        xint = x[l+1:r].copy()
        yint = y[l+1:r].copy()
        if side.find("l") > -1:
            xint = np.concatenate((xl, xint))
            yint = np.concatenate((yl, yint))
        if side.find("r") > -1:
            xint = np.concatenate((xint, xr))
            yint = np.concatenate((yint, yr))
        yint2 = interpolate(xint, yint, k = 5, ext=3)
    
    else:
        # nan_to_numb is to prevent (xx/x[l/r])^lneff to go to nan on the other side
        # since this value should be zero on the wrong side anyway
        #yint2 = lambda xx: (xx <= x[l]) * y[l]*(xx/x[l])**lneff \
        #                 + (xx >= x[r]) * y[r]*(xx/x[r])**rneff \
        #                 + (xx > x[l]) * (xx < x[r]) * interpolate(x, y, k = 5, ext=3)(xx)
        yint2 = lambda xx:   (xx <= x[l]) * y[l]* np.nan_to_num((xx/x[l])**lneff) \
                   + (xx >= x[r]) * y[r]* np.nan_to_num((xx/x[r])**rneff) \
                   + (xx > x[l]) * (xx < x[r]) * interpolate(x, y, k = 5, ext=3)(xx)

    return yint2







# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

class SphericalBesselTransform:

    def __init__(self, qs, L=15, low_ring=True, fourier=False):
    
        '''
        Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
        the untransformed coordinate, up to a given order L in bessel functions (j_l for l
        less than or equal to L. The point is to save time by evaluating the Mellin transforms
        u_m in advance.
        
        Does not use fftw as in spherical_bessel_transform_fftw.py, which makes it convenient
        to evaluate the generalized correlation functions in qfuncfft, as there aren't as many
        ffts as in LPT modules so time saved by fftw is minimal when accounting for the
        startup time of pyFFTW.
        
        Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
        with the above modifications.
        
        '''

        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2*np.pi**2)
        
        self.q = qs
        self.L = L
        
        self.Nx = len(qs)
        self.Delta = np.log(qs[-1]/qs[0])/(self.Nx-1)

        self.N = 2**(int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        self.pads = np.zeros( (self.N-self.Nx)//2  )
        self.pad_iis = np.arange(self.Npad - self.Npad//2, self.N - self.Npad//2)
        
        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = np.arange(0, self.N//2+1)
        self.ydict = {}; self.udict = {}; self.qdict= {}
        
        if low_ring:
            for ll in range(L):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta/np.pi * np.angle(self.UK(ll,q+1j*np.pi/self.Delta)) #ln(xmin*ymax)
                ys = np.exp( lnxy - self.Delta) * qs/ (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) \
                        * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q
        
        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(L):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q



    
    def sph(self, nu, fq):
        '''
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        '''
        q = self.qdict[nu]; y = self.ydict[nu]
        f = np.concatenate( (self.pads,self.q**(3-q)*fq,self.pads) )
        
        fks = np.fft.rfft(f)
        gks = self.udict[nu] * fks
        gs = np.fft.hfft(gks) / self.N

        return y, y**(-q) * gs[self.pad_iis]
    
    

    def UK(self, nu, z):
        '''
        The Mellin transform of the spherical bessel transform.
        '''
        return self.sqrtpi * np.exp(np.log(2)*(z-2) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))

    def update_tilt(self,nu,tilt):
        '''
        Update the tilt for a particular nu. Assume low ring coordinates.
        '''
        q = tilt; ll = nu
        
        ms = np.arange(0, self.N//2+1)
        lnxy = self.Delta/np.pi * np.angle(self.UK(ll,q+1j*np.pi/self.Delta)) #ln(xmin*ymax)
        ys = np.exp( lnxy - self.Delta) * self.q/ (self.q[0]*self.q[-1])
        us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) \
                * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)
                
        self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q





# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
        


class SphericalBesselTransform_fftw:
    '''
    Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
    the untransformed coordinate, up to a given order L in bessel functions (j_l for l
    less than or equal to L. The point is to save time by evaluating the Mellin transforms
    u_m in advance.
    
    Uses pyfftw, which can perform multiple (ncol) Fourier transforms at once, one for
    each bias contribution.
    
    Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
    with the above modifications.
    
    '''

    def __init__(self, qs, L=15, ncol = 1, low_ring=True, fourier=False, threads=1, 
                 force_high_ell = True, 
                 import_wisdom=False, wisdom_file='./fftw_wisdom.npy'):

        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2*np.pi**2)
        
        self.q = qs
        self.L = L
        self.ncol = ncol
        
        self.Nx = len(qs)
        self.Delta = np.log(qs[-1]/qs[0])/(self.Nx-1)
        
        # zero pad the arrays to the preferred length format for ffts, 2^N
        self.N = 2**(int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        self.ii_l = self.Npad - self.Npad//2 # left and right indices sandwiching the padding
        self.ii_r = self.N - self.Npad//2
        
        # Set up FFTW objects:
        if import_wisdom:
            pyfftw.import_wisdom(tuple(np.load(wisdom_file)))
        
        #flags = ('FFTW_DESTROY_INPUT','FFTW_MEASURE')
        
        self.force_high_ell = force_high_ell   # Shuren: set as `False` to stop at too large value at high ell

        self.fks = pyfftw.empty_aligned((self.ncol,self.N//2 + 1), dtype='complex128')
        self.fs  = pyfftw.empty_aligned((self.ncol,self.N), dtype='float64')
        
        self.gks = pyfftw.empty_aligned((self.ncol,self.N//2 + 1), dtype='complex128')
        self.gs  = pyfftw.empty_aligned((self.ncol,self.N), dtype='float64')
        
        pyfftw.config.NUM_THREADS = threads
        self.fft_object = pyfftw.FFTW(self.fs, self.fks, direction='FFTW_FORWARD',threads=threads)
        self.ifft_object = pyfftw.FFTW(self.gks, self.gs, direction='FFTW_BACKWARD',threads=threads)
        
        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = np.arange(0, self.N//2+1)
        self.ydict = {}; self.udict = {}; self.qdict= {}
        
        if low_ring:
            for ll in range(L):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta/np.pi * np.angle(self.UK(ll,q+1j*np.pi/self.Delta)) #ln(xmin*ymax)
                ys = np.exp( lnxy - self.Delta) * qs/ (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) \
                        * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)
                us[self.N//2] = us[self.N//2].real # manually impose low ring

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q
        
        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(L):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)
                us[self.N//2] = us[self.N//2].real # manually impose low ring

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q


    def export_wisdom(self, wisdom_file='./fftw_wisdom.npy'):
        np.save(wisdom_file, pyfftw.export_wisdom())
    
    def sph(self, nu, fq):
        '''
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        '''
        q = self.qdict[nu]; y = self.ydict[nu]
        self.fs[:] = 0 # on NERSC this seems necessary or this variable spills over from previous calls
        self.fs[:,self.Npad - self.Npad//2 : self.N - self.Npad//2] = fq * self.q**(3-q)
        
        fks = self.fft_object()
        if not self.force_high_ell:
            ## Shuren: the elements in `fks` array are at similar order of magnitude; check one element only is enough
            if np.abs(fks[0, 0]) > ALMOST_INF_VALUE or np.abs(fks[-1, -1]) > ALMOST_INF_VALUE:
                return y, None
                #raise ValueError('Infinite values encountered due to too high $ell$-modes required. Reduce the order!')
        self.gks[:] = np.conj(fks * self.udict[nu])
        gs = self.ifft_object()

        return y, gs[:,self.ii_l:self.ii_r] * y**(-q)
    
    

    def UK(self, nu, z):
        '''
        The Mellin transform of the spherical bessel transform.
        '''
        return self.sqrtpi * np.exp(np.log(2)*(z-2) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))









# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------


class QFuncFFT:
    '''
       Class to calculate all the functions of q, X(q), Y(q), U(q), xi(q) etc.
       as well as the one-loop terms Q_n(k), R_n(k) using FFTLog.
       
       Throughout we use the ``generalized correlation function'' notation of 1603.04405.
       
       This is modified to take an IR scale kIR
              
       Note that one should always cut off the input power spectrum above some scale.
       I use exp(- (k/20)^2 ) but a cutoff at scales twice smaller works equivalently,
       and probably beyond that. The important thing is to keep all integrals finite.
       This is done automatically in the Zeldovich class.
       
       Currently using the numpy version of fft. The FFTW takes longer to start up and
       the resulting speedup is unnecessary in this case.
       
    '''
    def __init__(self, k, P_r, P_d, P_x=None, kIR = None, qv = None, oneloop = False, low_ring=True, ):
        '''
        P_r :: the rescaled powe spectrum to seed the initial LPT fields
        P_d :: the power spectrum that matches the displacement field
        '''

        self.oneloop = oneloop
        self.k = k
        self.P_r = P_r
        self.P_d = P_d
        if P_x is None:
            self.P_x = np.sqrt(P_r*P_d)
            self.P_x[ np.isnan(self.P_x) ] = 0
        else:
            self.P_x = P_x
        
        if kIR is not None:
            self.ir_less = np.exp(- (self.k/kIR)**2 )
            self.ir_greater = -np.expm1(- (self.k/kIR)**2)
        else:
            self.ir_less = 1
            self.ir_greater = 0

        if qv is None:
            self.qv = np.logspace(-5,5,2e4)
        else:
            self.qv = qv
        
        self.sph = SphericalBesselTransform(self.k, L=5, low_ring=True, fourier=True)
        self.sphr = SphericalBesselTransform(self.qv, L=5, low_ring=True, fourier=False)

        
        self.setup_xiln()
        self.setup_2pts()
        self.setup_shear()
        
        if self.oneloop:
            self.setup_QR()
            self.setup_oneloop_2pts()
            #self.setup_third_order()
    


    def setup_xiln(self):
        
        # Compute a bunch of generalized correlation functions

        ## 
        ## <\delta_{ini}\delta_{disp}>$
        ## 
        self.xi00 = self.rxi_l_n(0, 0)      # linear xi00 always comes from the initial field
        self.xi0m2 = self.xi_l_n(0,-2, side='right') # since this approaches constant on the left only interpolate on right
        self.xi1m1 = self.xi_l_n(1,-1)
        self.xi2m2 = self.xi_l_n(2,-2)
        self.xi3m1 = self.xi_l_n(3,-1)
        
        # Also compute the IR-cut lm2's
        #self.xi0m2_lt = self.xi_l_n(0,-2, IR_cut = 'lt', side='right')
        #self.xi2m2_lt = self.xi_l_n(2,-2, IR_cut = 'lt')
        #self.xi0m2_gt = self.xi_l_n(0,-2, IR_cut = 'gt', side='right')
        #self.xi2m2_gt = self.xi_l_n(2,-2, IR_cut = 'gt')
        self.xi02 = self.xi_l_n(0,2)
        self.xi04 = self.xi_l_n(0,4)
        self.xi11 = self.xi_l_n(1,1)
        self.xi15 = self.xi_l_n(1,5)
        self.xi13 = self.xi_l_n(1,3)
        self.xi20 = self.xi_l_n(2,0)
        self.xi24 = self.xi_l_n(2,4)
        self.xi22 = self.xi_l_n(2,2)
        self.xi31 = self.xi_l_n(3,1)
        self.xi33 = self.xi_l_n(3,3)
        self.xi35 = self.xi_l_n(3,5)
        self.xi40 = self.xi_l_n(4,0)
        self.xi42 = self.xi_l_n(4,2)
        self.xi44 = self.xi_l_n(4,4)

        

        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        ## 
        ## <\delta_{disp}\delta_{disp}>$
        ## 
        self.dxi0m2 = self.dxi_l_n(0,-2, side='right') # since this approaches constant on the left only interpolate on right
        self.dxi1m1 = self.dxi_l_n(1,-1)
        self.dxi2m2 = self.dxi_l_n(2,-2)
        self.dxi3m1 = self.dxi_l_n(3,-1)
        
        self.dxi00 = self.dxi_l_n(0,0)
        self.dxi02 = self.dxi_l_n(0,2)
        self.dxi11 = self.dxi_l_n(1,1)
        self.dxi20 = self.dxi_l_n(2,0)
        self.dxi22 = self.dxi_l_n(2,2)
        self.dxi31 = self.dxi_l_n(3,1)
        self.dxi33 = self.dxi_l_n(3,3)
        self.dxi40 = self.dxi_l_n(4,0)

        ## 
        ## <\delta_{ini}\delta_{ini}>$
        ## 
        self.rxi00 = self.rxi_l_n(0,0)
        self.rxi02 = self.rxi_l_n(0,2)
        self.rxi04 = self.rxi_l_n(0,4)
        self.rxi11 = self.rxi_l_n(1,1)
        self.rxi20 = self.rxi_l_n(2,0)
        self.rxi22 = self.rxi_l_n(2,2)
        self.rxi31 = self.rxi_l_n(3,1)
        self.rxi40 = self.rxi_l_n(4,0)


        


    
    def setup_QR(self):
    
        # Computes Q_i(k), R_i(k)-- technically will want them transformed again!
        # then lump together into the kernels and reverse fourier

        Qfac = 4 * np.pi 
        ##_integrand_Q1 = Qfac * (8./15 * self.xi00**2 - 16./21 * self.xi20**2 + 8./35 * self.xi40**2)
        ##_integrand_Q2 = Qfac * (4./5 * self.xi00**2 - 4./7 * self.xi20**2 - 8./35 * self.xi40**2 \
        ##                        - 4./5 * self.xi11*self.xi1m1 + 4/5 * self.xi31*self.xi3m1)
        ##_integrand_Q3 = Qfac * (38./15 * self.xi00**2 + 2./3*self.xi02*self.xi0m2 \
        ##                        - 32./5*self.xi1m1*self.xi11 + 68./21*self.xi20**2 \
        ##                        + 4./3 * self.xi22*self.xi2m2 - 8./5 * self.xi31*self.xi3m1 + 8./35*self.xi40**2)
        ##_integrand_Q5 = Qfac * (2./3 * self.xi00**2 - 2./3*self.xi20**2 \
        ##                        - 2./5 * self.xi11*self.xi1m1 + 2./5 * self.xi31*self.xi3m1)
        ##_integrand_Q8 = Qfac * (2./3 * self.xi00**2 - 2./3*self.xi20**2)
        ##_integrand_Qs2 = Qfac * (-4./15 * self.xi00**2 + 20./21*self.xi20**2 - 24./35*self.xi40**2)

        ##Xk_1 = self.template_QR(0,self.xi00/self.qv)
        ##Xk_2 = self.template_QR(2,self.xi20/self.qv)
        ##Xk_3 = self.template_QR(1,self.xi11/self.qv)
        ##Xk_4 = self.template_QR(3,self.xi31/self.qv)
        ##self.XPk   = self.p * ( self.k* 2./3.*Xk_1 - self.k* 2./3.*Xk_2 - 2./5.*Xk_3 + 2./5.*Xk_4 )
        ##Xk_1 = self.template_QR(0,self.xi02/self.qv)
        ##Xk_2 = self.template_QR(2,self.xi22/self.qv)
        ##Xk_3 = self.template_QR(1,self.xi13/self.qv)
        ##Xk_4 = self.template_QR(3,self.xi33/self.qv)
        ##self.XPk_i2 = self.p * ( self.k* 2./3.*Xk_1 - self.k* 2./3.*Xk_2 - 2./5.*Xk_3 + 2./5.*Xk_4 )


        ## --------------------------------------------------------------------------
        ## --------------------------------------------------------------------------
        ##
        ## Q1, Q2, R1, R2  $\propto  <\delta_{disp}\delta_{disp}>^2 $ 
        ## --> for A_{ij} and W_{ijk} calculation
        ## Q5, Q8, Qs2 for others
        ##
        _integrand_Q1 = Qfac * (8./15 * self.dxi00**2 - 16./21 * self.dxi20**2 + 8./35 * self.dxi40**2)
        _integrand_Q2 = Qfac * (4./5  * self.dxi00**2 -  4./7 * self.dxi20**2  - 8./35 * self.dxi40**2 \
                               - 4./5 * self.dxi11*self.dxi1m1 + 4/5 * self.dxi31*self.dxi3m1)
        _integrand_Q5 = Qfac * (2./3 * self.xi00*self.dxi00  - 2./3* self.xi20*self.dxi20 \
                              - 2./5 * self.xi11*self.dxi1m1 + 2./5* self.xi31*self.dxi3m1 )
        _integrand_Q8 = Qfac * (2./3 * self.xi00**2 - 2./3*self.xi20**2)
        _integrand_Qs2 = Qfac * (-4./15 * self.xi00**2 + 20./21*self.xi20**2 - 24./35*self.xi40**2)

        self.Q1 = self.template_QR(0, _integrand_Q1)
        self.Q2 = self.template_QR(0, _integrand_Q2)
        self.Q5 = self.template_QR(0, _integrand_Q5)
        self.Q8 = self.template_QR(0, _integrand_Q8)
        self.Qs2 = self.template_QR(0, _integrand_Qs2)

        _integrand_Q5 = Qfac * (2./3 *self.xi02*self.dxi00  - 2./3* self.xi22*self.dxi20 \
                              - 2./5 *self.xi13*self.dxi1m1 + 2./5* self.xi33*self.dxi3m1 )
        self.Q5_i2 = self.template_QR(0, _integrand_Q5)

        R1_0 = self.template_QR(0, self.dxi00/self.qv )
        R1_2 = self.template_QR(2, self.dxi20/self.qv )
        R1_4 = self.template_QR(4, self.dxi40/self.qv )
        R2_1 = self.template_QR(1, self.dxi1m1/self.qv )
        R2_3 = self.template_QR(3, self.dxi3m1/self.qv )
        self.R1 = self.k**2 *self.P_d * ( 8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.R2 = self.k**2 *self.P_d * (-2./15 * R1_0 -  2./21* R1_2 + 8./35 * R1_4 + self.k * 2./5*R2_1 - self.k* 2./5*R2_3)


        ## --> for U3, A^{10} calculation  $\propto  <\delta_{disp}\delta_{disp}\delta_{disp}\delta_{ini}> $
        self.xdR1 = self.k**2 *self.P_x * ( 8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.xdR2 = self.k**2 *self.P_x * (-2./15 * R1_0 -  2./21* R1_2 + 8./35 * R1_4 + self.k * 2./5*R2_1 - self.k* 2./5*R2_3)


        
        ## --------------------------------------------------------------------------
        ## --------------------------------------------------------------------------
        ##
        ## $\propto  <\delta_{ini}\delta_{disp}>^2$
        ## --> for U^{11} and U^{20} calculation
        ##
        R1_0 = self.template_QR(0, self.xi00/self.qv )
        R1_2 = self.template_QR(2, self.xi20/self.qv )
        R1_4 = self.template_QR(4, self.xi40/self.qv )
        R2_1 = self.template_QR(1, self.xi1m1/self.qv )
        R2_3 = self.template_QR(3, self.xi3m1/self.qv )
        self.xxR1 = self.k**2 *self.P_x * ( 8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.xxR2 = self.k**2 *self.P_x * (-2./15 * R1_0 -  2./21* R1_2 + 8./35 * R1_4 + self.k * 2./5*R2_1 - self.k* 2./5*R2_3)
        
        ## --> for A^{10} calculation  $\propto  <\delta_{disp}\delta_{disp}\delta_{disp}\delta_{ini}> $
        self.dxR1 = self.k**2 *self.P_d * ( 8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.dxR2 = self.k**2 *self.P_d * (-2./15 * R1_0 -  2./21* R1_2 + 8./35 * R1_4 + self.k * 2./5*R2_1 - self.k* 2./5*R2_3)


        ## '*_i2' '*_i4' : in the integral, replace $\delta$ with $k^2\delta$ or $k^4\delta$
        R1_0 = self.template_QR(0, self.xi02/self.qv )
        R1_2 = self.template_QR(2, self.xi22/self.qv )
        R1_4 = self.template_QR(4, self.xi42/self.qv )
        R2_1 = self.template_QR(1, self.xi11/self.qv )
        R2_3 = self.template_QR(3, self.xi31/self.qv )
        self.xxR1_i2 = self.k**2 *self.P_x * ( 8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.xxR2_i2 = self.k**2 *self.P_x * (-2./15 * R1_0 -  2./21* R1_2 + 8./35 * R1_4 +  self.k * 2./5*R2_1 - self.k* 2./5*R2_3)

        ## --> for A^{10} calculation  $ \propto  <\delta_{disp}\delta_{disp}\delta_{disp}\delta_{ini}> $
        self.dxR1_i2 = self.k**2 *self.P_d * ( 8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.dxR2_i2 = self.k**2 *self.P_d * (-2./15 * R1_0 -  2./21* R1_2 + 8./35 * R1_4 +  self.k * 2./5*R2_1 - self.k* 2./5*R2_3)

        


        ## --------------------------------------------------------------------------
        ## --------------------------------------------------------------------------
        

    

    def setup_2pts(self):
        # Piece together xi_l_n into what we need
        
        ##self.Xlin = 2./3 * (self.xi0m2[0] - self.xi0m2 - self.xi2m2)
        ##self.Ylin = 2 * self.xi2m2
        self.Xlin = 2./3 * (self.dxi0m2[0] - self.dxi0m2 - self.dxi2m2)
        self.Ylin = 2 * self.dxi2m2

        ##self.Xlin_lt = 2./3 * (self.xi0m2_lt[0] - self.xi0m2_lt - self.xi2m2_lt)
        ##self.Ylin_lt = 2 * self.xi2m2_lt
        
        ##self.Xlin_gt = self.Xlin - self.Xlin_lt
        ##self.Ylin_gt = self.Ylin - self.Ylin_lt
        
        ##self.Xlin_gt = 2./3 * (self.xi0m2_gt[0] - self.xi0m2_gt - self.xi2m2_gt)
        ##self.Ylin_gt = 2 * self.xi2m2_gt
        
    
    def setup_shear(self):
        '''
        part of the codes from :
        https://github.com/sfschen/ZeNBu/blob/main/ZeNBu/Utils/qfuncfft.py
        '''
        J2 = 2.*self.xi1m1/15 - 0.2*self.xi3m1
        J3 = -0.2*self.xi1m1 - 0.2*self.xi3m1
        J4 = self.xi3m1
        
        ##J6 = (7*self.xi00 + 10*self.xi20 + 3*self.xi40)/105.
        ##J7 = (4*self.xi20 - 3*self.xi40)/21.
        ##J8 = (-3*self.xi20 - 3*self.xi40)/21.
        ##J9 = self.xi40
        J6 = (7*self.rxi00 + 10*self.rxi20 + 3*self.rxi40)/105.
        J7 = (4*self.rxi20 - 3*self.rxi40)/21.
        J8 = (-3*self.rxi20 - 3*self.rxi40)/21.
        J9 = self.rxi40
        
        self.V = 4 * J2 * self.rxi20
        self.Xs2 = 4 * J3**2
        self.Ys2 = 6*J2**2 + 8*J2*J3 + 4*J2*J4 + 4*J3**2 + 8*J3*J4 + 2*J4**2
        ##self.zeta = 2*(4*self.xi00**2/45. + 8*self.xi20**2/63. + 8*self.xi40**2/35)
        ##self.chi  = 4*self.xi20**2/3.
        self.zeta = 2*(4*self.rxi00**2/45. + 8*self.rxi20**2/63. + 8*self.rxi40**2/35)
        self.chi  = 4*self.rxi20**2/3.
        
        # one beyond one-loop contribution not reduceable to previous functions:
        # We write this as Lij = 1/8 * <Delta_i Delta_j s^2_1 s^2_2> = Xs4 * delta_ij + Ys4 qiqj
        K1 = 2*J3*(J6+J8)
        K2 = 2*J3*(J7+2*J8+J9) + J4*(2*J6+J7+4*J8+J9)
        
        self.Xs4 = 2*K1*J3
        self.Ys4 = 2*K1*(J2+J3+J4) + K2*(J2+2*J3+J4)
    



    def setup_oneloop_2pts(self):
        # same as above but for all the one loop pieces
        
        # Aij 1 loop
        self.xi0m2loop13 = self.xi_l_n(0,-2, _int=5./21*self.R1)
        self.xi2m2loop13 = self.xi_l_n(2,-2, _int=5./21*self.R1)
        
        self.Xloop13 = 2./3 * (self.xi0m2loop13[0] - self.xi0m2loop13 - self.xi2m2loop13)
        self.Yloop13 = 2 * self.xi2m2loop13
        
        self.xi0m2loop22 = self.xi_l_n(0,-2, _int=9./98*self.Q1)
        self.xi2m2loop22 = self.xi_l_n(2,-2, _int=9./98*self.Q1)

        self.Xloop22 = 2./3 * (self.xi0m2loop22[0] - self.xi0m2loop22 - self.xi2m2loop22)
        self.Yloop22 = 2 * self.xi2m2loop22
        
        # Wijk
        self.Tloop112 = self.xi_l_n(3,-3, _int=-3./7*(2*self.R1+4*self.R2+self.Q1+2*self.Q2))
        self.V1loop112 = self.xi_l_n(1,-3,_int=3./35*(-3*self.R1+4*self.R2+self.Q1+2*self.Q2)) - 0.2*self.Tloop112
        self.V3loop112 = self.xi_l_n(1,-3,_int=3./35*(2*self.R1+4*self.R2-4*self.Q1+2*self.Q2)) - 0.2*self.Tloop112
        

        # A10
        ##self.zerolag_10_loop12 = np.trapz((self.R1-self.R2)/7.,x=self.k) / (2*np.pi**2)
        ##self.xi0m2_10_loop12 = self.xi_l_n(0,-2, _int=4*self.R2+2*self.Q5)/14.
        ##self.xi2m2_10_loop12 = self.xi_l_n(2,-2, _int=3*self.R1+4*self.R2+2*self.Q5)/14.
        self.zerolag_10_loop12 = np.trapz((self.xdR1-self.xdR2)/7.,x=self.k) / (2*np.pi**2)
        self.xi0m2_10_loop12 = self.xi_l_n(0,-2, _int= 2*self.xdR2-2*self.xdR1 + 2*self.dxR1+2*self.dxR2 +2*self.Q5 )/14.
        self.xi2m2_10_loop12 = self.xi_l_n(2,-2, _int=   self.xdR1+2*self.xdR2 + 2*self.dxR1+2*self.dxR2 +2*self.Q5 )/14.
        self.X10loop12 = self.zerolag_10_loop12 - self.xi0m2_10_loop12 - self.xi2m2_10_loop12
        self.Y10loop12 = 3*self.xi2m2_10_loop12

        ## A10, replace  $\delta$  ->  $-k^2\delta$  
        ## note that the minus sign in `-k^2` is not included in the `*_i2` expression like `R1_i2`
        ksq = self.k**2
        temp_zerolag_10_loop12 = np.trapz( ksq* (self.xdR1-self.xdR2)/7.,x=self.k) / (2*np.pi**2)
        temp_xi0m2_10_loop12 = self.xi_l_n(0,-2, _int= ksq* (2*self.xdR2-2*self.xdR1) +2*self.dxR1_i2+2*self.dxR2_i2 +2*self.Q5_i2 )/14.
        temp_xi2m2_10_loop12 = self.xi_l_n(2,-2, _int= ksq* (  self.xdR1+2*self.xdR2) +2*self.dxR1_i2+2*self.dxR2_i2 +2*self.Q5_i2 )/14.
        self.X10loop12_i2 = temp_zerolag_10_loop12 - temp_xi0m2_10_loop12 - temp_xi2m2_10_loop12
        self.Y10loop12_i2 = 3*temp_xi2m2_10_loop12
        self.A10_X_i2, self.A10_Y_i2 = -2 *self.X10loop12_i2, -2 *self.Y10loop12_i2

    
        # the various Us
        ##self.U3 = self.xi_l_n(1,-1, _int=-5./21*self.R1)
        ##self.U11 = self.xi_l_n(1,-1,-6./7*(self.R1+self.R2))
        self.U3 = self.xi_l_n(1,-1, _int=-5./21*self.xdR1)
        self.U11 = self.xi_l_n(1,-1,-6./7*(self.xxR1+self.xxR2))
        self.U20 = self.xi_l_n(1,-1,-3./7*self.Q8)
        self.Us2 = self.xi_l_n(1,-1,-1./7*self.Qs2)  # earlier this was 2/7 but that's wrong
        self.U11_i2  = self.xi_l_n(1,-1, 3./7 *ksq* (self.xxR1+self.xxR2)) \
                     + self.xi_l_n(1,-1, 3./7*(self.xxR1_i2 +self.xxR2_i2))                ## \delta -> (-k^2)\delta
        self.U11_i4  = self.xi_l_n(1,-1,-6./7 *ksq* (self.xxR1_i2+self.xxR2_i2))     ## both two: \delta -> (-k^2)\delta
        ##self.U11    = self.xi_l_n(1, 0,-6./7*self.XPk)
        ##self.U11_i2 = self.xi_l_n(1, 0, 3./7*self.XPk_i2) + self.xi_l_n(1,0,3./7*self.XPk *self.k**2)    
        ##self.U11_i4 = self.xi_l_n(1, 0,-6./7.*self.XPk_i2 *self.k**2)  


        ## --------------------------------------------------------------------------
        ##self.U11_i2 = self.xi_l_n(1,-1,-6./7*(self.R1_i2+self.R2_i2))
        ##self.U11_i4 = self.xi_l_n(1,-1,-6./7*(self.R1_i4+self.R2_i4))
        ## --------------------------------------------------------------------------

    
    def setup_third_order(self):
        # All the terms involving the third order bias, which is really just a few
        '''
        P3_0 = self.k**2 * self.template_QR(0, 24./5*self.xi00/self.qv)
        P3_1 = self.k    * self.template_QR(1, -16./5*self.xi11/self.qv)
        P3_2 = self.k**2 * self.template_QR(2, -20./7*self.xi20/self.qv) + self.template_QR(2,4.*self.xi22/self.qv)
        P3_3 = self.k    * self.template_QR(3, -24./5*self.xi31/self.qv)
        P3_4 = self.k**2 * self.template_QR(4, 72./35*self.xi40/self.qv)
        '''
        P3_0 = self.k**2 * self.template_QR(0, 24./5*self.rxi00/self.qv)
        P3_1 = self.k    * self.template_QR(1, -16./5*self.rxi11/self.qv)
        P3_2 = self.k**2 * self.template_QR(2, -20./7*self.rxi20/self.qv) + self.template_QR(2,4.*self.rxi22/self.qv)
        P3_3 = self.k    * self.template_QR(3, -24./5*self.rxi31/self.qv)
        P3_4 = self.k**2 * self.template_QR(4, 72./35*self.rxi40/self.qv)

        self.Rb3 = 2 * 2./63 * (P3_0 + P3_1 + P3_2 + P3_3 + P3_4)
        
        ##self.theta = self.xi_l_n(0, 0, _int= self.P_x *self.Rb3)
        self.Ub3 = - self.xi_l_n(1,-1, _int= self.P_x *self.Rb3)
    

    
    def xi_l_n(self, l, n, _int=None, IR_cut = 'all', extrap=False, qmin=1e-3, qmax=1000, side='both', ):
        '''
        Calculates the generalized correlation function xi_l_n, which is xi when l = n = 0
        
        If _int is None assume integrating the power spectrum.
        '''
        if _int is None:
            integrand = self.P_x * self.k**n
        else:
            integrand = _int * self.k**n
        
        if IR_cut != 'all':
            if IR_cut == 'gt':
                integrand *= self.ir_greater
            elif IR_cut == 'lt':
                integrand *= self.ir_less
        
        qs, xint =  self.sph.sph(l,integrand)

        if extrap:
            qrange = (qs > qmin) * (qs < qmax)
            return loginterp(qs[qrange],xint[qrange],side=side)(self.qv)
        else:
            return np.interp(self.qv, qs, xint)
    

    def rxi_l_n(self, l, n, **kwargs):
        return self.xi_l_n( l, n, _int=self.P_r, **kwargs )
    

    def dxi_l_n(self, l, n, **kwargs):
        return self.xi_l_n( l, n, _int=self.P_d, **kwargs )



    def template_QR(self,l,integrand):
        '''
        Interpolates the Hankel transformed R(k), Q(k) back onto self.k
        '''
        kQR, QR = self.sphr.sph(l,integrand)
        return np.interp(self.k, kQR, QR)    

