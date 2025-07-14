import numpy as np


def _reform_pks( pks_list ):
    '''
    List[21] -> List[6][6]
    '''
    pk_ij = [ [ None for i in range(6) ] for j in range(6) ]
    k = -1
    for i in range(6):
        for j in range(i, 6):
            k += 1
            pk_ij[i][j] = pks_list[k]
            pk_ij[j][i] = pks_list[k]
    return pk_ij




class EFTofLSS_Base:
    
    @staticmethod
    def m2_v1v1( m2, v1, v2=None, ):
        if v2 is None : v2 = v1
        return np.einsum('i,j,ijx', v1, v2, m2 )
    
    def m4_v1v1v1v1( m4, v1, v2=None, v3=None, v4=None, ):
        if v2 is None : v2 = v1
        if v3 is None : v3 = v1
        if v4 is None : v4 = v1
        return np.einsum('i,j,k,l,ijklxy', v1, v2, v3, v4, m4 )
    

    def m2_t1t1( m2, v1, v2=None, ):
        if v2 is None : v2 = v1
        nb = v1.shape[0]
        return np.sum( m2 *v1.reshape(nb, 1, -1) *v2.reshape(1, nb, -1), 
                      axis=(0, 1), )
    
    
    def m4_t1t1t1t1( m4, v1, v2=None, v3=None, v4=None, ):
        if v2 is None : v2 = v1
        if v3 is None : v3 = v1
        if v4 is None : v4 = v1
        nb = v1.shape[0]
        return  np.sum( 
                    np.sum( 
                        np.sum( 
                            np.sum( m4 *v1.reshape(nb,1,1,1, -1, 1), axis=0, ) \
                            *v2.reshape(nb,1,1, -1, 1), axis=0, ) \
                        *v3.reshape(nb,1, 1,-1),  axis=0, ) \
                    *v4.reshape(nb, 1,-1),  axis=0, ) 
    





class EFTofLSS_Model:

    def EFTofLSS_Base():
        return EFTofLSS_Base
    

    @staticmethod
    def biased_spectra( k, pks, bs, ):
        '''
        combine the Lagrangian basis power spectrum to the biased tracer power spectrum
        not include the shot noise term in the auto-power spectrum
        ''' 
        nb = len(bs) + 1
        bss = np.hstack([ [1], bs ])
        P_cross = np.einsum('i,ik', bss, pks[0, :nb], )
        P_auto  = np.einsum('i,j,ijk', bss, bss, pks[:nb, :nb], )
        return P_auto, P_cross
    

    @staticmethod
    def biased_spectra_ErrorEstimate( k, pks, errors_frac, bs, ):
        '''
        pks : array of shape (6, 6, Nk)
        errors_frac : array of shape ( 6, 6, 6, 6, Nk, Nk)
        '''
        nb = len(bs) + 1
        bss = np.hstack([ [1], bs ])
        Errors_frac = pks[:nb, :nb,].reshape(nb, nb, 1, 1, -1, 1) * \
                      pks[:nb, :nb,].reshape(1, 1, nb, nb, 1, -1) * \
                    errors_frac[:nb, :nb, :nb, :nb, :, :]
        cov_hhhh = np.einsum('i,j,k,l,ijklxy', bss, bss, bss, bss, Errors_frac )
        cov_hhhm = np.einsum('i,j,l,ijlxy', bss, bss, bss, Errors_frac[:, :, :, 0] )
        cov_hmhm = np.einsum('i,j,ijxy', bss, bss, Errors_frac[:, 0, :, 0] )

        theoryErrCov =  \
            np.vstack([
                np.hstack([ cov_hhhh, cov_hhhm.T, ]), 
                np.hstack([ cov_hhhm, cov_hmhm  , ]), 
            ])
        vals, vecs = np.linalg.eigh( theoryErrCov )
        vals[ vals<0 ] = 0
        theoryErrCov = vecs * vals@ vecs.T 
        return theoryErrCov
    

    @staticmethod
    def biased_spectra_k2( k, pks, bs, ):
        '''
        same as `biased_spectra`, but replace the $\nabla^2\delta$ with $-k^2\delta$
        ''' 
        nb = len(bs)
        nk = pks.shape[-1]
        if nb != 4:
            raise ValueError('`biased_spectra_k2` only support 4 bias parameters, but got %d' %nb)
        bss = np.zeros((nb, nk), dtype='float64', )
        bss[1:] += bs[:-1].reshape(-1, 1)
        bss[0] = 1. - k**2 *bs[-1]
        P_cross = np.sum( pks[0, :nb] *bss, axis=(0), )
        P_auto  = np.sum( pks[:nb, :nb] *bss.reshape(nb, 1, -1) *bss.reshape(1, nb, -1), 
                      axis=(0, 1), )
        return P_auto, P_cross
    

    @staticmethod
    def biased_spectra_k2_ErrorEstimate( k, pks, errors_frac, bs, ):
        '''
        pks : array of shape (6, 6, Nk)
        errors_frac : array of shape ( 6, 6, 6, 6, Nk, Nk)
        '''
        nb = len(bs)
        if nb != 4:
            raise ValueError('`biased_spectra_k2` only support 4 bias parameters, but got %d' %nb)
        
        bss = np.hstack([ [1], bs[:-1] ])
        pkx = pks[:nb, :nb,].copy()
        pkx[0, :] *= (1 - k.reshape(1, -1)**2 *bs[-1])
        pkx[:, 0] *= (1 - k.reshape(1, -1)**2 *bs[-1])
        Errors_frac = pkx.reshape(nb, nb, 1, 1, -1, 1) * \
                      pkx.reshape(1, 1, nb, nb, 1, -1) * \
                    errors_frac[:nb, :nb, :nb, :nb, :, :]
        cov_hhhh = np.einsum('i,j,k,l,ijklxy', bss, bss, bss, bss, Errors_frac )
        cov_hhhm = np.einsum('i,j,l,ijlxy', bss, bss, bss, Errors_frac[:, :, :, 0] )
        cov_hmhm = np.einsum('i,j,ijxy', bss, bss, Errors_frac[:, 0, :, 0] )

        theoryErrCov =  \
            np.vstack([
                np.hstack([ cov_hhhh, cov_hhhm.T, ]), 
                np.hstack([ cov_hhhm, cov_hmhm  , ]), 
            ])
        vals, vecs = np.linalg.eigh( theoryErrCov )
        vals[ vals<0 ] = 0
        theoryErrCov = vecs * vals@ vecs.T 
        return theoryErrCov
    



    
    
    # ----------------------------------------------------------------------------
    # for check
    # ----------------------------------------------------------------------------
    
        
    @staticmethod
    def _biased_spectra( k, pks, 
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
    def _biased_spectra_ErrorEstimate( k, pks, errors_frac, 
                    b_1, b_2, b_s2, b_n2, b_3=None, ):
        '''
        pks : array of shape (21, Nk)
        errors_frac : array of shape ( 6, 6, 6, 6, Nk, Nk)
        '''
        if b_3 is None : 
            barr = np.array([1, b_1, b_2, b_s2, b_n2,]) 
        else : 
            barr = np.array([1, b_1, b_2, b_s2, b_n2, b_3, ])
        nb = len(barr)

        nk = pks[0][0].shape[-1]
        cov_hhhh = np.zeros((nk, nk), dtype='float64', )
        cov_hhhm = np.zeros((nk, nk), dtype='float64', )
        cov_hmhm = np.zeros((nk, nk), dtype='float64', )

        for i in range(nb):
            for j in range(i, nb):
                fac1 = 1 if i==j else 2
                
                for m in range(nb):
                    if i==j :
                        cov_hmhm += barr[i] *barr[m] *errors_frac[i, 0, m, 0] \
                                    * pks[i][0].reshape(-1, 1) *pks[m][0].reshape(1, -1)
                    cov_hhhm += fac1 *barr[i] *barr[j] *barr[m] *errors_frac[i, j, m, 0] \
                                    * pks[i][j].reshape(-1, 1) *pks[m][0].reshape(1, -1)

                    for n in range(m, nb):
                        fac2 = 1 if m==n else 2
                        cov_hhhh += fac1 *fac2 *barr[i] *barr[j] *barr[m] *barr[n] *errors_frac[i, j, m, n] \
                                    * pks[i][j].reshape(-1, 1) * pks[m][n].reshape(1, -1)
        
        theoryErrCov =  \
            np.vstack([
                np.hstack([ cov_hhhh, cov_hhhm.T, ]), 
                np.hstack([ cov_hhhm, cov_hmhm  , ]), 
            ])
        vals, vecs = np.linalg.eigh( theoryErrCov )
        vals[ vals<0 ] = 0
        theoryErrCov = vecs * vals@ vecs.T 
        return theoryErrCov
                    
        


    
    @staticmethod
    def _biased_spectra_k2( k, pks, b_1, b_2, b_s2, b_n2, b_3=None, ):
        '''
        same as `CombinePkij`, but replace the $\nabla^2\delta$ with $-k^2\delta$
        '''
        b_ct = 1 - k**2 *b_n2
        P_cross = b_ct* pks[0] + b_1 *pks[1] + b_2 *pks[2] + b_s2 *pks[3] 
        P_auto = b_ct*( b_ct* pks[0] + 2*b_1 *pks[1] + 2*b_2 *pks[2] + 2*b_s2 *pks[3]  ) + \
                    b_1*(                b_1 *pks[6] + 2*b_2 *pks[7] + 2*b_s2 *pks[8]  ) + \
                    b_2*(                                b_2 *pks[11]+ 2*b_s2 *pks[12] ) + \
                    b_s2*(                                               b_s2 *pks[15] ) 
        if b_3 is not None:
            P_cross += b_3 * pks[5] 
            P_auto  += 2* b_3 *( b_ct *pks[5] + b_1 *pks[10] + b_2 *pks[14] + b_s2 *pks[17] ) \
                        + b_3*b_3 *pks[20]
        return P_auto, P_cross
    
    

    @staticmethod
    def _biased_spectra_k2_ErrorEstimate( k, pks, errors_frac, 
                    b_1, b_2, b_s2, b_n2, ):
        '''
        pks : array of shape (21, Nk)
        errors_frac : array of shape ( 6, 6, 6, 6, Nk, Nk)
        '''
        k2L = k.reshape(-1, 1)**2
        k2R = k.reshape(1, -1)**2
        barrL = [ 1 - b_n2*k2L, b_1, b_2, b_s2,]
        barrR = [ 1 - b_n2*k2R, b_1, b_2, b_s2,]
        nb = len(barrL)
        
        nk = pks[0][0].shape[-1]
        cov_hhhh = np.zeros((nk, nk), dtype='float64', )
        cov_hhhm = np.zeros((nk, nk), dtype='float64', )
        cov_hmhm = np.zeros((nk, nk), dtype='float64', )

        for i in range(nb):
            for j in range(i, nb):
                fac1 = 1 if i==j else 2
                
                for m in range(nb):
                    if i==j :
                        cov_hmhm += barrL[i] *barrR[m] *errors_frac[i, 0, m, 0] \
                                    * pks[i][0].reshape(-1, 1) *pks[m][0].reshape(1, -1)
                    cov_hhhm += fac1 *barrL[i] *barrL[j] *barrR[m] *errors_frac[i, j, m, 0] \
                                    * pks[i][j].reshape(-1, 1) *pks[m][0].reshape(1, -1)

                    for n in range(m, nb):
                        fac2 = 1 if m==n else 2
                        cov_hhhh += fac1 *fac2 *barrL[i] *barrL[j] *barrR[m] *barrR[n] *errors_frac[i, j, m, n] \
                                    * pks[i][j].reshape(-1, 1) * pks[m][n].reshape(1, -1)
        
        theoryErrCov =  \
            np.vstack([
                np.hstack([ cov_hhhh, cov_hhhm.T, ]), 
                np.hstack([ cov_hhhm, cov_hmhm  , ]), 
            ])
        vals, vecs = np.linalg.eigh( theoryErrCov )
        vals[ vals<0 ] = 0
        theoryErrCov = vecs * vals@ vecs.T 
        return theoryErrCov
    
