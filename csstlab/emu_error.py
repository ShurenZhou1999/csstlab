import numpy as np


def estimate_emulation_error( PathData, ):

    array_gaussian = np.load( PathData + "SimulationSpectra_ErrorEstimate__Gaussian.npy", allow_pickle=True)[()]
    array_loo      = np.load( PathData + "SimulationSpectra_ErrorEstimate__LOOTest.npy" , allow_pickle=True)[()]

    kedges = array_gaussian["kedges"]
    r2_PsimP1pt = array_gaussian["r2_PsimP1pt"]
    Pk_simu = array_gaussian["Pk_simu"]

    ArrayLOOresults_pred = array_loo["samples_pred"]
    ArrayLOOresults_true = array_loo["samples_true"]

    alpha = 0.01
    s1, s2 = (83, 12, 45, 1), (83, 12, 1, 45)
    ArrayLOO_errors = np.zeros( (12, 6, 6, 6, 6, 45, 45) )
    for i1 in range(6):
        for j1 in range(i1+1):
            for i2 in range(6):
                for j2 in range(i2+1):
                    epsi1 = ArrayLOOresults_pred[:, :, i1, j1] - ArrayLOOresults_true[:, :, i1, j1]
                    epsi2 = ArrayLOOresults_pred[:, :, i2, j2] - ArrayLOOresults_true[:, :, i2, j2]
                    denom = ArrayLOOresults_true[:, :, i1, j1].reshape(s1) *ArrayLOOresults_true[:, :, i2, j2].reshape(s2)
                    denom = np.abs(denom) + alpha *np.mean( epsi1.reshape(s1) *epsi2.reshape(s2) , axis=0)
                    ArrayLOO_errors[:, i1, j1, i2, j2] = (1+alpha)* np.mean( epsi1.reshape(s1) *epsi2.reshape(s2) /denom, axis=0, )

                    ArrayLOO_errors[:, i1, j1, j2, i2] = ArrayLOO_errors[:, i1, j1, i2, j2]
                    ArrayLOO_errors[:, j1, i1, i2, j2] = ArrayLOO_errors[:, i1, j1, i2, j2]
                    ArrayLOO_errors[:, j1, i1, j2, i2] = ArrayLOO_errors[:, i1, j1, i2, j2]
    
    
    V_simu = 1e9
    dk = kedges[1:] - kedges[:-1]
    karr = 0.5* (kedges[1:] +kedges[:-1])
    kcut = 45
    karr, dk = karr[:kcut], dk[:kcut]

    F_k_k0 = (  lambda k, k0, delta_k : 0.5*( 1 - np.tanh((k-k0)/delta_k) )  ) \
            ( karr, 0.618, 0.167 )
    F_k_k0 **= 2
    CovFac = 2*np.pi**2 / (karr**2 *dk) /V_simu

    LOOgauss = np.zeros_like(ArrayLOO_errors)
    diagx, diagy = np.diag_indices(kcut, 2)
    for i in range(6):
        for j in range(6):
            for m in range(6):
                for n in range(6):
                    Vol_frac_1 = 1 - F_k_k0 *r2_PsimP1pt[:, i, j, :kcut]
                    Vol_frac_2 = 1 - F_k_k0 *r2_PsimP1pt[:, m, n, :kcut]
                    Vol_frac_12 = np.sqrt( np.abs(Vol_frac_1*Vol_frac_2) )
                    LOOgauss[:,i,j,m,n,][:, diagx, diagy] = CovFac *Vol_frac_12 \
                            *( Pk_simu[:,i,m,:] *Pk_simu[:,j,n,:] + Pk_simu[:,i,n,:] *Pk_simu[:,j,m,:] )[..., :kcut] \
                            / (Pk_simu[:,i,j,:kcut] *Pk_simu[:,m,n,:kcut])
    
    return LOOgauss, ArrayLOO_errors