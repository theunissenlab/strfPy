# Regularized Near Diagonal Inversion
import numpy as np


def nearDiagInv(diagA, u, s, v, tol=0):
    ''' Calculates the regularized inverse of a near diagonal matrix, decomposed into its diagonal element and an SVD of its off-diagonal terms
    diag A: The diagonal componnent of the diagonal matrix in its full matrix form
    u, s, v: The singular value decomposition of the offdiagonal terms
    tol: The regularization parameter
    '''
    
    # The dimensionality of square matrices - could check for dimensions here
    nDim = diagA.shape[0]

    # Initialize the inverse
    Ainv = np.diag(1/np.diag(diagA))
    uc = u@np.diag((s))

    # Calculate the inverse by iteration using the Sherman-Morisson formula
    for ie in range(nDim):
        ue = np.reshape(uc[:,ie], (uc.shape[0], 1))
        ve = np.reshape(v[ie,:], (1, v.shape[1]))
        Ainv = Ainv - (s[ie]/(s[ie]+tol)) * (Ainv @ ue @ ve @ Ainv)/(1+ve@Ainv@ue)

    return Ainv

def modifiedRidge(Cxx, Cxy, nSets, ranktol=[0, 0.1, 1, 10, 100]):

 # This is how I would use it in my code.  My actual code has a bit more bells and whistles and I have not tested this code - it is just some cut and paste.
# Cxx: matrix X auto-covariance for LOO across data divided into nSets
# Cxy: cross-covariance vector for LOO as Cxx

    u = np.zeros(Cxx.shape)
    v = np.zeros(Cxx.shape)
    s = np.zeros(Cxy.shape)     # This is just the diagonal
    hJN = np.zeros(Cxy.shape)  # This are the weights
    nb = Cxx.shape[1]

    for iS in range(nSets):
        diagCxx = np.diag(np.diag(np.squeeze(Cxx[iS,:,:])))
        u[iS,:,:],s[iS,:],v[iS,:,:] = np.linalg.svd(Cxx[iS,:,:]-diagCxx)

    mse = np.zeros(len(ranktol))
    for it, tolval in enumerate(ranktol):
        sse = 0.0
        ntot = 0
        for iS, iSet in enumerate(nSets):
            
            diagCxx = np.diag(np.diag(np.squeeze(Cxx[iS,:,:])))
            Cxx_inv = nearDiagInv(diagCxx, u[iS,:,:], s[iS,:], v[iS,:,:], tol=tolval)
            hJN[iS,:] = Cxx_inv @ Cxy[iS,:]

            # here you would also make predictions for set iS (the left out set) to calculate the MSE - for that you need the original x and y
            y = [0, 1, 2, 3]
            err = 0
            sse += err
            ntot += len(y)

        mse[it] = sse/ntot

    # Choose tolval that give the smallest mse.
