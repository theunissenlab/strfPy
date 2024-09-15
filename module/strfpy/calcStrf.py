import numpy as np
import os

def df_cal_Strf(params, fstim, fstim_spike, stim_spike_JNf, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol, save_flag=0):
    global DF_PARAMS
    DF_PARAMS=params
    
    # Check if we have all input
    if save_flag is None:
        save_flag = 0

    # Forward Filter - The algorithm is from FET's filters2.m
    nf = (nt-1)//2 + 1

    # Allocate space for all arrays
    stim_mat = np.zeros((nb, nb), dtype=np.complex_)
    cross_vect = np.zeros((nb, 1), dtype=np.complex_)
    cross_vectJN = np.zeros((nJN, nb), dtype=np.complex_)
    h = np.zeros((1, nb), dtype=np.complex_)
    hJN = np.zeros((nJN, nb), dtype=np.complex_)
    ffor = np.zeros(stim_spike_size, dtype=np.complex_)
    fforJN = np.zeros(stim_spike_JNsize, dtype=np.complex_)
    strfH = np.zeros(stim_spike_size, dtype = np.float_)
    strfHJN = np.zeros(stim_spike_JNsize, dtype=np.float_)
    cums = np.zeros((nf, nb+1), dtype=np.float_)
    ranktest = np.zeros((1, nf), dtype=np.float_)
    stimnorm = np.zeros((1, nf), dtype=np.float_)  

    # Generate the index to fill in the stimulus auto-correlation matrix
    # np_trill_indices does not work because it organizes indices differently
    # this convention matches the one used in calcAutoCorr.py
    indrow = np.zeros(int(nb*(nb+1)/2), dtype = 'int')
    indcol = np.zeros(int(nb*(nb+1)/2), dtype = 'int')
    indi = 0
    for ib1 in range(0, nb):
        for ib2 in range(ib1, nb):
            indrow[indi] = ib1
            indcol[indi] = ib2
            indi += 1


    # Find the maximum norm of all the matrices
    for iff in range(nf):
        stim_mat = np.matrix(np.zeros((nb, nb), dtype=np.complex_))
        stim_mat[(indrow, indcol)] = fstim[:, iff]
        stim_mat = stim_mat - np.diag(np.diag(stim_mat)) + stim_mat.getH()

        stimnorm[0,iff] =  np.linalg.norm(stim_mat)

    ranktol = tol * np.max(stimnorm)
    for iff in range(nf):
        stim_mat = np.matrix(np.zeros((nb, nb), dtype=np.complex_))
        stim_mat[(indrow, indcol)] = fstim[:, iff]
        stim_mat = stim_mat - np.diag(np.diag(stim_mat)) + stim_mat.getH()


        for fb_indx in range(nb):
            cross_vect[fb_indx] = fstim_spike[fb_indx,iff]
            for iJN in range(nJN):
                cross_vectJN[iJN,fb_indx] = stim_spike_JNf[fb_indx,iff,iJN]

        # do an svd decomposition
        ranktest[0, iff] = np.linalg.matrix_rank(stim_mat, ranktol, hermitian=True)
        u,s,v = np.linalg.svd(stim_mat)
        
        # Calculate cumulative iegenvalues for potential display
        tots = s[0]
        cums[iff,1] = s[0] 
        for ii in range(1, nb):
            tots = tots + s[ii]
            cums[iff,ii+1] = cums[iff,ii] + s[ii]

        
        cums[iff,:] = cums[iff,:]/tots



        is_mat = np.zeros((nb, nb))
        

        # ridge regression - regularized normal equation
        for ii in range(nb):
            is_mat[ii,ii] = 1.0/(s[ii] + ranktol)
            # is_mat[ii,ii ] = 1.0   # Testing without any stimulus normalization.
        
        h = v @ is_mat @ (u @ cross_vect)
        hJN = np.zeros((nJN, nb), dtype=np.complex_)

        for iJN in range(nJN):
            hJN[iJN,:] = np.transpose(v @ is_mat @ (u @ cross_vectJN[iJN,:].reshape(-1,1)))

  
        for ii in range(nb):
            ffor[ii,iff] = h[ii]
            fforJN[ii,iff,:] = hJN[:,ii]

            if iff != 0:
                ffor[ii,nt-iff] = np.conj(h[ii])
                fforJN[ii,nt-iff,:] = np.conj(hJN[:,ii])


    nt2 = (nt-1)//2
    xval = np.arange(-nt2, nt2+1)
    wcausal = (np.arctan(xval)+np.pi/2)/np.pi

    for ii in range(nb):
        strfH[ii,:] = np.real(np.fft.ifft(ffor[ii,:]))*wcausal
        for iJN in range(nJN):
            strfHJN[ii,:,iJN] = np.real(np.fft.ifft(fforJN[ii,:,iJN]))*wcausal

    strfHJN_mean = np.mean(strfHJN, axis=2)
    strfHJN_std = np.zeros_like(strfHJN)

    # The following implements the standard error of the estimate.
    # The cross correlation is in the Jackknife estimates so that the strfHJN is also in Jacknife estimates
    # We are calculating one strfHJN standard error per JN - so the number of JN estimates is nJN-1
    if nJN > 1:
        for iJN in range(nJN):
            strfHJN_std[:,:,iJN] = np.std(strfHJN[:,:,np.concatenate((np.arange(iJN), np.arange(iJN+1,nJN)))], axis=2, ddof=0)*np.sqrt((nJN-2))

    if save_flag == 1:
        currentPath = os.getcwd()
        outputPath = DF_PARAMS.outputPath
        if outputPath != '':
            os.chdir(outputPath)
        else:
            print('Saving output to Output Dir.')
            os.mkdir('Output')
            os.chdir('Output')
            outputPath = os.getcwd()

        np.save('strfH.npy', strfH)
        np.save('strfH_std.npy', strfHJN_std)
        for iJN in range(nJN):
            filename = 'strfHJN{}.npy'.format(iJN+1)
            strfHJN_nJN = strfHJN[:,:,iJN]
            np.save(filename, strfHJN_nJN)
        os.chdir(currentPath)

    return strfH, strfHJN, strfHJN_std


def calculateSTRF(fstim, fstim_resp, fstim_resp_JNf, tol):
    nstim = fstim_resp_JNf.shape[0]
    nt = fstim.shape[1]
    nf = (nt-1)//2 + 1
    nb = fstim_resp.shape[0]

    # reshape fstim to be a 3d array
    u_tri = np.triu_indices(nb)
    l_tri = np.tril_indices(nb) # -1 to exclude the diagonal
    stim_mat = np.zeros((nf,nb,nb),dtype=np.complex_)
    stim_mat[:,u_tri[0],u_tri[1]] = fstim[:,:nf].T
    # add the complex conj to the lower triangle
    # note thta this overwrites the diagonal w/ the conjugate (from old code)
    stim_mat[:,l_tri[0],l_tri[1]] = np.swapaxes(stim_mat.conj(),1,2)[:,l_tri[0],l_tri[1]]

    # calculate the norm of each stim_mat for each frequency
    stimnorm = np.linalg.norm(stim_mat,axis=(1,2))

    # find max norm of all matrices for ranktol
    ranktol = tol*np.max(stimnorm)

    # do an svd decomposition
    # ranktest = np.linalg.matrix_rank(stim_mat,ranktol,hermitian=True)

    u,s,v = np.linalg.svd(stim_mat)

    # Calculate cumulative eigenvalues for potential display
    # cums = np.cumsum(s,axis=1)/np.sum(s,axis=1)[:,np.newaxis]
    
    # Ridge Regression - regularized normal equation
    # make array of identity matrices for each frequency
    is_mat = np.broadcast_to(np.eye(nb),(nf,nb,nb))
    is_mat = is_mat / (s + ranktol)[:,np.newaxis,:] # note this is a float op

    # calculate the forward filter and the jackknife forward filter
    h = v @ is_mat @ (u @ fstim_resp[:,:nf].T[:,:,np.newaxis])
    hJN = v @ is_mat @ (u @ fstim_resp_JNf[:,:,:nf].T)

    # setup frequency forward filter array
    ffor = np.zeros((nb,nt),dtype=np.complex_)
    h = h.squeeze().T
    ffor[:,:nf] = h
    ffor[:,nf:] = np.conj(h[:,:0:-1])
    # setup jackknife forward filter array
    fforJN = np.zeros((nb,nt,nstim),dtype=np.complex_)
    hJN = hJN.swapaxes(0,1)
    fforJN[:,:nf,:] = hJN
    fforJN[:,nf:,:] = np.conj(hJN[:,:0:-1,:])

    # calculate the STRF
    nt2 = (nt-1)//2
    xval = np.arange(-nt2, nt2+1)
    wcausal = (np.arctan(xval)+np.pi/2)/np.pi

    strfH = np.real(np.fft.ifft(ffor,axis=1))*wcausal
    strfHJN = np.real(np.fft.ifft(fforJN,axis=1))*wcausal[np.newaxis,:,np.newaxis]
    strfHJN_mean = np.mean(strfHJN,axis=2)
    # now jacknife the STD from the jacknifed means
    jn_idx = np.arange(1, nstim) - np.tri(nstim, nstim-1, k=-1, dtype=bool)

    # This is prohibitevly memory intensive
    #strfHJN_std = np.std(strfHJN[:,:,jn_idx],axis=3,ddof=0)*np.sqrt((nstim-2))

    strfHJN_std = np.zeros_like(strfHJN)
    # here is the unrolled version of above
    for i in range(nstim):
        strfHJN_std[:,:,i] = np.std(strfHJN[:,:,jn_idx[i]],axis=2,ddof=0)*np.sqrt((nstim-2))
    return strfH, strfHJN, strfHJN_std


