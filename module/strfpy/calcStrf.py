import numpy as np
import os

def df_cal_Strf(params, fstim, fstim_spike, stim_spike_JNf, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol, save_flag=0):
    global DF_PARAMS
    DF_PARAMS=params
    
    # Check if we have all input
    if save_flag is None:
        save_flag = 0

    # Try to get stimuli's durations from variable 'DS'
    DS = DF_PARAMS['DS']
    durs = []
    for jj in range(len(DS)):
        durs.append(DS[jj]['nlen'])

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
        
        # This is PC regresion with a smooth border (orginal STRF code)
        #    for ii in range(nb):
        #        if ii > ranktest[:,0]:
        #            is_mat[ii,ii] = (1.0/ranktol)*np.exp(-((ii-ranktest[0])**2)/8)[0]
        #        else:
        #            is_mat[ii,ii] = 1.0/s[ii]


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