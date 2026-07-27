import numpy as np
import os

def df_cal_Strf(params, fstim, fstim_JN, fstim_spike, stim_spike_JNf, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol_vals):
    """
    Compute the ridge-regularized STRF (and its jackknife replicates) in the
    frequency domain, for every tolerance value in `tol_vals` at once.

    The stimulus autocorrelation matrix at each frequency bin (and its SVD,
    the dominant cost of this function) depends only on the data, not on the
    ridge tolerance. This computes that SVD once per frequency bin (and once
    per jackknifed stimulus per frequency bin) and reuses it across every
    tolerance value, rather than recomputing it from scratch once per
    tolerance value as calling this function once per entry of `tol_vals`
    used to require.

    Parameters
    ----------
    tol_vals : array_like
        One or more ridge tolerance values (as a fraction of the max
        stimulus-autocorrelation matrix norm across frequency bins).

    Returns
    -------
    list of (strfH, strfHJN, strfHJN_std)
        One tuple per entry of `tol_vals`, in the same order.
    """

    tol_vals = np.atleast_1d(np.asarray(tol_vals, dtype=np.float64))
    ntols = tol_vals.shape[0]

    # Forward Filter - The algorithm is from FET's filters2.m
    nf = (nt-1)//2 + 1

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

    # Find the maximum norm of all the matrices (data-dependent only, not
    # tolerance-dependent, so this is still only done once)
    stimnorm = np.zeros((1, nf), dtype=np.float64)
    for iff in range(nf):
        stim_mat = np.zeros((nb, nb), dtype=np.complex128)
        stim_mat[(indrow, indcol)] = fstim[:, iff]
        stim_mat = stim_mat - np.diag(np.diag(stim_mat)) + stim_mat.conj().T

        stimnorm[0,iff] =  np.linalg.norm(stim_mat)

    ranktol_vals = tol_vals * np.max(stimnorm)

    # ffor/fforJN hold the frequency-domain filter for every tolerance value
    # simultaneously: the causal time-domain filter for a given tolerance can
    # only be assembled (via ifft) once every frequency bin has been
    # visited, and visiting each frequency bin's (expensive) SVD exactly
    # once -- regardless of how many tolerance values are being swept -- is
    # the whole point of this restructuring.
    ffor = np.zeros(stim_spike_size + (ntols,), dtype=np.complex128)
    fforJN = np.zeros(stim_spike_JNsize + (ntols,), dtype=np.complex128)

    cross_vect = np.zeros((nb, 1), dtype=np.complex128)
    cross_vectJN = np.zeros((nJN, nb), dtype=np.complex128)
    is_mat = np.zeros((nb, nb))

    for iff in range(nf):
        stim_mat = np.zeros((nb, nb), dtype=np.complex128)
        stim_mat[(indrow, indcol)] = fstim[:, iff]
        stim_mat = stim_mat - np.diag(np.diag(stim_mat)) + stim_mat.conj().T

        for fb_indx in range(nb):
            cross_vect[fb_indx] = fstim_spike[fb_indx,iff]
            for iJN in range(nJN):
                cross_vectJN[iJN,fb_indx] = stim_spike_JNf[fb_indx,iff,iJN]

        # SVD computed once per frequency bin, reused for every tolerance
        # value in the loop below.
        u, s, v = np.linalg.svd(stim_mat)

        # Repeat for JN values -- also computed once per (frequency bin, JN
        # stimulus), reused across every tolerance value.
        u_JN = np.zeros((nJN, nb, nb), dtype=np.complex128)
        s_JN = np.zeros((nJN, nb), dtype=np.float64)
        v_JN = np.zeros((nJN, nb, nb), dtype=np.complex128)
        for iJN in range(nJN):
            stim_mat_JN = np.zeros((nb, nb), dtype=np.complex128)
            stim_mat_JN[(indrow, indcol)] = fstim_JN[iJN][:, iff]
            stim_mat_JN = stim_mat_JN - np.diag(np.diag(stim_mat_JN)) + stim_mat_JN.conj().T
            u_JN[iJN], s_JN[iJN], v_JN[iJN] = np.linalg.svd(stim_mat_JN)

        for itol in range(ntols):
            ranktol = ranktol_vals[itol]

            # Regularized inverse of stimulus auto-correlation in frequency domain
            is_mat[:] = 0.0
            for ii in range(nb):
                is_mat[ii,ii] = 1.0/(s[ii] + ranktol)

            h = (u @ is_mat @ (v @ cross_vect)).squeeze()

            hJN = np.zeros((nJN, nb), dtype=np.complex128)
            for iJN in range(nJN):
                is_mat[:] = 0.0
                for ii in range(nb):
                    is_mat[ii,ii] = 1.0/(s_JN[iJN, ii] + ranktol)
                hJN[iJN,:] = (u_JN[iJN] @ is_mat @ (v_JN[iJN] @ cross_vectJN[iJN,:].reshape(-1,1))).squeeze()

            for ii in range(nb):
                ffor[ii,iff,itol] = h[ii]
                fforJN[ii,iff,:,itol] = hJN[:,ii]

                if iff != 0:
                    ffor[ii,nt-iff,itol] = np.conj(h[ii])
                    fforJN[ii,nt-iff,:,itol] = np.conj(hJN[:,ii])

    nt2 = (nt-1)//2
    xval = np.arange(-nt2, nt2+1)
    wcausal = (np.arctan(xval)+np.pi/2)/np.pi

    results = []
    for itol in range(ntols):
        strfH = np.zeros((nb, nt), dtype=np.float64)
        strfHJN = np.zeros((nb, nt, nJN), dtype=np.float64)
        for ii in range(nb):
            strfH[ii,:] = np.real(np.fft.ifft(ffor[ii,:,itol]))*wcausal
            for iJN in range(nJN):
                strfHJN[ii,:,iJN] = np.real(np.fft.ifft(fforJN[ii,:,iJN,itol]))*wcausal

        strfHJN_std = np.zeros_like(strfHJN)

        # The following implements the standard error of the estimate.
        # The cross correlation is in the Jackknife estimates so that the strfHJN is also in Jacknife estimates
        # We are calculating one strfHJN standard error per JN - so the number of JN estimates is nJN-1
        if nJN > 1:
            for iJN in range(nJN):
                strfHJN_std[:,:,iJN] = np.std(strfHJN[:,:,np.concatenate((np.arange(iJN), np.arange(iJN+1,nJN)))], axis=2, ddof=0)*np.sqrt((nJN-2))

        results.append((strfH, strfHJN, strfHJN_std))

    return results
