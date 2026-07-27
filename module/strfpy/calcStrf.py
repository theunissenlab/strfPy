import numpy as np
import os

from .calcCrossCorr import fft_autocorr_rows


def _build_hermitian(vec, indrow, indcol, nb):
    """Reassemble the (nb, nb) Hermitian stimulus-autocorrelation matrix at
    one frequency bin from its packed upper-triangular representation.
    """
    mat = np.zeros((nb, nb), dtype=np.complex128)
    mat[(indrow, indcol)] = vec
    return mat - np.diag(np.diag(mat)) + mat.conj().T


def df_cal_Strf(CS, jn_info, fstim_spike, stim_spike_JNf, nb, TimeLag, tol_vals, batch_size=20):
    """
    Compute the ridge-regularized STRF (and its jackknife replicates) in the
    frequency domain, for every tolerance value in `tol_vals` at once.

    The stimulus autocorrelation matrix at each frequency bin (and its SVD,
    the dominant cost of this function) depends only on the data, not on the
    ridge tolerance, so it is computed once per frequency bin (and once per
    jackknifed stimulus per frequency bin) and reused across every tolerance
    value, rather than being recomputed from scratch per tolerance value.

    The jackknife replicates are additionally processed in batches of
    `batch_size`: for realistic channel/stimulus counts, holding every
    replicate's (nb(nb+1)/2 x nt) frequency-domain autocorrelation in memory
    at once (as this function used to, via a `fstim_JN` list of length
    nJN) is the dominant memory cost of a direct-fit run. Each replicate's
    leave-one-out autocorrelation is instead reconstructed on demand from
    `jn_info` (see calcAutoCorr.df_cal_AutoCorrJN) for only the current
    batch, FFT'd, run through the regularized inverse, and inverse-FFT'd to
    its small real-valued time-domain STRF slice before the batch's large
    intermediates are discarded and the next batch begins.

    Parameters
    ----------
    CS : ndarray
        Grand-total (leave-none-out) stimulus autocorrelation, packed
        upper-triangular, shape (nb(nb+1)/2, 2*TimeLag+1).
    jn_info : dict
        As returned by calcAutoCorr.df_cal_AutoCorrJN: the raw (un-
        normalized) grand totals (CS_raw, CS_ns) and the per-stimulus cache
        file paths (cache_paths) needed to reconstruct any stimulus's
        leave-one-out autocorrelation on demand.
    fstim_spike, stim_spike_JNf : ndarray
        Small, already-FFT'd stimulus-response cross-correlation and its
        jackknife replicates, as returned by
        calcCrossCorr.fft_crosscorr_jn. Not batched: this array is nb x nt
        x nJN (not nb(nb+1)/2 x nt x nJN), orders of magnitude smaller than
        the autocorrelation jackknife array for realistic channel counts.
    tol_vals : array_like
        One or more ridge tolerance values (as a fraction of the max
        stimulus-autocorrelation matrix norm across frequency bins).
    batch_size : int
        Number of jackknife replicates whose large frequency-domain
        intermediates are held in memory at once. 1 minimizes peak memory
        at the cost of the most per-batch overhead; larger values trade
        memory for fewer, larger batches. Does not affect the result.

    Returns
    -------
    list of (strfH, strfHJN, strfHJN_std)
        One tuple per entry of `tol_vals`, in the same order.
    """

    tol_vals = np.atleast_1d(np.asarray(tol_vals, dtype=np.float64))
    ntols = tol_vals.shape[0]

    nt = 2 * TimeLag + 1
    nf = (nt - 1) // 2 + 1
    nJN = len(jn_info['cache_paths'])

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

    fstim = fft_autocorr_rows(CS, TimeLag)

    # Find the maximum norm across all frequency bins (data-dependent only,
    # not tolerance-dependent, and only ever needed from the grand total)
    stimnorm = np.zeros(nf, dtype=np.float64)
    for iff in range(nf):
        stim_mat = _build_hermitian(fstim[:, iff], indrow, indcol, nb)
        stimnorm[iff] = np.linalg.norm(stim_mat)

    ranktol_vals = tol_vals * np.max(stimnorm)

    nt2 = (nt-1)//2
    xval = np.arange(-nt2, nt2+1)
    wcausal = (np.arctan(xval)+np.pi/2)/np.pi

    is_mat = np.zeros((nb, nb))

    # ---------------------------------------------------------------
    # 1. Main (non-JN) STRF -- small, computed once from the grand-total
    #    fstim/fstim_spike. No batching needed here: there is only one of
    #    these regardless of how many stimuli or tolerance values there are.
    # ---------------------------------------------------------------
    ffor = np.zeros((nb, nt, ntols), dtype=np.complex128)
    cross_vect = np.zeros((nb, 1), dtype=np.complex128)

    for iff in range(nf):
        stim_mat = _build_hermitian(fstim[:, iff], indrow, indcol, nb)
        for fb_indx in range(nb):
            cross_vect[fb_indx] = fstim_spike[fb_indx, iff]

        u, s, v = np.linalg.svd(stim_mat)

        for itol in range(ntols):
            ranktol = ranktol_vals[itol]
            is_mat[:] = 0.0
            for ii in range(nb):
                is_mat[ii,ii] = 1.0/(s[ii] + ranktol)
            h = (u @ is_mat @ (v @ cross_vect)).squeeze()

            for ii in range(nb):
                ffor[ii,iff,itol] = h[ii]
                if iff != 0:
                    ffor[ii,nt-iff,itol] = np.conj(h[ii])

    strfH_per_tol = []
    for itol in range(ntols):
        strfH = np.zeros((nb, nt), dtype=np.float64)
        for ii in range(nb):
            strfH[ii,:] = np.real(np.fft.ifft(ffor[ii,:,itol]))*wcausal
        strfH_per_tol.append(strfH)

    del ffor

    # ---------------------------------------------------------------
    # 2. Jackknife replicates -- processed in batches of `batch_size` so
    #    only a bounded number of stimuli's large (nb(nb+1)/2 x nt)
    #    frequency-domain intermediates are ever alive simultaneously,
    #    instead of all nJN of them at once.
    # ---------------------------------------------------------------
    strfHJN_per_tol = [np.zeros((nb, nt, nJN), dtype=np.float64) for _ in range(ntols)]

    for batch_start in range(0, nJN, batch_size):
        batch = list(range(batch_start, min(batch_start + batch_size, nJN)))
        n_batch = len(batch)

        # Reconstruct this batch's leave-one-out autocorrelation from the
        # small grand totals plus this batch's cached own-contributions,
        # and FFT only this batch -- never all nJN replicates at once.
        batch_fstim_JN = []
        for iJN in batch:
            cached = np.load(jn_info['cache_paths'][iJN])
            own_contrib, own_ns = cached['contrib'], cached['ns']
            loo_ns = jn_info['CS_ns'] - own_ns
            loo = (jn_info['CS_raw'] - own_contrib) / loo_ns
            batch_fstim_JN.append(fft_autocorr_rows(loo, TimeLag))

        fforJN = np.zeros((nb, nt, n_batch, ntols), dtype=np.complex128)

        for iff in range(nf):
            u_JN = np.zeros((n_batch, nb, nb), dtype=np.complex128)
            s_JN = np.zeros((n_batch, nb), dtype=np.float64)
            v_JN = np.zeros((n_batch, nb, nb), dtype=np.complex128)
            cross_vect_batch = np.zeros((n_batch, nb, 1), dtype=np.complex128)

            for local_i, iJN in enumerate(batch):
                stim_mat_JN = _build_hermitian(batch_fstim_JN[local_i][:, iff], indrow, indcol, nb)
                u_JN[local_i], s_JN[local_i], v_JN[local_i] = np.linalg.svd(stim_mat_JN)
                cross_vect_batch[local_i, :, 0] = stim_spike_JNf[:, iff, iJN]

            for itol in range(ntols):
                ranktol = ranktol_vals[itol]
                for local_i in range(n_batch):
                    is_mat[:] = 0.0
                    for ii in range(nb):
                        is_mat[ii,ii] = 1.0/(s_JN[local_i, ii] + ranktol)
                    hjn = (u_JN[local_i] @ is_mat @ (v_JN[local_i] @ cross_vect_batch[local_i])).squeeze()
                    for ii in range(nb):
                        fforJN[ii,iff,local_i,itol] = hjn[ii]
                        if iff != 0:
                            fforJN[ii,nt-iff,local_i,itol] = np.conj(hjn[ii])

        for itol in range(ntols):
            for local_i, iJN in enumerate(batch):
                for ii in range(nb):
                    strfHJN_per_tol[itol][ii,:,iJN] = np.real(np.fft.ifft(fforJN[ii,:,local_i,itol]))*wcausal

        # batch_fstim_JN and fforJN go out of scope here and are garbage
        # collected before the next batch begins.

    results = []
    for itol in range(ntols):
        strfHJN = strfHJN_per_tol[itol]
        strfHJN_std = np.zeros_like(strfHJN)

        # The following implements the standard error of the estimate.
        # The cross correlation is in the Jackknife estimates so that the strfHJN is also in Jacknife estimates
        # We are calculating one strfHJN standard error per JN - so the number of JN estimates is nJN-1
        if nJN > 1:
            for iJN in range(nJN):
                strfHJN_std[:,:,iJN] = np.std(strfHJN[:,:,np.concatenate((np.arange(iJN), np.arange(iJN+1,nJN)))], axis=2, ddof=0)*np.sqrt((nJN-2))

        results.append((strfH_per_tol[itol], strfHJN, strfHJN_std))

    return results
