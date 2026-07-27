
# chatGPT converted df_cal_AutoCorr.m
# + edit

import os
import numpy as np
from .calcAvg import df_Check_And_Load
from scipy.signal import correlate, correlation_lags


def _stim_autocorr_contribution(stim_env, weight, stim_avg, twindow, nband, spa_corr):
    """One stimulus's own (un-normalized) autocorrelation contribution and
    the corresponding weight-normalization vector -- the inner-loop body of
    df_cal_AutoCorrJN, extracted so it can be (re)computed for a single
    stimulus at a time rather than only ever inline in a loop over all of
    them at once.
    """
    nlen = np.shape(stim_env)[1]

    # subtract mean and normalize by the sqrt of weight.
    stimval = (stim_env[0:nband, :] - np.reshape(stim_avg[0:nband], (nband, 1))) * np.sqrt(weight[0:nlen])

    # The normalization vector: calculated and stored
    xcorr_w = correlate(np.sqrt(weight[0:nlen]), np.sqrt(weight[0:nlen]), mode='full')
    lags = correlation_lags(nlen, nlen, mode="full")
    itlow = np.argwhere(lags == twindow[0])[0][0]
    ithigh = np.argwhere(lags == twindow[1])[0][0]+1
    own_ns = xcorr_w[itlow:ithigh]

    own_contrib = np.zeros((spa_corr, ithigh-itlow))
    xb = 0
    for ib1 in range(nband):
        for ib2 in range(ib1, nband):
            xcorr_s = correlate(stimval[ib1,:], stimval[ib2,:], mode="full")
            own_contrib[xb,:] = xcorr_s[itlow:ithigh]
            xb += 1

    return own_contrib, own_ns


def df_cal_AutoCorrJN(DS, stim_avg, twindow, nband, PARAMS):

    # Initialize the output and set its size
    # Get the total data files
    filecount = len(DS)

    # Temporal axis range
    tot_corr = int(np.diff(twindow)[0] + 1)

    # Spatial axis range
    spa_corr = int((nband * (nband - 1)) // 2 + nband)

    # Initialize CS and CSJN variable (un-normalized grand totals)
    CS_raw = np.zeros((spa_corr, tot_corr))
    CS_ns = np.zeros(tot_corr)

    outputPath = PARAMS['outputPath']
    os.makedirs(outputPath, exist_ok=True)

    # Each stimulus's own (un-normalized) contribution is cached to a small
    # per-stimulus file on disk rather than kept as one of `filecount` full
    # in-RAM arrays (previously CS_JN, a list of filecount arrays each
    # (spa_corr x tot_corr)). For realistic channel/stimulus counts that
    # list was the dominant memory cost of a direct-fit run. The leave-one-
    # out value for a given stimulus is reconstructed on demand from the
    # (small) grand totals below and this stimulus's own cached
    # contribution -- see calcStrf.df_cal_Strf, which consumes these in
    # batches rather than needing all of them loaded at once.
    cache_paths = [os.path.join(outputPath, f'df_jn_autocorr_{fidx}.npz') for fidx in range(filecount)]

    # Do calculation. The algorithm is based on FET's dcp_stim.c
    for fidx in range(filecount):  # Loop through all data files
        # load stimulus file
        stim_env = df_Check_And_Load(DS[fidx]['stimfiles'])
        weight = df_Check_And_Load(DS[fidx]['weightfiles'])

        own_contrib, own_ns = _stim_autocorr_contribution(stim_env, weight, stim_avg, twindow, nband, spa_corr)
        np.savez(cache_paths[fidx], contrib=own_contrib, ns=own_ns)

        CS_ns += own_ns
        CS_raw += own_contrib

    # Finish the normalization of the grand total (this is the CS value
    # returned to callers, e.g. for diagnostic plots)
    CS = CS_raw / CS_ns

    # ========================================================
    # save stimulus auto-correlation matrix into file
    # ========================================================
    np.save(os.path.join(outputPath, 'Stim_autocorr.npy'), CS)

    # ========================================================
    # END OF CAL_AUTOCORR
    # ========================================================

    jn_info = {
        'CS_raw': CS_raw,
        'CS_ns': CS_ns,
        'cache_paths': cache_paths,
    }

    return CS, jn_info