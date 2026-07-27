
# chatGPT converted df_cal_CrossCorr.m
# 20230404
# + edit

import os
import numpy as np
import scipy.signal as sps

from .calcAvg import df_cal_AVG, df_Check_And_Load

def df_cal_CrossCorr(DS, PARAMS, stim_avg=None, avg_psth=None, psth=None,
                     twindow=None, nband=None, end_window=0, JN_flag=True):
    """
    Calculate stimulus spike cross-correlation.

    Parameters
    ----------
    DS : list
        The cell of each data struct that contains four fields:
        stimfiles  - stimulus file name
        respfiles  - response file name
        nlength    - length of time domain
        ntrials    - num of trials
        e.g. DS[0] = {'stimfiles': 'stim1.dat', 'respfiles': 'resp1.dat', 'nlen': 1723, 'ntrials': 20}
    stim_avg : numpy.ndarray, optional
        Avg stimulus that used to smooth the noise, by default None
    avg_psth : numpy.ndarray, optional
        Average psth over all trials and over all time, by default None
    psth : list of numpy.ndarray, optional
        The cell of avg. psth over trials, by default None
    twindow : numpy.ndarray, optional
        The variable to set the time interval to calculate
        autocorrelation. e.g. twindow=[-300 300], by default None
    nband : int, optional
        The size of spatio domain of the stimulus file, by default None
    end_window : int, optional
        The time interval which don't count for data analysis, by default 0
    JN_flag : bool, optional
        The flag that specify whether we calculate JackKnifed CS.
        The default value of JN_flag = True (calculate), False otherwise, by default True

    Returns
    -------
    tuple
        CSR : numpy.ndarray
            Stimulus spike cross correlation. Its size is: nband X (2*twindow +1).
        CSR_JN : list of numpy.ndarray
            Stimulus spike cross correlation. Its size is: nband X (2*twindow +1).
        errFlg : int
            The flag to indicate whether an error has occurred (0: no error, 1: error).
    """
 
    # ========================================================
    # check whether we have valid required input
    # ========================================================
    
    errFlg = 0

    if DS is None:
        errFlg = 1
        raise ValueError('ERROR: Please enter non-empty data filename')
        

    if not JN_flag:
        JN_flag = 1

    if end_window is None:
        end_window = 0

    # check whether stim_avg has been calculated or not
    if stim_avg is None:
        # calculate avg. of stimuli and psh and total_psth
        stim_avg, avg_psth, psth = df_cal_AVG(DS, nband)
    # ========================================================
    # initialize the output and allocate its size
    # ========================================================
    # get the total data files
    filecount = len(DS)
    
    # temporal axis range
    tot_corr = int(np.diff(twindow)[0] + 1)
    # spatial axis range
    spa_corr = int(nband)
    
    # initialize autoCorr and autoCorrJN variable
    CSR = np.zeros((spa_corr, tot_corr))
    CSR_ns = np.zeros(tot_corr)
    # JN varriables
    # CSR_JN = [np.zeros((spa_corr, tot_corr)) for i in range(filecount)]
    # CSR_JN_ns = [np.zeros(tot_corr) for i in range(filecount)]
    CSR_JN = [None]*filecount
    CSR_JN_ns = [None]*filecount

    if JN_flag == 1:
        CSR_JN = [np.zeros((spa_corr, tot_corr)) for i in range(filecount)]
        CSR_JN_ns = [np.zeros(tot_corr) for i in range(filecount)]

    print('Now doing cross-correlation calculation.')

    for fidx in range(filecount):
        # load stimulus file
        stim_env = df_Check_And_Load(DS[fidx]["stimfiles"])
        weight = df_Check_And_Load(DS[fidx]["weightfiles"])

        # get time length for data input set
        nlen = min(psth[fidx].shape[1], stim_env.shape[1])

        # subtract mean_stim from stim and mean_psth from psth
        stimval = np.zeros((nband, nlen))
        stimval = stim_env[0:nband, :] - stim_avg[0:nband].reshape((nband,1))

        # For Time-varying firing rate
        timevary_PSTH = PARAMS["timevary_PSTH"]
        if timevary_PSTH == 1:
            psthval = psth[fidx][0:nlen] - avg_psth[fidx, 0:nlen]
        else:
            psthval = psth[fidx] - avg_psth

        # New version of algorithm for computing cross-correlation
        CSR_JN[fidx] = df_internal_cal_CrossCorr(stimval, psthval*weight[0:nlen], twindow[1])
        CSR += CSR_JN[fidx]
        
        # For normalization and assign the count_ns
 
        CSR_JN_ns[fidx] = np.correlate(np.ones(nlen), weight[0:nlen], mode="same")[int(nlen/2-twindow[1]):int(nlen/2+twindow[1]+1)]
        CSR_ns += CSR_JN_ns[fidx]

    print("Done calculation of cross-correlation.")
    
    print('Now calculating JN cross-correlation.')
    # Calculate JN version of cross-correlation
    if JN_flag == 1:
        if filecount >1:
            for iJN in range(filecount):

                # Count ns for each JN and normalize it later on
                CSR_JN_ns[iJN] = CSR_ns - CSR_JN_ns[iJN]
                nozero_ns = np.isinf(1 / CSR_JN_ns[iJN]) + CSR_JN_ns[iJN]

                for ib in range(nband):
                    CSR_JN[iJN][ib,:] = (CSR[ib,:] - CSR_JN[iJN][ib,:]) / nozero_ns

            
    print('Done calculation of JN cross-correlation.')
    
    # Normalize CSR by CSR_ns
    nozero_ns = np.isinf(1 / CSR_ns) + CSR_ns
    CSR /= nozero_ns
    
    # Save stim-spike cross correlation matrix into a file
    currentPath = os.getcwd()
    outputPath = PARAMS['outputPath']
    if outputPath:
        os.chdir(outputPath)
    else:
        print('Saving output to Output Dir.')
        os.mkdir('Output')
        os.chdir('Output')
        outputPath = os.getcwd()
    
    np.save('StimResp_crosscorr.npy', CSR)
    np.save('SR_crosscorrJN.npy', CSR_JN)
    os.chdir(currentPath)

    return CSR, CSR_JN, errFlg


def df_internal_cal_CrossCorr(stimval, psthval, twin, do_fourier=None):
    nband = stimval.shape[0]
    CSR_JN = np.zeros((nband, 2*twin+1))
    N = psthval.shape[1]
    td_time = 2.5e-8 * nband * N * (1 + 2*twin)  # time in s to calculate using a time-domain algorithm
    fd_time = 2e-7 * N * np.log(N+1) * nband  # time in s to calculate using a Fourier-domain algorithm
    if do_fourier is None:
        do_fourier = fd_time < td_time
    if do_fourier:
        for ib1 in range(nband):
            CSR_JN[ib1,:] = sps.correlate(stimval[ib1,:], psthval.flatten(), mode='same')[int(N/2-twin):int(N/2+twin+1)]
    else:
        pt = psthval.T
        for tid in range(-twin, twin+1):
            onevect = slice(max(0,tid), min(N,N+tid))
            othervect = slice(onevect.start-tid, onevect.stop-tid)
            temp = stimval[:,onevect] @ pt[othervect,:]
            CSR_JN[:,tid+twin] = temp.flatten()
    return CSR_JN


def fft_autocorr_rows(stim, TimeLag):
    """FFT every row of a (ncorr, nt) real autocorrelation array after
    Hanning windowing and the circular shift that puts lag 0 first.

    This is the per-row transform that used to live inline inside
    df_fft_AutoCrossCorr, applied identically to the grand-total
    autocorrelation and to every one of `filecount` jackknife replicates in
    a single Python double loop. It is now a standalone, vectorized
    (row-batched instead of looped) helper so it can be called once for the
    (small) grand total and, separately, once per stimulus for however many
    jackknife replicates are being processed in the current batch --
    calcStrf.df_cal_Strf calls this per-batch rather than needing every
    replicate's transform computed and held at once.
    """
    nt = 2 * TimeLag + 1
    nt2 = (nt - 1) // 2
    w = np.hanning(nt)

    w_stim = stim * w
    sh_stim = np.empty_like(w_stim)
    sh_stim[:, :nt2+1] = w_stim[:, nt2:nt]
    sh_stim[:, nt2+1:] = w_stim[:, :nt2]

    return np.fft.fft(sh_stim, axis=1)


# converted with chatgpt: df_fft_AutoCrossCorr.m
# 20230405
def fft_crosscorr_jn(stim_spike, CSR_JN, TimeLag, NBAND, nstd_val):
    """FFT the (small, nb x nt x filecount) stimulus-response cross-
    correlation and its jackknife replicates, applying a shrinkage taper
    derived from the variance across replicates. Split out of the old
    df_fft_AutoCrossCorr, which bundled this together with the (large,
    quadratic-in-channel-count) autocorrelation FFT -- see
    fft_autocorr_rows for that half. This part does not need batching: the
    cross-correlation jackknife array is nb x nt x filecount (not
    nb(nb+1)/2 x nt x filecount), orders of magnitude smaller than the
    autocorrelation one for realistic channel counts.
    """

    nb = NBAND
    nt = 2 * TimeLag + 1
    nJN = len(CSR_JN)

    w = np.hanning(nt)

    stim_spike = np.fliplr(stim_spike)
    for ib in range(nb):
        stim_spike[ib,:] = stim_spike[ib,:] * w

    stim_spike_JN = np.zeros((nb, nt, nJN))
    for iJN in range(nJN):
        for ib in range(nb):
            stim_spike_JN[ib,:,iJN] = np.flipud(CSR_JN[iJN][ib,:]) * w

    stim_spike_JNf = np.fft.fft(stim_spike_JN, axis=1)
    stim_spike_JNmf = np.mean(stim_spike_JNf, axis=2)
    stim_spikef = np.fft.fft(stim_spike, axis=1)

    JNv = (nJN - 1) * (nJN - 1) / nJN
    j = 1j
    nf = (nt - 1) // 2 + 1

    stim_spike_JNvf = np.zeros((nb, nf), dtype=complex)

    stim_spike_sf = np.zeros((nb, nt),dtype=complex)#, nJN))
    #fstim_spike = stim_spike_sf

    for ib in range(nb):
        itstart = 0
        itend = nf
        below = 0
        for it in range(nf):
            stim_spike_JNvf[ib,it] = (JNv*np.cov(np.transpose(np.real(stim_spike_JNf[ib,it,:])))
                + j*JNv*np.cov(np.transpose(np.imag(stim_spike_JNf[ib,it,:]))))
            rmean = np.real(stim_spike_JNmf[ib,it])
            rstd = np.sqrt(np.real(stim_spike_JNvf[ib,it]))
            imean = np.imag(stim_spikef[ib,it])
            istd = np.sqrt(np.imag(stim_spike_JNvf[ib,it]))
            if abs(rmean) < nstd_val * rstd and abs(imean) < nstd_val * istd:
                if itstart == 0:
                    itstart = it
                below = below + 1
            else:
                below = 0
                itstart = 0
            stim_spike_sf[ib,it] = rmean + j*imean
            #fstim[ib,itstart:itend] = np.real(np.fft.ifft(stim_spikef[ib,itstart:itend]))
            #fstim_spike[ib,itstart:itend] = np.real(np.fft.ifft(stim_spike_JNf[ib,itstart:itend,:], axis=1))
        for it in range(nf):
            if it > itstart:
                expval = np.exp(-0.5*(it-itstart)**2/(itend-itstart)**2)
                stim_spike_sf[ib,it] = stim_spike_sf[ib,it]*expval
                stim_spike_JNf[ib,it,:] = stim_spike_JNf[ib,it,:]*expval
            if it > 0:
                stim_spike_sf[ib,nt-it] = np.conj(stim_spike_sf[ib,it])
                stim_spike_JNf[ib,nt-it,:] = np.conj(stim_spike_JNf[ib,it,:])
    fstim_spike = stim_spike_sf

    return fstim_spike, stim_spike_JNf


