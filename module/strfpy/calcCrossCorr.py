
# chatGPT converted df_cal_CrossCorr.m
# 20230404
# + edit

import os
import numpy as np
import scipy.signal as sps
import pandas

from .calcAvg import df_cal_AVG, df_Check_And_Load
from .calcAvg import calculateLeaveOneOutAverages

def crosscorr(stim, psth, nlags):
    # TODO there may be a smarter way to do this with tensors
    # also could use FFT to speed it up
    lags = np.arange(-nlags,nlags+1)
    nsamps = stim.shape[1]
    nbins = stim.shape[0]
    ac = np.zeros((nbins,len(lags)))
    for lag in lags:
        stim_slice = slice(max(0,lag), min(nsamps,nsamps+lag))
        psth_slice = slice(stim_slice.start-lag, stim_slice.stop-lag)
        # nbin x nsamp @ nsamp x 1 matrix by row
        temp = stim[:,stim_slice] @ psth.T[psth_slice]
        ac[:,lag] += temp.flatten()
    return ac #/ np.arange(stim.shape[1], stim.shape[1] - nlags, -1)

def calculateCrossCorr(dfStrfLab:pandas.DataFrame, nlags:int=74):
    """
    Calculate stimulus spike cross correlation.

    """
    if 'stim_loo_avg' not in dfStrfLab.columns:
        print("Leave one out averages not found in dataframe. running calculateLeaveOneOutAverages()")
        calculateLeaveOneOutAverages(dfStrfLab)

    # calculate cross correlation between stims and psths
    def _apply_crosscorr(row, nlags):
        normed_stim = row['stim'] - row['stim_loo_avg'].reshape(-1, 1)
        normed_psth = row['psth'] - row['psth_loo_avg']
        return crosscorr(normed_stim, normed_psth, nlags)
    cross_corrs = dfStrfLab.apply(_apply_crosscorr,nlags=nlags,axis=1) # each cross_corr has dims (nbands, 2*nlags-1)
    # scale by nTrials
    cross_corrs *= dfStrfLab['nTrials']
    # stack into 3d array for easier jackknifing
    cross_corrs = np.stack(cross_corrs.values) # stack into 3d array (nfiles, nbands, 2*nlags+1)
    
    # now calculate weights for normalization
    weights = dfStrfLab['stim'].apply(lambda x:
        np.correlate(np.ones(x.shape[1]), np.ones(x.shape[1]), mode="same")[int(x.shape[1]/2-nlags):int(x.shape[1]/2+nlags+1)]
        ) * dfStrfLab['nTrials']
    weights = np.stack(weights.values)

    # normalize cross_corrs by weights
    #cross_corrs /= weights[:,np.newaxis,:]

    # now calculate Jackknifed cross-correlations
    # this yields a matrix of indices leaving out the current row
    N = len(cross_corrs.shape[0])
    idx = np.arange(1, N) - np.tri(N, N-1, k=-1, dtype=bool)
    all_cross_corr_minus_one = cross_corrs[idx] # shape (nfiles, nfiles-1, nbands, 2*nlags+1)
    all_weights_minus_one = weights[idx] # shape (nfiles, nfiles-1, 2*nlags+1)
    # now calculate the mean of the cross-correlations leaving out each file
    # this is the jackknifed cross-correlation
    cross_corrs_jn = all_cross_corr_minus_one.sum(axis=1) / all_weights_minus_one.sum(axis=1)[:, np.newaxis]

    return cross_corrs

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
    global DF_PARAMS
    DF_PARAMS = PARAMS
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
    tot_corr = int(np.diff(twindow) + 1)
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
        for tgood in range(nlen):
            stimval[:, tgood] = stim_env[0:nband, tgood] - stim_avg[0:nband]

        # For Time-varying firing rate
        timevary_PSTH = DF_PARAMS["timevary_PSTH"]
        if timevary_PSTH == 1:
            psthval = psth[fidx][0:nlen] - avg_psth[fidx, 0:nlen]
        else:
            psthval = psth[fidx] - avg_psth

        # New version of algorithm for computing cross-correlation
        CSR_JN[fidx] = df_internal_cal_CrossCorr(stimval, psthval, twindow[1])
        CSR += CSR_JN[fidx]
        
        # For normalization and assign the count_ns
 
        CSR_JN_ns[fidx] = np.correlate(np.sqrt(weight), np.sqrt(weight), mode="same")[int(nlen/2-twindow[1]):int(nlen/2+twindow[1]+1)]
        CSR_ns += CSR_JN_ns[fidx]

        # clear workspace by deleting stim_env
        del stim_env, stimval, psthval

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
    outputPath = DF_PARAMS['outputPath']
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




# def df_internal_cal_CrossCorr(stimval, psthval, twin, do_fourier=None):
#     nband = stimval.shape[0]
#     CSR_JN = np.zeros((nband, 2*twin+1))
#     N = len(psthval)
#     td_time = 2.5e-8*nband*N*(1+2*twin) #time in s to calculate using a time-domain algorithm
#     fd_time = 2e-7*N*np.log(N+1)*nband #time in s to calculate using a Fourier-domain algorithm
#     if do_fourier is None:
#         do_fourier = fd_time < td_time
#     if do_fourier:
#         for ib1 in range(nband):

#             CSR_JN[ib1,:] = np.correlate(stimval[ib1,:], psthval.flatten(), mode="full")
#     else:
#         pt = psthval.T
#         for tid in range(-twin, twin+1):
#             onevect = np.arange(max(1,tid+1), min(N,N+tid))
#             othervect = onevect - tid
#             temp = np.dot(stimval[:,onevect], pt[othervect])
#             CSR_JN[:,tid+twin] = temp
#     return CSR_JN


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


# converted with chatgpt: df_fft_AutoCrossCorr.m
# 20230405
def df_fft_AutoCrossCorr(stim, stim_spike, CSR_JN, TimeLag, NBAND, nstd_val):
    
    ncorr = stim.shape[0]
    nb = NBAND
    nt = 2 * TimeLag + 1
    nJN = len(CSR_JN)
    
    asize = np.array([nt, nb])
    w = np.hanning(nt)
    
    stim_spike = np.fliplr(stim_spike)
    for ib in range(nb):
        stim_spike[ib,:] = stim_spike[ib,:] * w
    
    stim_spike_JN = np.zeros((nb, nt, nJN))
    for iJN in range(nJN):
        CSR = CSR_JN[iJN]
        for ib in range(nb):
            stim_spike_JN[ib,:,iJN] = np.flipud(CSR[ib,:]) * w
    
    stim_spike_JNf = np.fft.fft(stim_spike_JN, axis=1)
    stim_spike_JNmf = np.mean(stim_spike_JNf, axis=2)
    stim_spikef = np.fft.fft(stim_spike, axis=1)

    JNv = (nJN - 1) * (nJN - 1) / nJN
    j = 1j
    hcuttoff = 0
    nf = (nt - 1) // 2 + 1
    
    stim_spike_JNvf = np.zeros((nb, nf), dtype=complex)
    fstim = np.zeros(stim.shape, dtype=complex)
    stim_spike_sf = np.zeros((nb, nt),dtype=complex)#, nJN))
    #fstim_spike = stim_spike_sf
    
    for ib in range(nb):
        itstart = 0
        itend = nf
        below = 0
        for it in range(nf):
            stim_spike_JNvf[ib,it] = JNv*np.cov(np.transpose(np.real(stim_spike_JNf[ib,it,:])))
            + j*JNv*np.cov(np.transpose(np.imag(stim_spike_JNf[ib,it,:])))
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

    nt2=int((nt-1)/2)
    for i in range(ncorr):
        sh_stim = np.zeros(nt)
        w_stim =  stim[i,:]*w
        sh_stim[:nt2+1]=w_stim[nt2:nt]
        sh_stim[nt2+1:nt]=w_stim[:nt2]
        fstim[i,:] = np.fft.fft(sh_stim)
    return fstim, fstim_spike, stim_spike_JNf


