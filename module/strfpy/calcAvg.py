
# Converted using ChatGPT from df_cal_AVG.m, df_Check_And_Load.m
# 20230404

import numpy as np
import scipy.io as sio
import os
from scipy.signal import resample, hann

def df_cal_AVG(DDS, PARAMS, nband=None, psth_option=None, lin_flag=1, sil_window=0):
    """
    Calculate the average stimulus value across time.
    Calculate average of psth over all trials.
    Calculate the psth of response file over all trials.
    
    Args:
    - DDS (list): the cell of each data struct that contains four fields:
        - stimfiles (str): stimulus file name.
        - respfiles (str): response file name.
        - nlen (int): length of time domain.
        - ntrials (int): num of trials.
    - nband (int, optional): the number of frequency bands. Default is None.
    - psth_option (int, optional): the option for psth noise removal. Default is None.
    - lin_flag (int, optional): the flag to show whether we need take log on data. 0 if we need take log, 1(default) if otherwise. Default is 1.
    - sil_window (int, optional): the interval for not counting when preprocessing data. Default is 0.
    
    Returns:
    - stim_avg (ndarray): average stimulus value which is only function of space.
    - Avg_psth (float): average psth.
    - psth (list): the cell of psth for one data pair which is only func of time.
    - constmeanrate (float): the constant mean rate of the response.
    - errFlg (int): flag indicating whether an error occurred (0 for no error, 1 otherwise).
    """

    global DF_PARAMS
    DF_PARAMS = PARAMS

    # check whether we have valid required input
    errFlg = 0
    if not DDS:
        print('ERROR: Please enter non-empty data filename')
        errFlg = 1
        return None, None, None, None, errFlg
    
    NBAND = DF_PARAMS.get('NBAND')
    if nband is None:
        if NBAND is None:
            print('You need assign variable NBAND first.')
            errFlg = 1
            return None, None, None, None, errFlg
        nband = NBAND
    
    # Add parameter 'psth_option' to specifiy psth noise removal option
    timevary_PSTH = DF_PARAMS.get('timevary_PSTH')
    if psth_option is None:
        if timevary_PSTH == 0:
             psth_option = 0
        else:
            psth_option = 1
    
    # if user does not provide other input, we set default
    # lin_flag refers to linear data if lin_flag = 1.
    # otherwise, we take logrithm on data if lin_flag = 0.
    if lin_flag is None:
        lin_flag = 1
        
    if sil_window is None:
        sil_window = 0
        
    # initialize output and declare local variables
    ampsamprate = DF_PARAMS.get('ampsamprate')
    respsamprate = DF_PARAMS.get('respsamprate')
    ndata_files = len(DDS)
    stim_avg = np.zeros(nband)
    count_avg = 0
    tot_trials = 0
    psth = []
    timevary_psth = []
    Avg_psth = 0
    
    # calculate the output over all the data files
    for n in range(ndata_files):
        # load stimulus files
        stim_env = df_Check_And_Load(DDS[n]['stimfiles'])
        this_len = stim_env.shape[1]  # get stim duration
        # load response files
        rawResp = df_Check_And_Load(DDS[n]['respfiles'])

        if isinstance(rawResp, list):
            spiketrain = np.zeros((DDS[n]['ntrials'], this_len))
            for trial_ind in range(DDS[n]['ntrials']):
                spiketrain[trial_ind, rawResp[trial_ind]] = np.ones(1, len(rawResp[trial_ind]))
            if ampsamprate and respsamprate:
                newpsth = resample(spiketrain.T, ampsamprate, respsamprate)
            else:
                newpsth = spiketrain
            newpsth = newpsth.T  # make sure new response data is trials x T.
            newpsth[newpsth < 0] = 0
            psth_rec = newpsth
        else:
            psth_rec = rawResp
            psth_rec = psth_rec.reshape(1, len(psth_rec))
        
        nt = min(stim_env.shape[1], psth_rec.shape[1])
        # take logrithm of data based on lin_flag
        if lin_flag == 0:
            stim_env = np.log10(stim_env + 1.0)

        # calculate stim avg
        #
        # Before Do calculation, we want to check if we got the correct input
        tempXsize = stim_env.shape[0]
        if tempXsize != nband:
            print('Data error, first data file needs to be stimuli, second needs to be response.')
            errFlg = 1
            return
        stim_avg = stim_avg + np.sum(stim_env * DDS[n]['ntrials'], axis=1)
        count_avg = count_avg + (nt + 2 * sil_window) * DDS[n]['ntrials']

        # then calculate response_avg
        if DDS[n]['ntrials'] > 1:
            temp = np.sum(psth_rec, axis=1)    # Given that the stimulus auto correlation multiplies by ntrials - this should be sum and not mean... 
            psth.append(temp[:nt])
        else:
            psth.append(psth_rec[:nt])

        tot_trials = tot_trials + nt + sil_window

        # calculate the total spike/response avg.
        Avg_psth = Avg_psth + np.sum(psth[n][:nt]/DDS[n]['ntrials'])  # If this is average per trials 


        timevary_psth.append(psth[n].shape[1])

        # clear workspace
        stim_env = None
        psth_rec = None

    # -------------------------------------------------------
    #  Calculating Time Varying mean firing rate
    # -------------------------------------------------------
    max_psth_indx = max(timevary_psth)
    whole_psth = np.zeros((len(psth), max_psth_indx))
    count_psth = np.zeros((len(psth), max_psth_indx))

    for nn in range(len(psth)):
        # whole_psth[nn, :psth[nn].shape[1]] = psth[nn] * DDS[nn]['ntrials']
        # count_psth[nn, :psth[nn].shape[1]] = np.ones(psth[nn].shape[1]) * DDS[nn]['ntrials']

        whole_psth[nn, :psth[nn].shape[1]] = np.array(psth[nn]) * DDS[nn]['ntrials']
        count_psth[nn, :psth[nn].shape[1]] = np.ones(len(psth[nn])) * DDS[nn]['ntrials']

    sum_whole_psth = np.sum(whole_psth, axis=0)
    sum_count_psth = np.sum(count_psth, axis=0)

    # Make Delete one averages and smooth at window = 41 - This needs to be a parameter
    smooth_rt = DF_PARAMS.get('smooth_rt', None)
    if smooth_rt is None:
        smooth_rt = 41
    psthsmoothconst = smooth_rt
    if psthsmoothconst % 2 == 0:
        cutsize = 0
    else:
        cutsize = 1
    halfwinsize = np.floor(psthsmoothconst/2)
    wind1 = np.hanning(psthsmoothconst) / np.sum(np.hanning(psthsmoothconst))

    for nn in range(len(psth)):
        count_minus_one = sum_count_psth - count_psth[nn, :]
        count_minus_one[count_minus_one==0] = 1  # Set these to 1 to prevent 0/0
        whole_psth[nn,:] = (sum_whole_psth - whole_psth[nn,:]) / count_minus_one
        svagsm = np.convolve(whole_psth[nn,:], wind1, mode='full')
        whole_psth[nn,:] = svagsm[int(halfwinsize+cutsize):int(len(svagsm)-halfwinsize)+1]

    # save the stim_avg into the data file
    currentPath = os.getcwd()
    outputPath = DF_PARAMS['outputPath']
    if outputPath is not None:
        os.chdir(outputPath)
    else:
        # Take care of empty outputPath case
        currentPath = os.getcwd()
        print(f'No output path specified for intermediate results, defaulting to {currentPath}')
        outputPath = currentPath

    stim_avg = stim_avg / count_avg
    constmeanrate = Avg_psth / tot_trials
    if psth_option == 0:
        Avg_psth_out = constmeanrate
    else:
        Avg_psth_out = whole_psth

    Avg_psth = whole_psth

    np.savez_compressed(os.path.join(outputPath, 'stim_avg.npz'), stim_avg=stim_avg, Avg_psth=Avg_psth, constmeanrate=constmeanrate)

    flat_psth = [item for sublist in psth for item in sublist]
    return stim_avg, Avg_psth_out, psth, errFlg


def df_Check_And_Load(file_name):
    # Check if the file exists
    if not os.path.exists(file_name):
        filename = f"The given file: {file_name} does not exist."
        raise FileNotFoundError(filename)
    
    _, ext = os.path.splitext(file_name)
    
    if ext == ".npy":
        stim = np.load(file_name)
    elif ext in ['.dat', '.txt']:
        stim = np.loadtxt(file_name)
    elif ext == '.mat':
        stimMat = sio.loadmat(file_name)
        flds = list(stimMat.keys())
        
        if len(flds) == 1:
            stim = stimMat[flds[0]]
        elif len(flds) > 1:
            stim = [stimMat[fld] for fld in flds]
    else:
        raise TypeError('Wrong data file type.')
        
    return stim
