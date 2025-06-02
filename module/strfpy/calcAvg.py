# Converted using ChatGPT from df_cal_AVG.m, df_Check_And_Load.m
# 20230404

import numpy as np
import scipy.io as sio
import os
import pandas
from scipy.signal import resample
from scipy.signal.windows import hann


def calculateAverage(dfStrfLab: pandas.DataFrame, apply_log: bool = False):
    """
    Calculate the average stimulus and response value across time.
    Args:
    - dfStrfLab (pandas.DataFrame): the dataframe containing the stimulus and response data.
    - apply_log (bool, optional): whether to take the log of the stimulus data. Default is False.
    Returns:
    - stim_avg (ndarray): average stimulus value which is only function of space.
    - resp_avg (ndarray): average response value which is only function of time.
    - avg_firing_rate (float): the average firing rate of the response.
    """

    # first calculate the average stim (only function of space/frequency)
    stims = dfStrfLab.stim
    if apply_log:
        stims = stims.apply(np.log10)  # potential -inf values if there are zeros
    stim_avgs = stims.apply(np.mean, axis=1)  # get the average of each stim
    stim_nums = stims.apply(
        np.size, axis=1
    )  # get the number of time samples of each stim
    stim_avg = np.average(
        stim_avgs, weights=stim_nums * dfStrfLab.nTrials
    )  # weight each avg by how long the stim was, and how many trials there are

    # then calculate response average
    resp_type = dfStrfLab.type.iloc[0]
    # get the avg response for each stim per trial
    # NB: This assumes dfStrfLab['psth'] is summed and not an average
    # Could alternatively reclaculate using 'trial_psth'
    resp_avg = dfStrfLab[resp_type] / dfStrfLab.nTrials

    # single value which is average response over time
    avg_firing_rate = np.average(
        dfStrfLab[resp_type].apply(np.sum) / dfStrfLab[resp_type].apply(len),
        weights=dfStrfLab.nTrials,
    )

    return stim_avg, resp_avg, avg_firing_rate


def calculateLeaveOneOutAverages(
    dfStrfLab: pandas.DataFrame,
    apply_stim_log: bool = False,
    apply_resp_smoothing: bool = True,
    smooth_rt: int = 41,
):
    #  Calculating Time Varying mean firing rate
    # Args:
    # - dfStrfLab (pandas.DataFrame): the dataframe containing the stimulus and response data.
    # - apply_smoothing (bool, optional): whether to apply smoothing to the time varying average. Default is True.
    # - apply_log (bool, optional): whether to take the log of the stimulus data. Default is False.
    # - smooth_rt (int, optional): the size of the smoothing window. Default is 41.
    # Returns:
    # - avg_resp_minus_one (ndarray): the time varying average response with one trial left out.
    # calculate leave one out average
    # this yields a matrix of indices leaving out the current row
    N = len(dfStrfLab)
    idx = np.arange(1, N) - np.tri(N, N - 1, k=-1, dtype=bool)

    stims = dfStrfLab.stim
    if apply_stim_log:
        stims = stims.apply(np.log10)  # potential -inf values if there are zeros
    stim_avgs = np.vstack(stims.apply(np.mean, axis=1))  # get the average of each stim
    stim_nums = np.asarray(
        stims.apply(np.size, axis=1)
    )  # get the number of time samples of each stim
    num_trials = np.asarray(dfStrfLab.nTrials)

    all_stim_minus_one = stim_avgs[
        idx
    ]  # get the stim average for each stim except the current row
    stim_nums_minus_one = stim_nums[
        idx
    ]  # get the number of time samples for each stim except the current row
    num_trials_minus_one = num_trials[
        idx
    ]  # get the number of trials for each stim except the current row

    # first multiply by the weights (num_trials and num_stims)
    stim_weights = stim_nums_minus_one * num_trials_minus_one
    weighted_stims = all_stim_minus_one * stim_weights[:, :, np.newaxis]
    # now sum over the stims and divide by the summed weights
    weighted_sums = weighted_stims.sum(axis=1)
    avg_stim_minus_one = weighted_sums / stim_weights.sum(axis=1)[:, np.newaxis]

    resp_type = dfStrfLab.type.iloc[0]
    resp_avg = dfStrfLab[resp_type] / dfStrfLab.nTrials
    max_resp_len = (
        dfStrfLab[resp_type].apply(len).max()
    )  # get the max length of the responses
    # stack all the un-normed psth into a matrix (nTrials x nTime)
    # the trials that are smaller than the max len will have trailing zeros
    all_resp = np.vstack(
        dfStrfLab.psth.apply(lambda x: np.pad(x, (0, max_resp_len - len(x))))
    )
    all_counts = np.vstack(
        [
            np.pad(np.ones(num_samp), (0, max_resp_len - num_samp)) * nt
            for num_samp, nt in zip(resp_avg.apply(len), dfStrfLab.nTrials)
        ]
    )

    # get the sum of all the responses except the current row
    # and divide by the sum of all the counts except the current row
    all_resp_minus_one = all_resp[idx, :].sum(axis=1)
    all_counts_minus_one = all_counts[idx, :].sum(axis=1)
    # set all zeros to nan to prevent divide by zero
    all_counts_minus_one[all_counts_minus_one == 0] = np.nan
    avg_resp_minus_one = all_resp_minus_one / all_counts_minus_one
    # smooth the avg responses with the hann_window
    if apply_resp_smoothing:
        # create a window to do smoothing
        hann_window = np.hanning(smooth_rt) / np.sum(np.hanning(smooth_rt))
        avg_resp_minus_one = np.apply_along_axis(
            lambda m: np.convolve(m, hann_window, mode="same"),
            axis=1,
            arr=avg_resp_minus_one,
        )
    dfStrfLab["stim_loo_avg"] = [
        avg_stim_minus_one[x, :l]
        for (x, l) in zip(range(avg_stim_minus_one.shape[0]), stim_nums)
    ]
    dfStrfLab["psth_loo_avg"] = [
        avg_resp_minus_one[x, :l]
        for (x, l) in zip(range(avg_resp_minus_one.shape[0]), stim_nums)
    ]
    return avg_stim_minus_one, avg_resp_minus_one


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
        print("ERROR: Please enter non-empty data filename")
        errFlg = 1
        return None, None, None, None, errFlg

    NBAND = DF_PARAMS.get("NBAND")
    if nband is None:
        if NBAND is None:
            print("You need assign variable NBAND first.")
            errFlg = 1
            return None, None, None, None, errFlg
        nband = NBAND

    # Add parameter 'psth_option' to specifiy psth noise removal option
    timevary_PSTH = DF_PARAMS.get("timevary_PSTH")
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
    ampsamprate = DF_PARAMS.get("ampsamprate")
    respsamprate = DF_PARAMS.get("respsamprate")
    ndata_files = len(DDS)
    stim_avg = np.zeros(nband)
    count_avg = 0
    tot_trials = 0
    psth = []
    allweights = []
    timevary_psth = []
    Avg_psth = 0

    # calculate the output over all the data files
    for n in range(ndata_files):
        # load stimulus files
        stim_env = df_Check_And_Load(DDS[n]["stimfiles"])
        this_len = stim_env.shape[1]  # get stim duration
        # load response files
        rawResp = df_Check_And_Load(DDS[n]["respfiles"])   # This should be in units of spikes/s - averaged across trials 
        # load weight files
        weight = df_Check_And_Load(DDS[n]["weightfiles"])  # This is the number of of trials at each time point

        if isinstance(rawResp, list):
            spiketrain = np.zeros((DDS[n]["ntrials"], this_len))
            for trial_ind in range(DDS[n]["ntrials"]):
                spiketrain[trial_ind, rawResp[trial_ind]] = np.ones(
                    1, len(rawResp[trial_ind])
                )
            if ampsamprate and respsamprate:
                newpsth = resample(spiketrain.T, ampsamprate, respsamprate)
            else:
                newpsth = spiketrain
            newpsth = newpsth.T  # make sure new response data is trials x T.
            newpsth[newpsth < 0] = 0
            psth_rec = newpsth
            weight = np.ones_like(psth_rec)*DDS[n]["ntrials"]  # Overwrite weight by number of trials.
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
            print(
                "Data error, first data file needs to be stimuli, second needs to be response."
            )
            errFlg = 1
            return
        stim_avg += np.sum(stim_env[:nt] * weight[:nt], axis=1)
        count_avg += np.sum(weight[:nt])

        # then calculate response_avg
        if DDS[n]["ntrials"] > 1:
            temp = np.sum(
                psth_rec, axis=1
            )  # Given that the stimulus auto correlation multiplies by ntrials - this should be sum and not mean - to be checked... the ntrials might have to be incorporated in weights...
            psth.append(temp[:nt])
        else:
            psth.append(psth_rec[:nt])
        
        allweights.append(weight)

        # calculate the total spike/response avg by performing a weights sum.  The psth is already normalized and in units of spikes/s on average
        Avg_psth += np.sum(psth[n][:nt]*weight[:nt])
        tot_trials += np.sum(weight[:nt])

        timevary_psth.append(psth[n].shape[1])

    # -------------------------------------------------------
    #  Calculating Time Varying mean firing rate
    # -------------------------------------------------------
    max_psth_indx = max(timevary_psth)
    whole_psth = np.zeros((len(psth), max_psth_indx))
    count_psth = np.zeros((len(psth), max_psth_indx))

    for nn in range(len(psth)):
        # whole_psth[nn, :psth[nn].shape[1]] = psth[nn] * DDS[nn]['ntrials']
        # count_psth[nn, :psth[nn].shape[1]] = np.ones(psth[nn].shape[1]) * DDS[nn]['ntrials']

        whole_psth[nn, : psth[nn].shape[1]] = np.array(psth[nn]*allweights[nn])  
        count_psth[nn, : psth[nn].shape[1]] = np.array(allweights[nn])

    sum_whole_psth = np.sum(whole_psth, axis=0)
    sum_count_psth = np.sum(count_psth, axis=0)

    # Make Delete one averages and smooth at window = 41 - This needs to be a parameter
    smooth_rt = DF_PARAMS.get("smooth_rt", None)
    if smooth_rt is None:
        smooth_rt = 41
    psthsmoothconst = smooth_rt
    if psthsmoothconst % 2 == 0:
        cutsize = 0
    else:
        cutsize = 1
    halfwinsize = np.floor(psthsmoothconst / 2)
    wind1 = np.hanning(psthsmoothconst) / np.sum(np.hanning(psthsmoothconst))

    for nn in range(len(psth)):
        count_minus_one = sum_count_psth - count_psth[nn, :]
        count_minus_one[count_minus_one == 0] = 1  # Set these to 1 to prevent 0/0
        whole_psth[nn, :] = (sum_whole_psth - whole_psth[nn, :]) / count_minus_one
        svagsm = np.convolve(whole_psth[nn, :], wind1, mode="full")
        whole_psth[nn, :] = svagsm[
            int(halfwinsize + cutsize) : int(len(svagsm) - halfwinsize) + 1
        ]

    # save the stim_avg into the data file
    currentPath = os.getcwd()
    outputPath = DF_PARAMS["outputPath"]
    if outputPath is not None:
        os.chdir(outputPath)
    else:
        # Take care of empty outputPath case
        currentPath = os.getcwd()
        print(
            f"No output path specified for intermediate results, defaulting to {currentPath}"
        )
        outputPath = currentPath

    stim_avg = stim_avg / count_avg
    constmeanrate = Avg_psth / tot_trials
    if psth_option == 0:
        Avg_psth_out = constmeanrate
    else:
        Avg_psth_out = whole_psth

    Avg_psth = whole_psth

    np.savez_compressed(
        os.path.join(outputPath, "stim_avg.npz"),
        stim_avg=stim_avg,
        Avg_psth=Avg_psth,
        constmeanrate=constmeanrate,
    )

    return stim_avg, Avg_psth_out, psth, errFlg


def df_Check_And_Load(file_name):
    # Check if the file exists
    if not os.path.exists(file_name):
        filename = f"The given file: {file_name} does not exist."
        raise FileNotFoundError(filename)

    _, ext = os.path.splitext(file_name)

    if ext == ".npy":
        stim = np.load(file_name)
    elif ext in [".dat", ".txt"]:
        stim = np.loadtxt(file_name)
    elif ext == ".mat":
        stimMat = sio.loadmat(file_name)
        flds = list(stimMat.keys())

        if len(flds) == 1:
            stim = stimMat[flds[0]]
        elif len(flds) > 1:
            stim = [stimMat[fld] for fld in flds]
    else:
        raise TypeError("Wrong data file type.")

    return stim
