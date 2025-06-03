
# chatGPT converted df_cal_AutoCorr.m
# + edit

import os
import numpy as np
from .calcAvg import calculateLeaveOneOutAverages, df_Check_And_Load
from scipy.signal import correlate, correlation_lags


def df_cal_AutoCorrJN(DS, stim_avg, twindow, nband, PARAMS):

    # Initialize the output and set its size
    # Get the total data files
    filecount = len(DS)
    
    # Temporal axis range
    tot_corr = int(np.diff(twindow) + 1)
    
    # Spatial axis range
    spa_corr = int((nband * (nband - 1)) // 2 + nband)
    
    # Initialize CS and CSJN variable
    CS = np.zeros((spa_corr, tot_corr))
    CS_ns = np.zeros(tot_corr)

    # JN varriables
    CS_JN = [np.zeros((spa_corr, tot_corr)) for i in range(filecount)]
    CS_JN_ns = [np.zeros(tot_corr) for i in range(filecount)]
    
    # Do calculation. The algorithm is based on FET's dcp_stim.c
    for fidx in range(filecount):  # Loop through all data files
        # load stimulus file
        stim_env = df_Check_And_Load(DS[fidx]['stimfiles'])
        weight = df_Check_And_Load(DS[fidx]['weightfiles'])

        nlen = np.shape(stim_env)[1]
        stimval = np.zeros((nband, nlen))

        # subtract mean and normalize by the sqrt of weight.
        stimval = (stim_env[0:nband, :] - np.reshape(stim_avg[0:nband], (nband, 1)) )* np.sqrt(weight[0:nlen])

        # The normalization vector: calculated and stored
        xcorr_w = correlate(np.sqrt(weight[0:nlen]), np.sqrt(weight[0:nlen]), mode='full')
        lags = correlation_lags(nlen, nlen, mode="full")
        itlow = np.argwhere(lags == twindow[0])[0][0]
        ithigh = np.argwhere(lags == twindow[1])[0][0]+1
        CS_JN_ns[fidx] = xcorr_w[itlow:ithigh]
        CS_ns += xcorr_w[itlow:ithigh]

        xb = 0
        for ib1 in range(nband):
            for ib2 in range(ib1, nband):
                xcorr_s = correlate(stimval[ib1,:], stimval[ib2,:], mode="full")

                CS_JN[fidx][xb,:] = xcorr_s[itlow:ithigh]
                CS[xb, :] += xcorr_s[itlow:ithigh]
                xb += 1

    # Finish the JN Calculations
    for fidx in range(filecount):
        norm = (CS_ns-CS_JN_ns[fidx])
        for i in range(spa_corr):
            CS_JN[fidx][i,:] = (CS[i,:]-CS_JN[fidx][i,:])/norm
    
    # Finish the normalization
    for i in range(spa_corr):
        CS[i,:] = CS[i,:]/CS_ns

    # ========================================================
    # save stimulus auto-correlation matrix into file
    # ========================================================
    currentPath = os.getcwd()
    outputPath = PARAMS['outputPath']

    if outputPath:
        os.chdir(outputPath)
    else:
        print('Saving output to Output Dir.')
        os.makedirs('Output', exist_ok=True)
        os.chdir('Output')
        outputPath = os.getcwd()

    np.save('Stim_autocorr.npy', CS)
    np.save('CS_JN.npy', CS_JN)
    os.chdir(currentPath)

    # ========================================================
    # END OF CAL_AUTOCORR
    # ========================================================

    return CS, CS_JN



# function definition
def df_cal_AutoCorrSep(DS, stim_avg, twindow, nband, JN_flag=None):
    # global variable
    global DF_PARAMS
    
    # check for valid inputs
    errFlg = 0
    if not DS:
        print("ERROR: Please enter non-empty data filename")
        errFlg = 1
        return None, None, errFlg
    
    if JN_flag is None:
        JN_flag = 0
    
    # check if stim_avg is calculated or not
    if stim_avg is None:
        stim_avg, _, _ = df_cal_AVG(DS, nband)
    
    # initialize output
    filecount = len(DS)
    tot_corr = np.diff(twindow) + 1
    spa_corr = int((nband * (nband - 1))/2 + nband)
    CSspace = np.zeros((nband, nband))
    CStime = np.zeros((1, tot_corr*2-1))
    CSspace_ns = 0
    CStime_ns = np.zeros((1, tot_corr*2-1))
    
    # do calculation
    for fidx in range(filecount):
        # load stimulus file
        stim_env = df_Check_And_Load(DS[fidx].stimfiles)
        nlen = DS[fidx].nlen
        xb = 1
        stimval = np.zeros((nband, nlen))
        
        # check input data
        thisLength = stim_env.shape[1]
        if thisLength < nlen:
            answ = input('Data Error: Please check your input data by clicking "Get Files" Button in the main window: The first data file need to be stimuli and the second data file need to be its corresponding response file. If you made a mistake, please type "clear all" or hit "reset" button first and then choose input data again. Otherwise, we will truncate the longer length. Do you want to continue? [Y/N]: ')
            if answ.lower() == 'n':
                errFlg = 1
                return None, None, errFlg
        
        nlen = min(nlen, thisLength)
        
        # subtract mean of stim from each stim
        for tgood in range(nlen):
            stimval[:, tgood] = stim_env[0:nband, tgood] - stim_avg[0:nband]
        
        # do autocorrelation calculation
        CSspace += DS[fidx].ntrials * np.matmul(stimval, stimval.T)
        CSspace_ns += DS[fidx].ntrials * nlen
        
        for ib1 in range(nband):
            CStime += DS[fidx].ntrials * correlate(stimval[ib1,:], stimval[ib1,:], mode='full')[twindow[0]:twindow[1]+1]
        CStime_ns += DS[fidx].ntrials * nband * correlate(np.ones(nlen), np.ones(nlen), mode='full')[twindow[0]:twindow[1]+1]
        
        # clear workspace
        del stim_env, stimval
        
    # normalize CS matrix
    CStime = np.divide(CStime, CStime_ns + (CStime == 0))
    CSspace = np.divide(CSspace, CSspace_ns + (CSspace == 0))
    CStime = np.divide(CStime, np.mean(np.diag(CSspace)))


    currentPath = os.getcwd()
    outputPath = DF_PARAMS['outputPath']
    if outputPath:
        os.chdir(outputPath)
    else:
        print('Saving output to Output Dir.')
        os.makedirs('Output', exist_ok=True)
        os.chdir('Output')
        outputPath = os.getcwd()

    np.savez('Stim_autocorr.npz', CSspace=CSspace, CStime=CStime)
    os.chdir(currentPath)

    return CSspace, CStime, errFlg