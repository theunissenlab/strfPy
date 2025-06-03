
# Converted using ChatGPT from direct_fit.m
# May-June 2025 Fixed and stream-lined by Frederic Theunissen

import time

from .calcAvg import df_cal_AVG
from .calcAutoCorr import df_cal_AutoCorrJN
from .calcCrossCorr import df_cal_CrossCorr
from .calcStrf_script import calcStrfs



def direct_fit(params):

    # get paramters used frequently
    DS = params['DS']
    NBAND = params['NBAND']
    Tol_val = params['Tol_val']
    TimeLag = params['TimeLag']
    TimeLagUnit = params['TimeLagUnit']
    ampsamprate = params['ampsamprate']

    # the intermediate result path
    outputPath = params['outputPath']

    # =========================================================
    # calculate avg. of stimulus and response that used later
    # =========================================================

    stim_avg, avg_psth, psth, errFlg = df_cal_AVG(DS, params)
    # The psth is not normalized - i.e it is the sum of all trials and trials can have variable length - use weights to normalize

    # =========================================================
    # Now calculating stimulus AutoCorr.
    # =========================================================

    if TimeLagUnit == 'msec':
        twindow = [-round(TimeLag*ampsamprate/1000), round(TimeLag*ampsamprate/1000)]
    elif TimeLagUnit == 'frame':
        twindow = [-TimeLag, TimeLag]

    print('Now calculating stim auto-correlation')
    autocorr_start_time = time.process_time()
    CS, CS_JN = df_cal_AutoCorrJN(DS, stim_avg, twindow, NBAND, params)
    autocorr_end_time = time.process_time()
    params['CS'] = CS
    params['CS_JN'] = CS_JN
    print('The auto-correlation took', autocorr_end_time - autocorr_start_time, 'seconds.')

    # =========================================================
    # Now calculating stimulus-response CrossCorr.
    # =========================================================
    print('Now calculating stim-response cross-correlation')
    CSR, CSR_JN, errFlg = df_cal_CrossCorr(DS, params, stim_avg, avg_psth, psth, twindow, NBAND)
    params['CSR'] = CSR
    params['CSR_JN'] = CSR_JN
    crosscorr_end_time = time.process_time()
    print('The cross-correlation took', crosscorr_end_time - autocorr_end_time, 'seconds.')

    # =========================================================
    # Now calculating the STRFs.
    # =========================================================

    print('Calculating strfs for each tol value.')

    calcStrfs(params, CS, CS_JN, CSR, CSR_JN)
    calculation_endtime = time.process_time()
        
    print(f'The STRF calculation took {calculation_endtime - crosscorr_end_time} seconds.')
    

    strfFiles = [None]*len(Tol_val)
    for k in range(len(Tol_val)):
        fname = f'{outputPath}/strfResult_Tol{k+1}.npz'

        strfFiles[k] = fname

    return strfFiles

