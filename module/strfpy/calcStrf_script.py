
# # Converted using ChatGPT from df_calStrf_script.m
# 20230404

import numpy as np
import os


from .cache import df_create_stim_cache_file, df_create_spike_cache_file, df_checksum, df_dir_of_caches
from .calcStrf import df_cal_Strf
from .calcCrossCorr import df_fft_AutoCrossCorr


def calcStrfs(params, CS, CS_JN, CSR, CSR_JN):

    DS = params['DS']
    nb = params['NBAND']
    Tol_val = params['Tol_val']
    TimeLag = params['TimeLag']
    TimeLagUnit = params['TimeLagUnit']

    ampsamprate = params['ampsamprate']

    # the intermediate result path
    outputPath = params['outputPath']

    nstd_val = 0.5

    # ===========================================
    # FFT Auto-correlation and Cross-correlation
    # ===========================================

    if TimeLagUnit == 'msec':
        twindow = round(TimeLag*ampsamprate/1000)
    elif TimeLagUnit == 'frame':
        twindow = round(TimeLag)
    nt = 2*twindow + 1

    
    fstim, fstim_JN, fstim_spike, stim_spike_JNf = df_fft_AutoCrossCorr(
            CS, CS_JN, CSR, CSR_JN, twindow, nb, nstd_val)
    print('Done df_fft_AutoCrossCorr.')


    # ===========================================
    #  Prepare for call STRF_calculation
    # ===========================================
    TimeLagUnit = params['TimeLagUnit']
    if TimeLagUnit == 'msec':
        nt = 2*round(TimeLag*ampsamprate/1000) + 1
    else:
        nt = 2*round(TimeLag) + 1
    nJN = len(DS)
    stim_spike_size = fstim_spike.shape
    stim_spike_JNsize = stim_spike_JNf.shape

    # ===========================================
    # Get tolerance values
    # ===========================================
    Tol_val = params['Tol_val']
    ntols = len(Tol_val)
    outputPath = params['outputPath']
    if not outputPath:
        print('Saving output to Output Dir.')
        os.mkdir('Output')
        outputPath = os.path.join(os.getcwd(), 'Output')

    # ===========================================
    print('Calculating STRF for each tol value...')
    nf = (nt-1)//2 + 1

    # The stimulus-autocorrelation SVD at each frequency bin (the dominant
    # cost below) does not depend on the tolerance value, so compute STRFs
    # for every tolerance value in one call instead of once per tolerance
    # value -- df_cal_Strf reuses each frequency bin's SVD across the whole
    # sweep rather than recomputing it from scratch per tolerance.
    results = df_cal_Strf(
                params, fstim, fstim_JN, fstim_spike, stim_spike_JNf,
                stim_spike_size, stim_spike_JNsize, nb, nt, nJN, Tol_val)

    for itol in range(1, ntols+1):
        tol = Tol_val[itol-1]
        STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = results[itol-1]

        print(f"Done calculation of STRF for tol_value: {tol}\n")

        sfilename = f"strfResult_Tol{itol}.npz"
        strfFiles = os.path.join(outputPath, sfilename)
        np.savez_compressed(strfFiles, STRF_Cell=STRF_Cell, STRFJN_Cell=STRFJN_Cell, STRFJNstd_Cell=STRFJNstd_Cell)

    return

