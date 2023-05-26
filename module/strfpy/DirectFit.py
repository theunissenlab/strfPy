
# Converted using ChatGPT from direct_fit.m
# 20230321

import os
import numpy as np
import time
import scipy.signal as sio

from .cache import df_create_stim_cache_file, df_create_spike_cache_file, df_checksum, df_dir_of_caches
from .calcAvg import df_cal_AVG
from .calcAutoCorr import df_cal_AutoCorr, df_cal_AutoCorrSep
from .calcCrossCorr import df_cal_CrossCorr, df_fft_AutoCrossCorr
from .calcStrf_script import calcStrfs



def direct_fit(params):
    global DF_PARAMS
    DF_PARAMS = params

    DS = params['DS']
    NBAND = params['NBAND']
    Tol_val = params['Tol_val']
    setSep = params['setSep']
    TimeLag = params['TimeLag']
    TimeLagUnit = params['TimeLagUnit']
    timevary_PSTH = params['timevary_PSTH']
    smooth_rt = params['smooth_rt']
    ampsamprate = params['ampsamprate']
    respsamprate = params['respsamprate']

    # the intermediate result path
    outputPath = params['outputPath']

    # OnlyOne flag used for avoiding overfitting
    OnlyOne = 1

    # =========================================================
    # calculate avg. of stimulus and response that used later
    # =========================================================

    stim_avg, avg_psth, psth, errFlg = df_cal_AVG(DS, DF_PARAMS)

    # Check if cal_Avg ends normally

    if errFlg == 1:
        print('df_cal_AVG ended with error!')

    # =========================================================
    # Now calculating stimulus AutoCorr.
    # =========================================================

    if ampsamprate is None:
        ampsamprate = 1000

    if TimeLagUnit == 'msec':
        twindow = [-round(TimeLag*ampsamprate/1000), round(TimeLag*ampsamprate/1000)]
    elif TimeLagUnit == 'frame':
        twindow = [-TimeLag, TimeLag]

    if setSep == 0:  # Nonseparable space-time algorithm
        print('Now calculating stim autocorrelation')
        do_long_way = 1
        cached_dir, maxsize = df_dir_of_caches()
        autocorr_start_time = time.process_time()
        hashes_of_stims = df_create_stim_cache_file(outputPath, DS)

        # if cached_dir != 'null':
        #     do_long_way = 0
        #     loaded, order = np.sort(hashes_of_stims)  # Sort to make the checksum invarient to dataset shuffling
        #     n_trial_array = get_ntrials(DS)
        #     checksum_for_autocorr_calc = df_checksum(df_load_function_text('df_cal_AutoCorr'), loaded, n_trial_array[order], stim_avg, twindow, NBAND)  # Sort the ntrial array the same way as the stimuli
        #     CS, errFlg = do_cached_calc_checksum_known('df_cal_AutoCorr', checksum_for_autocorr_calc, 1, DS, stim_avg, twindow, NBAND)
        # else:
        CS, errFlg = df_cal_AutoCorr(1, DS, stim_avg, twindow, NBAND, PARAMS=DF_PARAMS)

        autocorr_end_time = time.process_time()
        print('The autocorrelation took', autocorr_end_time - autocorr_start_time, 'seconds.')
        currentPath = os.getcwd()
        if outputPath is not None and outputPath != '':
            os.chdir(outputPath)
        else:
            print('Saving output to Output Dir.')
            os.makedirs('Output', exist_ok=True)
            os.chdir('Output')
            outputPath = os.getcwd()

        np.savez('Stim_autocorr.npz', CS=CS)
        os.chdir(currentPath)

        # Check if df_cal_AutoCorr ends normally
        if errFlg == 1:
            print('df_cal_AutoCorr ended in failure!')

        # Done calculation of stimulus AutoCorr

        # Now calculating stimulus spike CrossCorr
        # Let's assume that if the user has caching on and is using the GUI
        # that they might want to evaluate df_cal_CrossCorr more than once; so:
        # Main code
        cached_dir = "null"

        # cache_crosscorr = not strcmp(cached_dir, 'null')  # This is the flag to say if we'll cache results specific to the current spike train.

        hashes_of_stims = df_create_stim_cache_file(outputPath, DS)
        hashes_of_spikes = df_create_spike_cache_file(outputPath, DS)

        smooth_rt = 41 if smooth_rt is None else smooth_rt

        # if psth_option is None:
        if timevary_PSTH == 0:
            psth_option = 0
        else:
            psth_option = 1

        # DID NOT IMPLEMENT:
        # checksum_CrossCorr = df_checksum(df_load_function_text('df_cal_CrossCorr'), hashes_of_spikes, hashes_of_stims, twindow, smooth_rt, psth_option)

        if not cached_dir == 'null':
            pass
            # #[CSR, CSR_JN, errFlg]= do_cached_calc_checksum_known('df_cal_CrossCorr',checksum_CrossCorr,DS,stim_avg,avg_psth,psth,twindow,NBAND);
            # CSR, CSR_JN, errFlg = df_do_locally_cached_calc_checksum_known(df_get_local_cache_dir(), 'df_cal_CrossCorr', checksum_CrossCorr, DS, stim_avg, avg_psth, psth, twindow, NBAND)
            # np.save(os.path.join(outputPath, 'StimResp_crosscorr.npy'), CSR)
            # np.save(os.path.join(outputPath, 'SR_crosscorrJN.npy'), CSR_JN)
        else:
            CSR, CSR_JN, errFlg = df_cal_CrossCorr(DS, DF_PARAMS, stim_avg, avg_psth, psth, twindow, NBAND)

        # Check if df_cal_CrossCorr ends normally
        if errFlg == 1:
            # set(handles.figure1, 'Pointer', 'Arrow')
            return

        # Done calculation of stimulus-spike CrossCorr in GUI window
        print('Calculating strfs for each tol value.')

        calcStrfs(DF_PARAMS, CS, CSR, CSR_JN)

        calculation_endtime = time.process_time()
        
        print(f'The STRF calculation took {calculation_endtime - autocorr_start_time} seconds.')
    
    else: 
        # Separable space-time algorithm
        # Provide Space-time separability algorithm to estimate STRF
        # [CSspace, CStime, errFlg] = df_cal_AutoCorrSep(DS, stim_avg, twindow, NBAND, 1)
        print('Now calculating stim autocorrelation')
        do_long_way = 1
        cached_dir, maxsize = df_dir_of_caches()
        autocorr_start_time = time.process_time()
        hashes_of_stims = df_create_stim_cache_file(outputPath, DS)

        if cached_dir != 'null':
            do_long_way = 0
            loaded, order = np.sort(hashes_of_stims)  # Sort to make the checksum invarient to dataset shuffling
            n_trial_array = get_ntrials(DS)
            checksum_for_autocorr_calc = df_checksum(df_load_function_text('df_cal_AutoCorrSep'), 'df_cal_AutoCorrSep', loaded, n_trial_array[order], stim_avg, twindow, NBAND)  # Sort the ntrial array the same way as the stimuli
            CSspace, CStime, errFlg = do_cached_calc_checksum_known('df_cal_AutoCorrSep', checksum_for_autocorr_calc, DS, stim_avg, twindow, NBAND, 1)
        else:
            CSspace, CStime, errFlg = df_cal_AutoCorrSep(DS, stim_avg, twindow, NBAND, 1)

        autocorr_end_time = time.process_time()
        print(f'The autocorrelation took {autocorr_end_time - autocorr_start_time} seconds.')

        # Check if df_cal_AutoCorrSep ends normally
        if errFlg == 1:
            print('df_cal_AutoCorrSep ended with error!')

        # Calculate cross-correlation between stimuli and spike


        #  Let's assume that if the user has caching on and is using the GUI
        #  that they might want to evaluate df_cal_CrossCorr more than once; so:
        cache_crosscorr = (cached_dir != 'null')  # This is the flag to say if we'll cache results specific to the current spike train.
        if 'psth_option' not in locals():
            if timevary_PSTH == 0:
                psth_option = 0
            else:
                psth_option = 1

        hashes_of_stims = df_create_stim_cache_file(outputPath,DS)
        hashes_of_spikes = df_create_spike_cache_file(outputPath,DS)
        checksum_CrossCorr = df_checksum('sep', df_load_function_text('df_cal_CrossCorr'), hashes_of_spikes, hashes_of_stims, twindow, smooth_rt, psth_option)

        if smooth_rt is None or smooth_rt == "":
            smooth_rt = 41

        if cache_crosscorr:
            CSR, CSR_JN, errFlg = df_do_locally_cached_calc_checksum_known(df_get_local_cache_dir(), 'df_cal_CrossCorr', checksum_CrossCorr, DS, stim_avg, avg_psth, psth, twindow, NBAND)
            np.save(os.path.join(outputPath, 'StimResp_crosscorr.mat'), 'CSR')
            np.save(os.path.join(outputPath, 'SR_crosscorrJN.mat'), 'CSR_JN')
        else:
            CSR, CSR_JN, errFlg = df_cal_CrossCorr(DS, stim_avg, avg_psth, psth, twindow, NBAND)

        # Check if df_cal_CrossCorr ends normally
        if errFlg != 1:
            # Now call calStrfSep_script to calculate STRF, STRF_JN, STRF_JNstd for each tol value
            
            calcfd = os.path.join(os.getcwd(), 'strfpy', 'calcStrf.py')
            exec(open(calcfd).read())

    strfFiles = [None]*len(Tol_val)
    for k in range(len(Tol_val)):
        fname = f'{outputPath}/strfResult_Tol{k+1}.npz'

        strfFiles[k] = fname

    return strfFiles


# =============================================
# =============================================

# def df_load_function_text(function_name):
#     function_path = os.path.abspath(function_name)
#     with open(function_path, 'r') as file:
#         out = file.read()
#     return out
def df_load_function_text(function_name):
    with open(function_name) as f:
        out = f.read()
    return out



# instead of calcStrf.py, try:
def calc_strf(params):

    global DF_PARAMS
    DF_PARAMS = params

    DS = params.DS
    NBAND = params.NBAND
    Tol_val = params.Tol_val
    setSep = params.setSep
    TimeLag = params.TimeLag
    TimeLagUnit = params.TimeLagUnit
    timevary_PSTH = params.timevary_PSTH
    smooth_rt = params.smooth_rt
    ampsamprate = params.ampsamprate
    respsamprate = params.respsamprate

    # the intermediate result path
    outputPath = params.outputPath
    nstd_val = 0.5

    # ===========================================
    # FFT Auto-correlation and Cross-correlation
    # ===========================================

    #pack;
    checksum_fft_ACC = df_checksum(checksum_CrossCorr, round(TimeLag*ampsamprate/1000), nstd_val)

    cache_crosscorr = 0
    if cache_crosscorr:
        fstim, fstim_spike, stim_spike_JNf = df_do_locally_cached_calc_checksum_known(
            df_get_local_cache_dir, 'df_fft_AutoCrossCorr', checksum_fft_ACC, CS,
            CSR, CSR_JN, round(TimeLag*ampsamprate/1000), NBAND, nstd_val)
    else:
        fstim, fstim_spike, stim_spike_JNf = df_fft_AutoCrossCorr(
            CS, CSR, CSR_JN, round(TimeLag*ampsamprate/1000), NBAND, nstd_val)
    print('Done df_fft_AutoCrossCorr.')

    # clear some memory
    del CSR, CSR_JN, CS

    #pack;
    # ===========================================
    #  Prepare for call STRF_calculation
    # ===========================================
    nb = NBAND
    TimeLagUnit = DF_PARAMS['TimeLagUnit']
    if TimeLagUnit == 'msec':
        nt = 2*round(TimeLag*ampsamprate/1000) + 1
    else:
        nt = 2*round(TimeLag) + 1
    nJN = len(DS)
    stim_size = fstim.shape
    stim_spike_size = fstim_spike.shape
    stim_spike_JNsize = stim_spike_JNf.shape

    # ===========================================
    # Get tolerance values
    # ===========================================
    Tol_val = DF_PARAMS['Tol_val']
    ntols = len(Tol_val)
    outputPath = DF_PARAMS['outputPath']
    if not outputPath:
        print('Saving output to Output Dir.')
        os.mkdir('Output')
        outputPath = os.path.join(os.getcwd(), 'Output')

    # ===========================================
    print('Calculating STRF for each tol value...')
    nf = (nt-1)//2 + 1
    hashes_of_stims = df_create_stim_cache_file(outputPath, DS)
    hashes_of_spikes = df_create_spike_cache_file(outputPath, DS)

    use_more_memory = 1  #Turning this off will break the re-using subspace special option.
    use_alien_space = DF_PARAMS.get('use_alien_space', False)
    alien_space_file = DF_PARAMS.get('alien_space_file', '')
    if not use_alien_space:
        use_alien_space = 0
    if use_alien_space:
        cached_dir = df_dir_of_caches
        loaded_alien_pointer = np.load(alien_space_file, allow_pickle=True)
        if 'original_subspace_checksum' not in loaded_alien_pointer:
            msg = f'Error: STRFPAK expected the file "{alien_space_file}"\n\
                to be a well-formed subspace file.'
            raise Exception(msg)
        hash_of_usv = loaded_alien_pointer['original_subspace_checksum']
        to_load_file = os.path.join(cached_dir, f'{hash_of_usv}.npy')
        if os.path.exists(to_load_file):
            loaded = np.load(to_load_file, allow_pickle=True)
            big_u_alien = loaded.out1
            big_s_alien = loaded.out2
            big_v_alien = loaded.out3
            max_stimnorm_alien = loaded.out4
            big_stim_mat_alien = loaded.out5
            max_s_alien = loaded.out6
            del loaded
            
            if len(big_u_alien) != nf:
                msg = 'Error: mismatch in time domain between the alien subspace and the current subspace.'
                raise ValueError(msg)
                
            if big_u_alien[0].shape[0] != nb:
                msg = 'Error: mismatch in space domain between the alien subspace and the current subspace.'
                raise ValueError(msg)
        else:
            msg = 'Error: STRFPAK needed a cache file of the alien subspace to use, \n' \
                'but it has been deleted since the beginning of this STRFPAK session.\n' \
                'Try increasing the size of your cache to fix this problem.'
            raise ValueError(msg)

    if use_more_memory:
        big_u = []
        big_s = []
        big_v = []

        # checksum_usv = df_checksum(df_load_function_text('df_make_big_usv'),hashes_of_spikes,hashes_of_stims,twindow,round(TimeLag*ampsamprate/1000),nstd_val)
        checksum_usv = df_checksum(df_load_function_text('df_make_big_usv'), hashes_of_stims, twindow, round(TimeLag*ampsamprate/1000))
        if 1:  # This is being forced so that re-using subspaces will not work only in gui mode.
            big_u, big_s, big_v, max_stimnorm, big_stim_mat, max_s = df_do_cached_calc_checksum_known('df_make_big_usv', checksum_usv, nb, nf, fstim)  # max_s needed here or it won't be cached
            if not use_alien_space:
                original_subspace_tol_vals = Tol_val
                original_subspace_checksum = checksum_usv
                np.save(os.path.join(outputPath, 'subspace.mat'), 'original_subspace_tol_vals', 'original_subspace_checksum')
        else:
            big_u, big_s, big_v, max_stimnorm, big_stim_mat, max_s = df_make_big_usv(nb, nf, fstim)

    if cache_crosscorr:
        checksum_cal_strf = df_checksum(checksum_fft_ACC, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN)
        if use_alien_space:
            checksum_cal_strf = df_checksum(checksum_cal_strf, loaded_alien_pointer)

    for itol in range(1, ntols+1):
        tol = Tol_val[itol-1]
        if use_alien_space:
            if tol not in loaded_alien_pointer.original_subspace_tol_vals:
                msg = f'Error: STRFPAK was asked to calculate a STRF using the tol value {tol}.\n'
                msg += f'using a previously-computed subspace, but that subspace used only the tol values:\n'
                msg += f'{loaded_alien_pointer.original_subspace_tol_vals}\n\n'
                msg += '(If you intended to use a new tol value with the alien subspace, disable this alarm by typing\n'
                msg += f'"edit {mfilename}" and commenting out this error. STRFPAK won''t balk, but\n'
                msg += 'we can''t think of a reason to use an alien subspace other than to get the same regularization bias\n'
                msg += 'for two different stim ensembles, and using different tol values defeats the point here.)'
                raise ValueError(msg)
        
        print(f'Now calculating STRF for tol_value: {tol}')

        # =======================================
        # Calculate strf for each tol val.
        # =======================================
        checksum_this_tol = df_checksum(checksum_cal_strf, df_load_function_text('df_cal_Strf'), df_load_function_text('df_cal_Strf_cache2'), tol)
        if not cache_crosscorr:
            if not use_more_memory:
                STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = df_cal_Strf(fstim, fstim_spike, stim_spike_JNf, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol)
            else:
                # [STRF_Cell, STRFJN_Cell, STRFJNstd_Cell] = df_cal_Strf_use_cache(fstim, fstim_spike, stim_spike_JNf, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol, big_u, big_s, big_v, max_stimnorm, big_stim_mat)
                if not use_alien_space:
                    STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = df_cal_Strf_cache2(fstim, fstim_spike, stim_spike_JNf, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol, big_u, big_s, big_v, max_s)
                else:
                    STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = df_cal_Strf_cache2(fstim, fstim_spike, stim_spike_JNf, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol, big_u, big_s, big_v, max_s, big_u_alien, big_s_alien, big_v_alien, max_s_alien)

        else:
            if not use_more_memory:
                STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = df_do_locally_cached_calc_checksum_known(df_get_local_cache_dir, 'df_cal_Strf', checksum_this_tol, fstim, fstim_spike, stim_spike_JNf, stim_size, stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol)
            else:
                if not use_alien_space:
                    STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = df_do_locally_cached_calc_checksum_known(df_get_local_cache_dir,'df_cal_Strf_cache2',checksum_this_tol,fstim,
                        fstim_spike, stim_spike_JNf,stim_size, stim_spike_size,
                        stim_spike_JNsize, nb, nt, nJN, tol,big_u,big_s,big_v,max_s)
                else:
                    STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = df_do_locally_cached_calc_checksum_known(df_get_local_cache_dir,'df_cal_Strf_cache2',checksum_this_tol,fstim,
                        fstim_spike, stim_spike_JNf,stim_size, stim_spike_size,
                        stim_spike_JNsize, nb, nt, nJN, tol,big_u,big_s,big_v,max_s,big_u_alien,big_s_alien,big_v_alien,max_s_alien)
        
        print(f"Done calculation of STRF for tol_value: {tol}\n")

        sfilename = f"strfResult_Tol{itol}.mat"
        strfFiles = os.path.join(outputPath, sfilename)
        sio.savemat(strfFiles, {'STRF_Cell': STRF_Cell, 'STRFJN_Cell': STRFJN_Cell, 'STRFJNstd_Cell': STRFJNstd_Cell})

        if cache_crosscorr:
            strf_checksum = checksum_this_tol
            posslash = strfFiles.rfind('/')
            the_dir = strfFiles[:(posslash+1)]
            the_name = strfFiles[(posslash+1):]
            strf_hash_filename = the_dir + 'hash_of_' + the_name
            sio.savemat(strf_hash_filename, {'strf_checksum': strf_checksum})

        del STRF_Cell, STRFJN_Cell, STRFJNstd_Cell




