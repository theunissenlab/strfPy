
# # Converted using ChatGPT from df_calStrf_script.m
# 20230404

import numpy as np
import os
import scipy.io as sio

from .cache import df_create_stim_cache_file, df_create_spike_cache_file, df_checksum, df_dir_of_caches
from .calcAvg import df_cal_AVG
from .calcAutoCorr import df_cal_AutoCorr, df_cal_AutoCorrSep
from .calcCrossCorr import df_cal_CrossCorr, df_fft_AutoCrossCorr
from .calcStrf import df_cal_Strf
#from .DirectFit import df_load_function_text


def calcStrfs(params, CS, CSR, CSR_JN):
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

    nstd_val = 0.5

    # ===========================================
    # FFT Auto-correlation and Cross-correlation
    # ===========================================

    #pack;

    # DID NOT IMPLEMENT
    # ---------------------
    # checksum_CrossCorr = df_checksum(df_load_function_text('df_cal_CrossCorr'),hashes_of_spikes,hashes_of_stims,twindow,smooth_rt,psth_option);

    # checksum_fft_ACC = df_checksum(checksum_CrossCorr, round(TimeLag*ampsamprate/1000), nstd_val)
    # ---------------------
    if TimeLagUnit == 'msec':
        twindow = round(TimeLag*ampsamprate/1000)
    elif TimeLagUnit == 'frame':
        twindow = round(TimeLag)
    nt = 2*twindow + 1

    cache_crosscorr = 0
    if cache_crosscorr:
        fstim, fstim_spike, stim_spike_JNf = df_do_locally_cached_calc_checksum_known(
            df_get_local_cache_dir, 'df_fft_AutoCrossCorr', checksum_fft_ACC, CS,
            CSR, CSR_JN, round(TimeLag*ampsamprate/1000), NBAND, nstd_val)
    else:
        fstim, fstim_spike, stim_spike_JNf = df_fft_AutoCrossCorr(
            CS, CSR, CSR_JN, twindow, NBAND, nstd_val)
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



    # use_more_memory = 1  #Turning this off will break the re-using subspace special option.
    use_more_memory = 0
    
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
   
   
    # ------------------
    # NOT IMPLEMENTED
    # ------------------
    # if use_more_memory: 

    #     big_u = []
    #     big_s = []
    #     big_v = []

    #     # checksum_usv = df_checksum(df_load_function_text('df_make_big_usv'),hashes_of_spikes,hashes_of_stims,twindow,round(TimeLag*ampsamprate/1000),nstd_val)
    #     checksum_usv = df_checksum(df_load_function_text('df_make_big_usv'), hashes_of_stims, twindow, round(TimeLag*ampsamprate/1000))
    #     if 1:  # This is being forced so that re-using subspaces will not work only in gui mode.
    #         big_u, big_s, big_v, max_stimnorm, big_stim_mat, max_s = df_do_cached_calc_checksum_known('df_make_big_usv', checksum_usv, nb, nf, fstim)  # max_s needed here or it won't be cached
    #         if not use_alien_space:
    #             original_subspace_tol_vals = Tol_val
    #             original_subspace_checksum = checksum_usv
    #             np.save(os.path.join(outputPath, 'subspace.mat'), 'original_subspace_tol_vals', 'original_subspace_checksum')
    #     else:
    #         big_u, big_s, big_v, max_stimnorm, big_stim_mat, max_s = df_make_big_usv(nb, nf, fstim)
    # ------------------
    # NOT IMPLEMENTED
    # ------------------



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
                errordlg(msg)
                raise ValueError(msg)
        
        print(f'Now calculating STRF for tol_value: {tol}')

        # =======================================
        # Calculate strf for each tol val.
        # =======================================

        # ------------------
        # NOT IMPLEMENTED
        # ------------------
        # checksum_this_tol = df_checksum(checksum_cal_strf, df_load_function_text('df_cal_Strf'), df_load_function_text('df_cal_Strf_cache2'), tol)
        
        
        if not cache_crosscorr:
            if not use_more_memory:
                STRF_Cell, STRFJN_Cell, STRFJNstd_Cell = df_cal_Strf(
                    DF_PARAMS, fstim, fstim_spike, stim_spike_JNf, stim_size, 
                    stim_spike_size, stim_spike_JNsize, nb, nt, nJN, tol)
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

        sfilename = f"strfResult_Tol{itol}.npz"
        strfFiles = os.path.join(outputPath, sfilename)
        np.savez_compressed(strfFiles, STRF_Cell=STRF_Cell, STRFJN_Cell=STRFJN_Cell, STRFJNstd_Cell=STRFJNstd_Cell)

        # if cache_crosscorr:
        #     strf_checksum = checksum_this_tol
        #     posslash = strfFiles.rfind('/')
        #     the_dir = strfFiles[:(posslash+1)]
        #     the_name = strfFiles[(posslash+1):]
        #     strf_hash_filename = the_dir + 'hash_of_' + the_name
        #     sio.savemat(strf_hash_filename, {'strf_checksum': strf_checksum})

        del STRF_Cell, STRFJN_Cell, STRFJNstd_Cell





