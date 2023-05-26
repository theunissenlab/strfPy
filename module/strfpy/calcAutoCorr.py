
# chatGPT converted df_cal_AutoCorr.m
# + edit

import os
import numpy as np
import time
from .calcAvg import df_cal_AVG, df_Check_And_Load
from scipy.signal import correlate

def df_cal_AutoCorr(running_flag, DS, stim_avg, twindow, nband, PARAMS, JN_flag=None):
    global DF_PARAMS
    DF_PARAMS=PARAMS

    # Check if we have valid required input
    t_autocorr_start = time.process_time()
    
    errFlg = 0
    if DS is None:
        raise ValueError('ERROR: Please enter non-empty data filename')
        errFlg = 1
        return
    
    if JN_flag is None:
        JN_flag = 0
    
    # Check whether stim_avg has been calculated or not
    if stim_avg is None:
        # Calculate stim_avg and avg_psth and psth
        stim_avg, avgr, psth = df_cal_AVG(DS, nband)
    
    # Initialize the output and set its size
    # Get the total data files
    filecount = len(DS)
    
    # Temporal axis range
    tot_corr = np.diff(twindow) + 1
    
    # Spatial axis range
    spa_corr = (nband * (nband - 1)) // 2 + nband
    
    # Initialize CS and CSJN variable
    CS = np.zeros((int(spa_corr), int(tot_corr)))
    #CSJN = np.zeros((filecount, spa_corr, tot_corr))
    CS_ns = np.zeros((1, int(tot_corr)))
    #CS_ns_JN = np.zeros((filecount, spa_corr, tot_corr))
    
    # See if we can use the stimuli's hashes instead of the whole stimulus for
    # the checksums of the autocorrelation. (Computing the checksum of a
    # whole stimulus can take at least a second - it's faster this way.)
    
    use_stim_hashes = False
    outputPath = DF_PARAMS['outputPath']
    cached_dir, maxsize = 'null', 10
    # if cached_dir != 'null':
    #     hashes_of_stims = df_create_stim_cache_file(outputPath, DS)
    #     use_stim_hashes = True
    # else:
    #     hashes_of_stims = ['Not used.']
    
    # Do calculation. The algorithm is based on FET's dcp_stim.c
    for fidx in range(filecount):  # Loop through all data files
        # do autocorrelation calculation
        just_load_answer = 0
        
        if cached_dir != 'null':
            pass
        #     checksum_for_1_stim_autocorr = df_checksum(df_load_function_text('df_small_autocorr4'), hashes_of_stims[fidx - 1], nband, np.shape(CS), twindow[1])
        # if use_stim_hashes:
        #     if os.path.exists(os.path.join(cached_dir, f'{checksum_for_1_stim_autocorr}.mat')):
        #         just_load_answer = 1

        # if just_load_answer:
        #     # If the CS_diff is already computed, we don't have to load stimval at all (!)
        #     CS_diff = do_cached_calc_checksum_known('df_small_autocorr4', checksum_for_1_stim_autocorr) * DS[fidx - 1].ntrials
        #     nlen = DS[fidx - 1].nlen
        else:
            # load stimulus file
            stim_env = df_Check_And_Load(DS[fidx]['stimfiles'])

            nlen = np.shape(stim_env)[1]
            stimval = np.zeros((nband, nlen))

            # subtract mean of stim from each stim
            for tgood in range(nlen):
                stimval[:, tgood] = stim_env[0:nband, tgood] - stim_avg[0:nband]
            if cached_dir != 'null':
                pass
                # CS_diff = do_cached_calc_checksum_known('df_small_autocorr4', checksum_for_1_stim_autocorr, stimval, nband, np.shape(CS), twindow[1]) * DS[fidx - 1].ntrials
            else:
                CS_diff = df_small_autocorr4(stimval, nband, np.shape(CS), twindow[1]) * DS[fidx]['ntrials']
        CS = CS + CS_diff

        # Count the total trials for later normalization
        lengthVec = np.ones(nlen)
        # CS_ns[0, :] = CS_ns[0, :] + DS[fidx - 1]['ntrials'] * np.correlate(lengthVec, lengthVec, mode='same')[twindow[0]:twindow[1]]

        xcorr_res = np.correlate(lengthVec, lengthVec, mode="same")
        len_xcorr = len(xcorr_res)
        xcorr_res = xcorr_res[len_xcorr//2 + twindow[0]:len_xcorr//2 + twindow[1]+1]
        # xcorr_res = xcorr_res[(twindow[0] - 1):(twindow[1])]
        # CS_ns[0, (twindow[0] - 1):(twindow[1])] += DS[fidx]['ntrials'] * xcorr_res
        CS_ns[0, :] += DS[fidx]['ntrials'] * xcorr_res




        # clear workspace
        if 'stim_env' in locals():
            del stim_env
        if 'stimval' in locals():
            del stimval
        if 'lengthVec' in locals():
            del lengthVec

    print('Done auto-correlation calculation')

    # ========================================================
    # To normalize CS by CS_ns:
    #   if CS_ns != 0:
    #      CS = CS / CS_ns
    # ========================================================
    # eliminate zero in CS_ns
    nozero_ns = np.isinf(1 / CS_ns) + CS_ns

    # normalize CS matrix
    for i in range(spa_corr):
        CS[i, :] = CS[i, :] / nozero_ns

    # ========================================================
    # save stimulus auto-correlation matrix into file
    # ========================================================
    currentPath = os.getcwd()

    if outputPath:
        os.chdir(outputPath)
    else:
        print('Saving output to Output Dir.')
        os.makedirs('Output', exist_ok=True)
        os.chdir('Output')
        outputPath = os.getcwd()

    np.save('Stim_autocorr.npy', CS)
    os.chdir(currentPath)
    # ========================================================
    # END OF CAL_AUTOCORR
    # ========================================================

    t_autocorr_end = time.process_time()
    #print(f'It took {t_autocorr_end - t_aut

    return CS, errFlg 



def df_small_autocorr4(stimval, nband, size_CS, twin, use_fourier=None):
    # Same as small_autocorr, but without the ntrials multiplication, which can be done outside.
    xb = 1
    nband = stimval.shape[0]
    CS = np.zeros((int(nband*(nband+1)/2),2*twin+1)) # zeros(size_CS)
    N = stimval.shape[1]
    S = nband
    time_domain_comp_time = 5e-12 * N * (2*twin +1) * S**2.9 # Time in seconds the time-domain autocorr would take
    fourier_domain_comp_time = 1.15e-8 * S**2 * N * np.log(N+1) # Time in seconds the fourier-domain autocorr would take
    if use_fourier is None:
        use_fourier = time_domain_comp_time > fourier_domain_comp_time
    if use_fourier:
        savedStimFft = np.fft.fft(stimval.T, 2**np.ceil(np.log2(2*N-1)).astype(int)).T

        for ib1 in range(nband):
            for ib2 in range(ib1, nband):
                c = np.real(np.fft.ifft(np.conj(savedStimFft[ib2,:])*(savedStimFft[ib1, :])))

                # NEW version of algorithm by using xcorr
                CS[xb, :] = np.concatenate((c[-twin:], c[:twin+1]))
                xb += 1
    else:
        st = stimval.T
        for tid in range(-twin, twin+1):
            onevect = slice(max(0,tid), min(N,N+tid))
            othervect = slice(onevect.start-tid, onevect.stop-tid)
            temp = stimval[:,onevect] @ st[othervect,:]
            # temp = np.outer(stimval[:,onevect],st[othervect,:])
            CS[:,twin + tid] = df_uppertrivect(temp)

    return CS


def df_uppertrivect(in_mat):
    """
    Returns the upper triangular part of the square matrix "in_mat" in a vector.
    """
    S = in_mat.shape[0]
    if S != in_mat.shape[1]:
        raise ValueError(f"Error in function df_uppertrivect: input has shape {in_mat.shape}, but it should be a square matrix.")
    
    # by rows:
    out = []
    for j in range(S):
        out += list(in_mat[j:S, j])
    return out





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