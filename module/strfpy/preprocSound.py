import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve

from .timeFreq import timefreq
import pandas as pd


def preprocess_sound_from_df(playback_and_response_df:pd.DataFrame, preprocess_type:str='ft', stim_params:dict=None,
                     output_dir:str=None, stim_output_pattern:str='preprocessed_stim_%s',
                     resp_output_pattern:str='preprocessed_resp_%d'):
    """Preprocess a set of stimuli and responses from the pipeline
    Args:
        playback_and_response_df (pandas.DataFrame): Dataframe from GenPlaybackSpikesDB containing stim and response pairs.
        preprocess_type (str): Type of time-frequency representation. Must be one of 'ft', 'wavelet', or 'lyons'.
        stim_params (dict): Parameters to assign to tfrep (optional). If not given, default values will be specified for type.
        output_dir (str): Directory to save preprocessed stimuli and responses.
        stim_output_pattern (str): Pattern for naming preprocessed stimuli. Must contain a single %s.
        resp_output_pattern (str): Pattern for naming preprocessed responses. Must contain a single %d.
    """
    if preprocess_type not in ['ft', 'wavelet', 'lyons']:
        raise ValueError('Unknown time-frequency representation type: %s' % preprocess_type)
    
    if stim_params is None:
        stim_params = {}

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(output_dir, exist_ok=True)

    stim_sample_rate = 1000.0
    resp_sample_rate = 1000.0

    # setup stim_params structure
    stim_params['fband'] = 125
    stim_params['nstd'] = 6
    stim_params['high_freq'] = 8000
    stim_params['low_freq'] = 250
    stim_params['log'] = 1
    stim_params['stim_rate'] = stim_sample_rate # this will be the STFT sample rate
    stim_params['output_pattern'] = stim_output_pattern
    stim_params['output_dir'] = output_dir

    DBNOISE = 80.0

    # TODO: First pass I will consider each interrupted trial its own stimulus
    #       In the future, I will want to concatenate all interrupted trials of the same stim 
    #       into one stimulus.
    uninterrupted_df = playback_and_response_df[playback_and_response_df['Peck'] == 0]
    interrupted_df = playback_and_response_df[playback_and_response_df['Peck'] != 0]
    
    datasets = []
    # first deal with uninterrupted trials
    for i, stim_df in uninterrupted_df.groupby('Rendition_Name'):
        ds = preprocess_stimuli_df(stim_df, stim_params, preprocess_type, resp_sample_rate)
        datasets.append(ds)

    # now deal with interrupted trials
    for i, stim_df in interrupted_df.iterrows():
        ds = preprocess_stimuli_df(stim_df, stim_params, preprocess_type, resp_sample_rate)
        datasets.append(ds)

    # Threshold the spectrogram to DBNOISE
    max_stim_amp = np.max([ds['stim']['maxStimAmp'] for ds in datasets])
    for k in range(len(datasets)):
        spec = datasets[k]['stim']['tfrep']['spec']
        spec = spec - max_stim_amp + DBNOISE
        spec[spec<0] = 0.0
        datasets[k]['stim']['tfrep']['spec'] = spec

    # check that all stims have the same number of channels
    if not np.all([ds['stim']['nStimChannels'] == datasets[0]['stim']['nStimChannels'] for ds in datasets]):
        raise ValueError('All stimuli must have the same number of channels')

    # set dataset wide values
    srData = {
        'stimSampleRate': stim_sample_rate,
        'respSampleRate': resp_sample_rate,
        'nStimChannels': datasets[0]['stim']['nStimChannels'],
        'datasets': datasets
    }

    # return srData
    # compute averages
    max_resp_len = np.max([len(ds['resp']['psth']) for ds in datasets])
    stim_avg, resp_avg, tv_resp_avg = compute_srdata_means(srData, max_resp_len)
    srData['stimAvg'] = stim_avg
    srData['respAvg'] = resp_avg
    srData['tvRespAvg'] = tv_resp_avg
    srData['type'] = preprocess_type

    return srData


def preprocess_stimuli_df(stim_df,stim_params,preprocess_type, resp_sample_rate=1000.0):
        ds = {}
        is_row = len(stim_df.shape) == 1
        # preprocess responses to a single stimulus
        if not is_row and len(stim_df.Rendition_Name.unique()) > 1:
            raise ValueError('More than one stimuli present')
        
        # first process the stimulus
        if is_row:
            stim_name = os.path.basename(stim_df.Rendition_Name).strip('.wav')
        else:
            stim_name = os.path.basename(stim_df.Rendition_Name.iloc[0]).strip('.wav')
        # preprocess stimulus
        stim_output_fname = os.path.join(stim_params['output_dir'], stim_params['output_pattern'] % (stim_name))

        if os.path.isfile(stim_output_fname):
            # use cached preprocessed stimulus
            #print(f'Using cached preprocessed stimulus from {stim_output_fname}')
            stim = np.load(stim_output_fname,  allow_pickle = True)['stim'].item()
            ds['stim'] = stim
        else:
            # we have not processed this stimulus yet
            # get the stim name
            if is_row:
                wav_file_name = stim_df.PlaybackFile
            else:
                wav_file_name = stim_df.PlaybackFile.iloc[0]

            # Compute a time-frequency representation of the stimulus
            tfrep = timefreq(wav_file_name, preprocess_type, stim_params)
            # store metadata in a dictionary
            stim = {
                'type': 'tfrep',
                'rawFile': wav_file_name,
                'tfrep': tfrep,
                'rawSampleRate': tfrep['params']['rawSampleRate'],
                'sampleRate': stim_params['stim_rate'],
                'stimLength': tfrep['spec'].shape[1] / stim_params['stim_rate'],
                'nStimChannels' : tfrep['f'].shape[0],
                'maxStimAmp': np.max(tfrep["spec"])
            }
            np.savez(stim_output_fname, stim=stim)
            ds['stim'] = stim

        # now trim the stim to be the length of the trial
        if is_row:
            if stim_df.Peck > 0:
                ds['stim']['stimLength'] = stim_df.Peck
                ds['stim']['tfrep']['spec'] = ds['stim']['tfrep']['spec'][:, :round(stim_df.Peck*stim_params['stim_rate'])]
        else:
            # TODO: If they are not the same length this will need to change
            if stim_df.Peck.iloc[0] > 0:
                ds['stim']['stimLength'] = stim_df.Peck.iloc[0]
                ds['stim']['tfrep']['spec'] = ds['stim']['tfrep']['spec'][:, :round(stim_df.Peck.iloc[0]*stim_params['stim_rate'])]

        # take all spike_times and convert to ms
        # only take times above 0
        if is_row:
            ms_times = (stim_df.Spike_Times - stim_df.Start) * 1000.0
            spike_times_ms = [ms_times[ms_times > 0]]
        else:
            spike_times_ms = [x[x>0] for x in (stim_df.Spike_Times - stim_df.Start).values * 1000.0]
        resp = preprocess_response(spike_times_ms, stim['stimLength'], resp_sample_rate)
        ds['resp'] = resp

        return ds

def preprocess_sound(raw_stim_files:list, raw_resp_files:list, preprocess_type:str='ft', stim_params:dict=None,
                     output_dir:str=None, stim_output_pattern:str='preprocessed_stim_%d',
                     resp_output_pattern:str='preprocessed_resp_%d'):
    """Preprocess a set of stimuli and responses.
    Args:
        raw_stim_files (list): List of .wav files containing the stimuli.
        raw_resp_files (list): List of .txt files containing the responses.
        preprocess_type (str): Type of time-frequency representation. Must be one of 'ft', 'wavelet', or 'lyons'.
        stim_params (dict): Parameters to assign to tfrep (optional). If not given, default values will be specified for type.
        output_dir (str): Directory to save preprocessed stimuli and responses.
        stim_output_pattern (str): Pattern for naming preprocessed stimuli. Must contain a single %d.
        resp_output_pattern (str): Pattern for naming preprocessed responses. Must contain a single %d.
    """
    if len(raw_stim_files) != len(raw_resp_files):
        raise ValueError('# of stim and response files must be the same!')
    
    if preprocess_type not in ['ft', 'wavelet', 'lyons']:
        raise ValueError('Unknown time-frequency representation type: %s' % preprocess_type)
    
    if stim_params is None:
        stim_params = {}
    
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(output_dir, exist_ok=True)
    
    
    max_stim_len = -1.0   # Stimulus length is seconds
    max_resp_len = -1     # Stimulus-response length is number of points
    n_stim_channels = -1
    stim_sample_rate = 1000.0
    resp_sample_rate = 1000.0
    max_stim_amp = 0.0
    DBNOISE = 80.0
    pair_count = len(raw_stim_files)
    srData = {}
    datasets = [None] * pair_count

    # preprocess each stimulus and response
    for k in range(pair_count):
        ds = {}

        # preprocess stimulus
        stim_output_fname = os.path.join(output_dir, stim_output_pattern % (k))

        if os.path.isfile(stim_output_fname):
            # use cached preprocessed stimulus
            #print(f'Using cached preprocessed stimulus from {stim_output_fname}')
            stim = np.load(stim_output_fname,  allow_pickle = True)['stim'].item()
            ds['stim'] = stim
        else:
            wav_file_name = raw_stim_files[k]
            stim_params['fband'] = 125
            stim_params['nstd'] = 6
            stim_params['high_freq'] = 8000
            stim_params['low_freq'] = 250
            stim_params['log'] = 1
            stim_params['stim_rate'] = stim_sample_rate
            tfrep = timefreq(wav_file_name, preprocess_type, stim_params)
            stim = {
                'type': 'tfrep',
                'rawFile': wav_file_name,
                'tfrep': tfrep,
                'rawSampleRate': tfrep['params']['rawSampleRate'],
                'sampleRate': stim_sample_rate,
                'stimLength': tfrep['spec'].shape[1] / stim_sample_rate,
                'nStimChannels' : tfrep['f'].shape[0],
                'maxStimAmp': np.max(tfrep["spec"])
            }
            np.savez(stim_output_fname, stim=stim)
            ds['stim'] = stim
            
        
        if (n_stim_channels == -1 ):
            n_stim_channels = stim['nStimChannels']
        else:
            if (n_stim_channels != stim['nStimChannels']):
                print('Error: number of spatial (frequency) channels does not match across stimuli')

        max_stim_amp = np.max((max_stim_amp, stim['maxStimAmp']))
        max_stim_len = np.max((max_stim_len, stim['stimLength']))

        # preprocess response
        resp_output_fname = os.path.join(output_dir, resp_output_pattern % (k))
        if os.path.isfile(resp_output_fname):
            # use cached preprocessed response
            #print(f'Using cached preprocessed response from {resp_output_fname}')
            resp = np.load(resp_output_fname, allow_pickle=True)['resp'].item()
            ds['resp'] = resp
        else:
            spike_trials = read_spikes_from_file(raw_resp_files[k])
            resp = preprocess_response(spike_trials, ds['stim']['stimLength'], resp_sample_rate)
            np.savez(resp_output_fname, resp=resp)
            ds['resp'] = resp

        max_resp_len = np.max((max_resp_len, len(resp['psth'])))

        datasets[k] = ds

    # Threshold spectrogram - this should probably be elsewhere or at least consistent with log
    for k in range(pair_count):
        spec = datasets[k]['stim']['tfrep']['spec']
        spec = spec - max_stim_amp + DBNOISE
        spec[spec<0] = 0.0
        datasets[k]['stim']['tfrep']['spec'] = spec

    
       
    # set dataset-wide values
    srData = {
        'stimSampleRate': stim_sample_rate,
        'respSampleRate': resp_sample_rate,
        'nStimChannels': n_stim_channels,
        'datasets': datasets
    }

    # return srData
    # compute averages
    stim_avg, resp_avg, tv_resp_avg = compute_srdata_means(srData, max_resp_len)
    srData['stimAvg'] = stim_avg
    srData['respAvg'] = resp_avg
    srData['tvRespAvg'] = tv_resp_avg
    srData['type'] = preprocess_type

    return srData



def read_spikes_from_file(file_name):
    spike_file = open(file_name, 'r')
    Trials = spike_file.readlines()
 
    spike_trials = []
    for trial in Trials:
        trial_data = np.fromstring(trial, dtype=float, sep=' ')
        spike_trials.append(trial_data)
    
    return spike_trials



def preprocess_response(spikeTrials, stimLength, sampleRate):
    spikeIndicies = []
    trial_psths = []
    for trials in spikeTrials:
        # turn spike times (ms) into indexes at response sample rate
        stimes = np.round(trials*1e-3 * sampleRate).astype(int) - 1
        # Choose within stimulus
        stimes = stimes[stimes >= 0]
        stimes = stimes[stimes < np.round(stimLength*sampleRate)]
        spikeIndicies.append(stimes)
        # also save individual psths per trial
        trial_psths.append(make_psth(stimes, np.round(stimLength*sampleRate), 1))

    psth = make_psth(spikeIndicies, np.round(stimLength*sampleRate), 1)
    resp = {
        'type': 'psth',
        'sampleRate': sampleRate,
        'rawSpikeTimes': spikeTrials,
        'rawSpikeIndicies': spikeIndicies,
        'psth': psth,
        'trial_psth': trial_psths,
        'nTrials': len(spikeTrials)
    }
    return resp

def make_psth(spikeTrialsInd:list, stimdur:int, binsize:int):
    """Make a PSTH from a list of spike times (in samples) and a stimulus duration (in samples)
    Args:
        spikeTrialsInd (list): list of spike times (in samples)
        stimdur (int): stimulus duration (in samples)
        binsize (int): bin size (in samples)
    """
    nbins = int(np.round(stimdur / binsize))
    psth = np.zeros(nbins)

    for trial in spikeTrialsInd:
        sindxs = np.round(trial / binsize).astype(int) 
        psth[sindxs] += 1

    return psth

def make_normed_psth(spikeTrialsInd:list, stimdur:int, binsize:int):
    """Make a normalized PSTH from a list of spike times (in samples) and a stimulus duration (in samples)
    Args:
        spikeTrialsInd (list): list of spike times (in samples)
        stimdur (int): stimulus duration (in samples)
        binsize (int): bin size (in samples)
    """

    psth = make_psth(spikeTrialsInd, stimdur, binsize)
    ntrials = len(spikeTrialsInd)
    psth /= ntrials
    return psth

def compute_srdata_means(srData, maxRespLen):
    # Computes the average stimulus for each spatial dimension, the average response (spikes/sample),
    # and the time-varying response in delete one format for JN statistics.

    pairCount = len(srData['datasets'])

    # compute stim and response averages
    stimSum = np.zeros((srData['nStimChannels'], ))
    stimCountSum = 0
    respCountSum = 0
    respSum = np.zeros((1, maxRespLen))
    meanSum = 0
    tvRespCount = np.zeros((pairCount, maxRespLen))

    # first compute all the sums
    for k in range(pairCount):
        # Stimulus mean for each spatial dimension (i.e. frequency band for spectro-temporal)
        ds = srData['datasets'][k]
        spec = ds['stim']['tfrep']['spec']
        stimSum += np.sum(spec, axis=1) # sum across time
        stimCountSum += spec.shape[1] # weight by number of timebins

        # Response averages
        # get unnormed psth
        psth = ds['resp']['psth']

        # Time varying average response
        respSum[0,:len(psth)] += psth
        # weight by the number of trials
        tvRespCount[k, :len(psth)] = ds['resp']['nTrials']

        # Overall response average
        meanSum = meanSum + psth.mean()
        respCountSum += ds['resp']['nTrials'] * len(psth)

    # construct the time-varying mean for the response. each row of the tv-mean is the average PSTH (across pairs)
    # computed with the PSTH of that row index left out
    tvRespCountSum = np.sum(tvRespCount, axis=0)
    tvRespAvg = np.zeros((pairCount, maxRespLen))
    smoothWindowTau = 41
    hwin = np.hanning(smoothWindowTau)
    hwin = hwin / np.sum(hwin)
    halfTau = smoothWindowTau // 2
    coff = smoothWindowTau % 2
    for k in range(pairCount):
        ds = srData['datasets'][k]
        psth = ds['resp']['psth']
        rlen = maxRespLen - len(psth)
        nresp = np.append(psth, np.zeros((1, rlen)))

        # subtract off this pair's PSTH, construct mean
        tvcnts = tvRespCountSum - tvRespCount[k, :]
        tvcnts[tvcnts < 1] = 1 # TODO: this is a hack to avoid divide by zero

        tvRespAvg[k, :] = (respSum - nresp) / tvcnts

        # smooth with hanning window
        pprod = convolve(tvRespAvg[k, :], hwin, mode='full')
        sindx = halfTau+coff
        eindx = round(len(pprod)-halfTau) + 1
        tvRespAvg[k, :] = pprod[sindx:eindx]

    stimAvg = stimSum / stimCountSum
    respAvg = meanSum / respCountSum

    return stimAvg, respAvg, tvRespAvg


def split_psth(spikeTrialsInd, stimLengthMs):

    halfSize = round(len(spikeTrialsInd)/2)
    spikeTrials1 = [None]*halfSize
    spikeTrials2 = [None]*halfSize


    for j, trial in enumerate(spikeTrialsInd):
        indx = int((j + 1) // 2)
        if j%2 == 0:
            spikeTrials1[indx-1] = trial
        else:
            spikeTrials2[indx-1] = trial
    # if odd, truncate spikeTrials2
    if len(spikeTrialsInd)%2 == 1:
        spikeTrials2 = spikeTrials2[:-1]
            
    psth = make_normed_psth(spikeTrialsInd, stimLengthMs, 1)
    psth1 = make_normed_psth(spikeTrials1, stimLengthMs, 1)
    psth2 = make_normed_psth(spikeTrials2, stimLengthMs, 1)

    psthdata = {'psth': psth, 'psth_half1': psth1, 'psth_half2': psth2}
    return psthdata