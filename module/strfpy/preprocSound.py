import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve, windows

#sys.path.append("/Users/frederictheunissen/Code/crcns-kailin/module")
from module.strfpy.timeFreq import timefreq, timefreq_raw
import pandas as pd
import pynwb as nwb

def weighted_mean(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def weighted_cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

def weighted_corr(x, y, w):
    """Weighted Correlation"""
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))


def preprocess_sound_raw_nospike(stim_lookup, all_trials, preprocess_type='ft', stim_params={}):
    # params
    DBNOISE = 80.0  
    stim_sample_rate = 1000.0
    
    # now we have all the spike times aligned to the stimulus onset for all stimuli
    # now we group them by stimulus and preprocess them
    srData = {}
    datasets = []
    max_stim_amp = 0.0
    n_stim_channels = -1
    for stim_name, stim_df in all_trials.groupby('stimuli_name'):
        ds = {}
        # preprocess the stimuli by loading the wav and generating the tfrep
        wav_file_name = stim_name #raw_stim_files[k]
        stim_fs, stim_data = stim_lookup(stim_name)
        stim_params['fband'] = 120
        stim_params['nstd'] = 6
        stim_params['high_freq'] = 8000
        stim_params['low_freq'] = 250
        stim_params['log'] = 1
        stim_params['stim_rate'] = stim_sample_rate
        tfrep = timefreq_raw(stim_data,stim_fs, preprocess_type, stim_params)
        stim = {
            'type': 'tfrep',
            'rawFile': stim_name,
            'tfrep': tfrep,
            'rawSampleRate': tfrep['params']['rawSampleRate'],
            'sampleRate': stim_sample_rate,
            'stimLength': tfrep['spec'].shape[1] / stim_sample_rate,
            'nStimChannels' : tfrep['f'].shape[0],
            'maxStimAmp': np.max(tfrep["spec"])
        }
        ds['stim'] = stim

        if (n_stim_channels == -1 ):
            n_stim_channels = stim['nStimChannels']
        else:
            if (n_stim_channels != stim['nStimChannels']):
                print('Error: number of spatial (frequency) channels does not match across stimuli')

        max_stim_amp = np.max((max_stim_amp, stim['maxStimAmp']))

        datasets.append(ds)
    # end loop over stimuli

    # Threshold spectrogram - this should probably be elsewhere or at least consistent with log
    for k in range(len(datasets)):
        spec = datasets[k]['stim']['tfrep']['spec']
        spec = spec - max_stim_amp + DBNOISE
        spec[spec<0] = 0.0
        datasets[k]['stim']['tfrep']['spec'] = spec

    
    
    # set dataset-wide values
    srData = {
        'stimSampleRate': stim_sample_rate,
        'nStimChannels': n_stim_channels,
        'datasets': datasets
    }

    # # return srData
    # # compute averages
    # stim_avg, resp_avg, tv_resp_avg = compute_srdata_means(srData, max_resp_len)
    # srData['stimAvg'] = stim_avg
    srData['type'] = preprocess_type

    return srData

def preprocess_sound_raw(unit_spike_times, stim_lookup, all_trials, preprocess_type='ft', stim_params={}, resp_type='spikes'):
    # params
    DBNOISE = 80.0  
    stim_sample_rate = 1000.0
    resp_sample_rate = 1000.0
    
    # now we have all the spike times aligned to the stimulus onset for all stimuli
    # now we group them by stimulus and preprocess them
    srData = {}
    datasets = []
    max_stim_amp = 0.0
    max_resp_len = -1     # Stimulus-response length is number of points
    n_stim_channels = -1
    for stim_name, stim_df in all_trials.groupby('stimuli_name'):
        ds = {}
        # preprocess the stimuli by loading the wav and generating the tfrep
        wav_file_name = stim_name #raw_stim_files[k]
        stim_fs, stim_data = stim_lookup(stim_name)
        stim_params['fband'] = 120
        stim_params['nstd'] = 6
        stim_params['high_freq'] = 8000
        stim_params['low_freq'] = 250
        stim_params['log'] = 1
        stim_params['stim_rate'] = stim_sample_rate
        tfrep = timefreq_raw(stim_data,stim_fs, preprocess_type, stim_params)
        stim = {
            'type': 'tfrep',
            'rawFile': stim_name,
            'tfrep': tfrep,
            'rawSampleRate': tfrep['params']['rawSampleRate'],
            'sampleRate': stim_sample_rate,
            'stimLength': tfrep['spec'].shape[1] / stim_sample_rate,
            'nStimChannels' : tfrep['f'].shape[0],
            'maxStimAmp': np.max(tfrep["spec"])
        }
        ds['stim'] = stim

        if (n_stim_channels == -1 ):
            n_stim_channels = stim['nStimChannels']
        else:
            if (n_stim_channels != stim['nStimChannels']):
                print('Error: number of spatial (frequency) channels does not match across stimuli')

        max_stim_amp = np.max((max_stim_amp, stim['maxStimAmp']))

        # preprocess the response by loading the individual responses and calculating the psth
        if resp_type == 'spikes':
            # lets get the spike times for this stimulus
            # get trial spike times
            trial_starts = stim_df.start_time.values
            trial_stops = stim_df.stop_time.values
            spike_idx_start = np.searchsorted(unit_spike_times, trial_starts)
            spike_idx_stop = np.searchsorted(unit_spike_times, trial_stops)
            spike_times = [unit_spike_times[spike_idx_start[i]:spike_idx_stop[i]] - trial_starts[i] for i in range(len(trial_starts))]
            stim_len_samples = int(np.round(stim['stimLength']*1000))
            bin_size = 1
            nbins = int(stim_len_samples // bin_size)
            # psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
            # psth = np.zeros(nbins)
            # psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]

            weights = np.zeros(nbins)
            trial_durations_samples = ((stim_df.stop_time.values - stim_df.start_time.values) * stim_sample_rate)
            weights = (trial_durations_samples >= np.arange(nbins)[:, None]).sum(axis=1)
            psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
            psth = np.zeros(nbins)
            psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
            psth[weights > 0] /= weights[weights > 0]
            
            resp = {
                'type': 'psth',
                'sampleRate': resp_sample_rate,
                'rawSpikeTimes': spike_times,
                'rawSpikeIndicies': [(st * 1000 / bin_size).astype(int) for st in spike_times],
                'trialDurations': trial_durations_samples,
                'psth': psth,
                'weights': weights
            }
            ds['resp'] = resp
        elif resp_type == 'lfp':
            # in this case unit_spike_times is None
            # LFP is in the trials_df
            # we take the average LFP for this stim
            bin_size = 1
            stim_len_samples = int(np.round(stim['stimLength']*1000))
            nbins = int(stim_len_samples // bin_size)
            weights = np.zeros(nbins)
            trial_durations_samples = ((stim_df.stop_time.values - stim_df.start_time.values) * 1000).astype(int)
            weights = (trial_durations_samples >= np.arange(nbins)[:, None]).sum(axis=1)
            lfps = []
            for ix, trial in stim_df.iterrows():
                # we want to have nbins samples
                lfp_vec = np.zeros(nbins)
                lfp_ind = 2
                len_trial_data = trial.timeseries[lfp_ind].data.shape[0]
                if len_trial_data < nbins:
                    lfp_vec[:len_trial_data] = np.mean(trial.timeseries[lfp_ind].data,axis=1)
                else:
                    lfp_vec = np.mean(trial.timeseries[lfp_ind].data[:nbins],axis=1)
                lfps.append(lfp_vec)
            lfps = np.array(lfps)
            lfp_sums = lfps.sum(axis=0)
            lfp_avg_w = np.zeros(nbins)
            lfp_avg_w[weights > 0] = lfp_sums[weights>0] / weights[weights > 0]
            # nwo get variance for each bin
            # stdev calc is sqrt((sum(x^2) - sum(x)^2/n) / (n-1))
            #lfp_stdev = np.zeros(nbins)
            #lfp_stdev[weights > 0] = np.sqrt((np.sum(lfps**2,axis=0)[weights>0] - lfp_sums[weights>0]**2 / weights[weights > 0]) / (weights[weights > 0] - 1))
            resp = {
                'type': 'psth',
                'sampleRate': resp_sample_rate,
                'rawSpikeTimes': None,
                'rawSpikeIndicies': None,
                'trialDurations': trial_durations_samples,
                'psth': lfp_avg_w,
                'weights': weights
            }
            ds['resp'] = resp

        max_resp_len = np.max((max_resp_len, len(resp['psth'])))
        datasets.append(ds)
    # end loop over stimuli

    # Threshold spectrogram - this should probably be elsewhere or at least consistent with log
    for k in range(len(datasets)):
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

def balance_trials(trials: pd.DataFrame, 
                  grouping: list = ['bird_name', 'stimuli_name', 'stim_class'],
                  abs_min_trials: int = 5,
                  random_state: int = None) -> pd.DataFrame:
    """
    Balance trials by randomly sampling the same number of trials for each group.
    Preserves original DataFrame indices in the sampled data.

    Parameters:
    -----------
    trials : pd.DataFrame
        DataFrame containing trial information with columns for grouping variables.
    grouping : list of str
        List of column names to group trials by.
        Default is ['bird_name', 'stimuli_name', 'class'].
    random_state : int, optional
        Random seed for reproducibility. Default is None.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the balanced trials with original indices preserved.
    """
    # Validate grouping columns exist
    missing_cols = [col for col in grouping if col not in trials.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Find minimum group size
    n_trials = trials.groupby(grouping).size().min()
    # ignore groups less than abs_min_trials and dont add them to balanced trials
    if n_trials < abs_min_trials:
        n_trials = abs_min_trials
        # all_trials_filt = all_trials_filt.groupby(grouping).filter(lambda x: len(x) > 6)
        trials = trials.groupby(grouping).filter(lambda x: len(x) > 6)

    # Sample rows while preserving indices
    balanced_trials = trials.groupby(grouping).apply(
        lambda x: x.sample(n=n_trials, replace=False)
    )
    
    # Reset only the grouping index, keep the original row indices
    balanced_trials = balanced_trials.reset_index(level=list(range(len(grouping))), drop=True)
    
    return balanced_trials
def get_mic_data(nwb, trial, ch=1):
    rate = nwb.acquisition['audio'].rate
    mic_data = nwb.acquisition['audio'].data
    start_id = int(trial.start_time * rate)
    end_id = int(trial.stop_time * rate)
    mic_trial = mic_data[start_id:end_id]
    return mic_trial[:,1]

def generate_srData_nwb_single_trials(nwb, intervals_name, unit_id):
    # params
    DBNOISE = 80.0  
    stim_sample_rate = 1000.0
    resp_sample_rate = 1000.0

    # get intervals and spike times from database
    all_trials = nwb.intervals[intervals_name].to_dataframe()
    unit_spike_times = nwb.units[unit_id].spike_times.values[0]

    # get unit valid intervals
    all_valid_intervals = nwb.intervals['unit_intervals'].to_dataframe()
    unit_valid_intervals = all_valid_intervals[all_valid_intervals['unit_id'] == unit_id]

    # remove trials that are not in valid intervals
    valid_trials = all_trials.apply(lambda x: any((unit_valid_intervals.start_time < x.start_time) & (unit_valid_intervals.stop_time > x.stop_time)), axis=1)
    all_trials = all_trials[valid_trials]

    # lets balance the trials by stimuli name
    all_trials = balance_trials(all_trials, ['stimuli_name'])
    

    # lets get the precomputed spectrograms
    spectrograms = nwb.processing['stimuli_spectrograms']


    # now we have all the spike times aligned to the stimulus onset for all stimuli
    # now we group them by stimulus and preprocess them
    srData = {}
    datasets = []
    max_stim_amp = 0.0
    max_resp_len = -1     # Stimulus-response length is number of points
    n_stim_channels = -1
    for ix, row in all_trials.iterrows():
        stim_name = row['stimuli_name']
        ds = {}
        wav_file_name = stim_name #raw_stim_files[k]
        audio_rate = nwb.acquisition['audio'].rate
        stim_data = get_mic_data(nwb, row)
        stim_fs = audio_rate
        stim_params=dict()
        stim_params['fband'] = 120
        stim_params['nstd'] = 6
        stim_params['high_freq'] = 8000
        stim_params['low_freq'] = 250
        stim_params['log'] = 1
        stim_params['stim_rate'] = stim_sample_rate
        tfrep = timefreq_raw(stim_data,stim_fs, 'ft', stim_params)
        stim = {
            'type': 'tfrep',
            'rawFile': stim_name,
            'tfrep': tfrep,
            'rawSampleRate': tfrep['params']['rawSampleRate'],
            'sampleRate': stim_sample_rate,
            'stimLength': tfrep['spec'].shape[1] / stim_sample_rate,
            'nStimChannels' : tfrep['f'].shape[0],
            'maxStimAmp': np.max(tfrep["spec"])
        }
        ds['stim'] = stim
        
        if (n_stim_channels == -1 ):
            n_stim_channels = stim['nStimChannels']
        else:
            if (n_stim_channels != stim['nStimChannels']):
                print('Error: number of spatial (frequency) channels does not match across stimuli')

        max_stim_amp = np.max((max_stim_amp, stim['maxStimAmp']))
        # lets get the spike times for this stimulus
        # get trial spike times
        trial_start = row.start_time
        trial_stop = row.stop_time
        spike_idx_start = np.searchsorted(unit_spike_times, trial_start)
        spike_idx_stop = np.searchsorted(unit_spike_times, trial_stop)
        spike_times = unit_spike_times[spike_idx_start:spike_idx_stop] - trial_start
        stim_len_samples = int(np.round(stim['stimLength']*1000))
        bin_size = int(np.round( 1000 / resp_sample_rate )) # ms
        nbins = int(stim_len_samples // bin_size)

        weights = np.zeros(nbins)
        trial_duration_samples = ((trial_stop - trial_start) * stim_sample_rate)
        weights = (trial_duration_samples >= np.arange(nbins)[:, None]).sum(axis=1)
        psth_idx, counts = np.unique(np.round(spike_times * 1000 / bin_size).astype(int), return_counts=True)
        psth = np.zeros(nbins)
        psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
        psth[weights > 0] /= weights[weights > 0]
        psth = psth * 1000 / bin_size
        
        resp = {
            'type': 'psth',
            'sampleRate': resp_sample_rate,
            'rawSpikeTimes': spike_times,
            'rawSpikeIndicies': [(st * 1000 / bin_size).astype(int) for st in spike_times],
            'trialDurations': [trial_duration_samples],
            'psth': psth,
            'weights': weights
        }
        ds['resp'] = resp

        max_resp_len = np.max((max_resp_len, len(resp['psth'])))
        datasets.append(ds)
    # end loop over trials

    # Threshold spectrogram - this should probably be elsewhere or at least consistent with log
    for k in range(len(datasets)):
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
    srData['type'] = 'ft'

    return srData

def generate_srData_nwb(nwb, intervals_name, unit_id):
    # params
    DBNOISE = 80.0  
    stim_sample_rate = 1000.0
    resp_sample_rate = 1000.0

    # get intervals and spike times from database
    all_trials = nwb.intervals[intervals_name].to_dataframe()
    unit_spike_times = nwb.units[unit_id].spike_times.values[0]

    # get unit valid intervals
    all_valid_intervals = nwb.intervals['unit_intervals'].to_dataframe()
    unit_valid_intervals = all_valid_intervals[all_valid_intervals['unit_id'] == unit_id]

    # remove trials that are not in valid intervals
    valid_trials = all_trials.apply(lambda x: any((unit_valid_intervals.start_time < x.start_time) & (unit_valid_intervals.stop_time > x.stop_time)), axis=1)
    all_trials = all_trials[valid_trials]

    # lets balance the trials by stimuli name
    all_trials = balance_trials(all_trials, ['stimuli_name'])
    

    # lets get the precomputed spectrograms
    spectrograms = nwb.processing['stimuli_spectrograms']


    # now we have all the spike times aligned to the stimulus onset for all stimuli
    # now we group them by stimulus and preprocess them
    srData = {}
    datasets = []
    max_stim_amp = 0.0
    max_resp_len = -1     # Stimulus-response length is number of points
    n_stim_channels = -1
    for stim_name, stim_df in all_trials.groupby('stimuli_name'):
        ds = {}
        # preprocess the stimuli by loading the wav and generating the tfrep
        wav_file_name = stim_name #raw_stim_files[k]
        stim_spec = spectrograms[stim_name].data[:].T # nfreq x ntime
        stim_t = spectrograms[stim_name].timestamps[:] # ntime
        stim_fs = 1.0 / (stim_t[1] - stim_t[0]) # sample rate of the spectrogram
        # lets check that stim_fs is close to stim_sample_rate
        if np.abs(stim_fs - stim_sample_rate) > 1e-3:
            print("Warning: stim_fs is not close to stim_sample_rate")
        stim = {
            'type': 'tfrep',
            'rawFile': stim_name,
            'tfrep': dict({'spec':stim_spec,
                            'f':np.arange(stim_spec.shape[0])*(8000-250) + 250,}),
            'sampleRate': stim_sample_rate,
            'stimLength': stim_t[-1],
            'nStimChannels' : stim_spec.shape[0],
            'maxStimAmp': np.max(stim_spec)
        }
        ds['stim'] = stim

        if (n_stim_channels == -1 ):
            n_stim_channels = stim['nStimChannels']
        else:
            if (n_stim_channels != stim['nStimChannels']):
                print('Error: number of spatial (frequency) channels does not match across stimuli')

        max_stim_amp = np.max((max_stim_amp, stim['maxStimAmp']))

        # preprocess the response by loading the individual responses and calculating the psth
        
        # lets get the spike times for this stimulus
        # get trial spike times
        trial_starts = stim_df.start_time.values
        trial_stops = stim_df.stop_time.values
        spike_idx_start = np.searchsorted(unit_spike_times, trial_starts)
        spike_idx_stop = np.searchsorted(unit_spike_times, trial_stops)
        spike_times = [unit_spike_times[spike_idx_start[i]:spike_idx_stop[i]] - trial_starts[i] for i in range(len(trial_starts))]
        stim_len_samples = int(np.round(stim['stimLength']*1000))
        bin_size = int(np.round( 1000 / resp_sample_rate )) # ms
        nbins = int(stim_len_samples // bin_size)
        # psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
        # psth = np.zeros(nbins)
        # psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]

        weights = np.zeros(nbins)
        trial_durations_samples = ((stim_df.stop_time.values - stim_df.start_time.values) * stim_sample_rate)
        weights = (trial_durations_samples >= np.arange(nbins)[:, None]).sum(axis=1)
        psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
        psth = np.zeros(nbins)
        psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
        psth[weights > 0] /= weights[weights > 0]
        psth = psth * 1000 / bin_size
        
        resp = {
            'type': 'psth',
            'sampleRate': resp_sample_rate,
            'rawSpikeTimes': spike_times,
            'rawSpikeIndicies': [(st * 1000 / bin_size).astype(int) for st in spike_times],
            'trialDurations': trial_durations_samples,
            'psth': psth,
            'weights': weights
        }
        ds['resp'] = resp

        max_resp_len = np.max((max_resp_len, len(resp['psth'])))
        datasets.append(ds)
    # end loop over stimuli

    # Threshold spectrogram - this should probably be elsewhere or at least consistent with log
    for k in range(len(datasets)):
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
    srData['type'] = 'ft'

    return srData


def calc_psth(spike_times,  psth_dur_s, t_start_s=0, bin_size=1, durations=None, bSmooth=False):
    """
    Calculate Peri-Stimulus Time Histogram (PSTH).

    Parameters:
    spike_times (array-like): Array of spike times in seconds.
    psth_dur_s (float): Duration of the psth to be calculated in seconds.
    t_start_s (float, optional): Start time of the PSTH in seconds. Default is 0.
    bin_size (int, optional): Size of the bins in milliseconds. Default is 1 ms.
    durations (array-like): Array of trial durations in seconds for weighting bins.
                             if None, no weighting.
    bSmooth (bool, optional): If True, the PSTH will be smoothed using a Hanning window. Default is False.

    Returns:
    tuple: A tuple containing:
        - bins (numpy.ndarray): Array of bin centers in seconds.
        - psth (numpy.ndarray): Array of PSTH values (spikes per second).
    """
    # spike_times can be a list of arrays or an array
    if isinstance(spike_times, list):
        n_trials = len(spike_times)
        spike_times = np.concatenate(spike_times)
    else:
        n_trials = 1
    assert np.all(np.round(1000*spike_times) >= np.round(1000*t_start_s)), "Spike times must be larger than the start_time."
    spike_times += t_start_s # offset by t_start_s to align with the start time

    nbins = int(np.round(psth_dur_s*1000) // bin_size)
    #print(min(spike_times), max(spike_times), psth_dur_s)
    psth_idx, counts = np.unique(
        np.round(spike_times * 1000 / bin_size).astype(int), return_counts=True)
    # counts = counts[psth_idx >= 0]
    # psth_idx = psth_idx[psth_idx >= 0]
    
    psth = np.zeros(nbins)
    psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
    if durations is None:
        psth /= n_trials
    else:
        if n_trials != len(durations):
            raise ValueError(
                "The number of durations must match the number of trials.")
        weights = np.zeros(nbins)
        trial_durations_ms = (durations * 1000).astype(int)
        weights = (trial_durations_ms >= np.arange(
            nbins)[:, None]*bin_size).sum(axis=1)
        psth[weights > 0] /= weights[weights > 0]
    if bSmooth:
        # The 21 ms (number of points) hanning window used to smooth the PSTH
        # I think we should shoot for a 20 ms kernel no matter what the bin size is
        kernel_size = 40 // bin_size + 1
        wHann = windows.hann(kernel_size, sym=True)
        wHann = wHann/sum(wHann)
        psth = np.convolve(psth, wHann, mode='same')
    return np.arange(nbins)*bin_size/1000 + t_start_s, psth * 1000 / bin_size

def preprocess_sound_nwb(nwb_file, intervals_name, unit_id, preprocess_type='ft', stim_params={}, stim_loader=None, pb_fix=None, ignore_intervals=False):
    with nwb.NWBHDF5IO(nwb_file, 'r') as io:
        # params
        DBNOISE = 80.0  
        stim_sample_rate = 1000.0
        resp_sample_rate = 1000.0

        nwbfile = io.read()
        
        # get intervals and spike times from database
        all_trials = nwbfile.intervals[intervals_name].to_dataframe()
        unit_spike_times = nwbfile.units[unit_id].spike_times.values[0]
        print(nwbfile.units[unit_id])
        # get unit valid intervals
        all_valid_intervals = nwbfile.intervals['unit_intervals'].to_dataframe()
        unit_valid_intervals = all_valid_intervals[all_valid_intervals['unit_id'] == unit_id]

        # remove trials that are not in valid intervals
        valid_trials = all_trials.apply(lambda x: any((unit_valid_intervals.start_time < x.start_time) & (unit_valid_intervals.stop_time > x.stop_time)), axis=1)
        all_trials = all_trials[valid_trials]

        # # if 'response' in all_trials.columns:
        # #     all_trials = all_trials[all_trials['response'] == False]

        # now we have all the spike times aligned to the stimulus onset for all stimuli
        # now we group them by stimulus and preprocess them
        srData = {}
        datasets = []
        max_stim_amp = 0.0
        max_resp_len = -1     # Stimulus-response length is number of points
        n_stim_channels = -1
        for stim_name, stim_df in all_trials.groupby('stimuli_name'):
            ds = {}
            # preprocess the stimuli by loading the wav and generating the tfrep
            wav_file_name = stim_name #raw_stim_files[k]
            stim_data = nwbfile.stimulus[stim_name].data[:]
            stim_fs = nwbfile.stimulus[stim_name].rate
            stim_params['fband'] = 120
            stim_params['nstd'] = 6
            stim_params['high_freq'] = 8000
            stim_params['low_freq'] = 250
            stim_params['log'] = 1
            stim_params['stim_rate'] = stim_sample_rate
            tfrep = timefreq_raw(stim_data,stim_fs, preprocess_type, stim_params)
            stim = {
                'type': 'tfrep',
                'rawFile': stim_name,
                'tfrep': tfrep,
                'rawSampleRate': tfrep['params']['rawSampleRate'],
                'sampleRate': stim_sample_rate,
                'stimLength': tfrep['spec'].shape[1] / stim_sample_rate,
                'nStimChannels' : tfrep['f'].shape[0],
                'maxStimAmp': np.max(tfrep["spec"])
            }
            ds['stim'] = stim

            if (n_stim_channels == -1 ):
                n_stim_channels = stim['nStimChannels']
            else:
                if (n_stim_channels != stim['nStimChannels']):
                    print('Error: number of spatial (frequency) channels does not match across stimuli')

            max_stim_amp = np.max((max_stim_amp, stim['maxStimAmp']))

            # preprocess the response by loading the individual responses and calculating the psth
            
            # lets get the spike times for this stimulus
            # get trial spike times
            
            trial_starts = stim_df.start_time.values
            trial_stops = stim_df.stop_time.values

            spike_idx_start = np.searchsorted(unit_spike_times, trial_starts)
            spike_idx_stop = np.searchsorted(unit_spike_times, trial_stops)
            spike_times = [unit_spike_times[spike_idx_start[i]:spike_idx_stop[i]] - trial_starts[i] for i in range(len(trial_starts))]
            stim_len_samples = int(np.round(stim['stimLength']*1000))  # Stimulus length in ms
            bin_size = 1000.0/resp_sample_rate
            nbins = int(stim_len_samples // bin_size)
            # psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
            # psth = np.zeros(nbins)
            # psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]

            weights = np.zeros(nbins)
            trial_durations_samples = ((stim_df.stop_time.values - stim_df.start_time.values) * stim_sample_rate)
            weights = (trial_durations_samples >= np.arange(nbins)[:, None]).sum(axis=1)
            psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
            psth = np.zeros(nbins)
            psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
            psth[weights > 0] /= weights[weights > 0]
            psth = psth * resp_sample_rate / bin_size
            
            resp = {
                'type': 'psth',
                'sampleRate': resp_sample_rate,
                'rawSpikeTimes': spike_times,
                'rawSpikeIndicies': [(st * 1000 / bin_size).astype(int) for st in spike_times],
                'trialDurations': trial_durations_samples,
                'psth': psth,
                'weights': weights
            }
            ds['resp'] = resp

            max_resp_len = np.max((max_resp_len, len(resp['psth'])))
            datasets.append(ds)
        # end loop over stimuli

        # Threshold spectrogram - this should probably be elsewhere or at least consistent with log
        for k in range(len(datasets)):
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

def preprocess_sound_nwb_singletrial(nwb_file, intervals_name, unit_id, preprocess_type='ft', stim_params={}, stim_loader=None, pb_fix=None, ignore_intervals=False):
    # TODO this is WIP
    with nwb.NWBHDF5IO(nwb_file, 'r') as io:
        # params
        DBNOISE = 80.0  
        stim_sample_rate = 1000.0
        resp_sample_rate = 1000.0

        nwbfile = io.read()
        
        # get intervals and spike times from database
        all_trials = nwbfile.intervals[intervals_name].to_dataframe()
        unit_spike_times = nwbfile.units[unit_id].spike_times.values[0]
        print(nwbfile.units[unit_id])
        # get unit valid intervals
        all_valid_intervals = nwbfile.intervals['unit_intervals'].to_dataframe()
        unit_valid_intervals = all_valid_intervals[all_valid_intervals['unit_id'] == unit_id]

        # remove trials that are not in valid intervals
        valid_trials = all_trials.apply(lambda x: any((unit_valid_intervals.start_time < x.start_time) & (unit_valid_intervals.stop_time > x.stop_time)), axis=1)
        all_trials = all_trials[valid_trials]

        # # if 'response' in all_trials.columns:
        # #     all_trials = all_trials[all_trials['response'] == False]

        # now we have all the spike times aligned to the stimulus onset for all stimuli
        # now we group them by stimulus and preprocess them
        srData = {}
        datasets = []
        max_stim_amp = 0.0
        max_resp_len = -1     # Stimulus-response length is number of points
        n_stim_channels = -1
        for stim_name, stim_df in all_trials.groupby('stimuli_name'):
            ds = {}
            # preprocess the stimuli by loading the wav and generating the tfrep
            wav_file_name = stim_name #raw_stim_files[k]
            stim_data = nwbfile.stimulus[stim_name].data[:]
            stim_fs = nwbfile.stimulus[stim_name].rate
            stim_params['fband'] = 120
            stim_params['nstd'] = 6
            stim_params['high_freq'] = 8000
            stim_params['low_freq'] = 250
            stim_params['log'] = 1
            stim_params['stim_rate'] = stim_sample_rate
            tfrep = timefreq_raw(stim_data,stim_fs, preprocess_type, stim_params)
            stim = {
                'type': 'tfrep',
                'rawFile': stim_name,
                'tfrep': tfrep,
                'rawSampleRate': tfrep['params']['rawSampleRate'],
                'sampleRate': stim_sample_rate,
                'stimLength': tfrep['spec'].shape[1] / stim_sample_rate,
                'nStimChannels' : tfrep['f'].shape[0],
                'maxStimAmp': np.max(tfrep["spec"])
            }
            ds['stim'] = stim

            if (n_stim_channels == -1 ):
                n_stim_channels = stim['nStimChannels']
            else:
                if (n_stim_channels != stim['nStimChannels']):
                    print('Error: number of spatial (frequency) channels does not match across stimuli')

            max_stim_amp = np.max((max_stim_amp, stim['maxStimAmp']))

            # preprocess the response by loading the individual responses and calculating the psth
            
            # lets get the spike times for this stimulus
            # get trial spike times
            
            trial_starts = stim_df.start_time.values
            trial_stops = stim_df.stop_time.values

            spike_idx_start = np.searchsorted(unit_spike_times, trial_starts)
            spike_idx_stop = np.searchsorted(unit_spike_times, trial_stops)
            spike_times = [unit_spike_times[spike_idx_start[i]:spike_idx_stop[i]] - trial_starts[i] for i in range(len(trial_starts))]
            stim_len_samples = int(np.round(stim['stimLength']*1000))  # Stimulus length in ms
            bin_size = 1000.0/resp_sample_rate
            nbins = int(stim_len_samples // bin_size)
            # psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
            # psth = np.zeros(nbins)
            # psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]

            weights = np.zeros(nbins)
            trial_durations_samples = ((stim_df.stop_time.values - stim_df.start_time.values) * stim_sample_rate)
            weights = (trial_durations_samples >= np.arange(nbins)[:, None]).sum(axis=1)
            psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
            psth = np.zeros(nbins)
            psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
            psth[weights > 0] /= weights[weights > 0]
            psth = psth * resp_sample_rate / bin_size
            
            resp = {
                'type': 'psth',
                'sampleRate': resp_sample_rate,
                'rawSpikeTimes': spike_times,
                'rawSpikeIndicies': [(st * 1000 / bin_size).astype(int) for st in spike_times],
                'trialDurations': trial_durations_samples,
                'psth': psth,
                'weights': weights
            }
            ds['resp'] = resp

            max_resp_len = np.max((max_resp_len, len(resp['psth'])))
            datasets.append(ds)
        # end loop over stimuli

        # Threshold spectrogram - this should probably be elsewhere or at least consistent with log
        for k in range(len(datasets)):
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


def calculate_EV(srData, nPoints=200, mult_values=False):
    # iterate through each pair
    pairCount = len(srData['datasets'])
    all_psth_half1 = []
    all_psth_half2 = []
    all_weights = []
    for iSet in range(pairCount):
        pair = srData['datasets'][iSet]
        spike_inds = pair['resp']['rawSpikeIndicies']
        nT = pair['resp']['psth'].size
        trial_durations = pair['resp']['trialDurations']
        weights = pair['resp']['weights']
        # lets split the trials into two halves
        stim_df1 = spike_inds[:len(spike_inds):2]
        durations_1 = trial_durations[0:len(trial_durations):2]
        stim_df2 = spike_inds[1:len(spike_inds):2]
        durations_2 = trial_durations[1:len(trial_durations):2]
        # get the psth for each half
        psth_idx_1, counts_1 = np.unique(np.concatenate(stim_df1), return_counts=True)
        psth_idx_2, counts_2 = np.unique(np.concatenate(stim_df2), return_counts=True)
        psth1 = np.zeros(nT)
        psth2 = np.zeros(nT)
        psth1[psth_idx_1[psth_idx_1<nT]] = counts_1[psth_idx_1<nT]
        psth2[psth_idx_2[psth_idx_2<nT]] = counts_2[psth_idx_2<nT]
        # get the weights for each half
        weights1 = np.zeros(nT)
        weights2 = np.zeros(nT)
        weights1 = (durations_1 >= np.arange(nT)[:,None]).sum(axis=1)
        weights2 = (durations_2 >= np.arange(nT)[:,None]).sum(axis=1)
        psth1[weights1 > 0] /= weights1[weights1 > 0]
        psth2[weights2 > 0] /= weights2[weights2 > 0]
        # append the psth and weights
        all_psth_half1.append(psth1)
        all_psth_half2.append(psth2)
        assert(np.all(weights == weights1 + weights2))
        all_weights.append(weights)
    all_psth_half1 = np.concatenate(all_psth_half1)
    all_psth_half2 = np.concatenate(all_psth_half2)
    all_weights = np.concatenate(all_weights)
    wHann = windows.hann(21, sym=True)   # The 21 ms (number of points) hanning window used to smooth the PSTH
    wHann = wHann/sum(wHann)
    psth_half1 = np.convolve(all_psth_half1, wHann, mode='same')
    psth_half2 = np.convolve(all_psth_half2, wHann, mode='same')
    r12 = weighted_corr(psth_half1, psth_half2, all_weights)
    SNRHalf = 1/(r12*r12)-1
    SNRAll = SNRHalf*2
    R2Ceil = 1/(1+SNRAll)   # This is the ceiling value of R2 

    print('R2Ceil = %.3f'%R2Ceil)
    return R2Ceil

def psth_nwb(stim_df, unit_spike_times, stim_dur):
    trial_starts = stim_df.start_time.values
    trial_stops = stim_df.stop_time.values
    spike_idx_start = np.searchsorted(unit_spike_times, trial_starts)
    spike_idx_stop = np.searchsorted(unit_spike_times, trial_stops)
    spike_times = [unit_spike_times[spike_idx_start[i]:spike_idx_stop[i]] - trial_starts[i] for i in range(len(trial_starts))]

    stim_len_samples = int(np.round(stim_dur*1000))
    bin_size = 1

    nbins = int(stim_len_samples // bin_size)
    weights = np.zeros(nbins)
    trial_durations_samples = ((stim_df.stop_time.values - stim_df.start_time.values) * 1000).astype(int)
    weights = (trial_durations_samples >= np.arange(nbins)[:, None]).sum(axis=1)
    psth_idx, counts = np.unique(np.round(np.concatenate(spike_times) * 1000 / bin_size).astype(int), return_counts=True)
    psth = np.zeros(nbins)
    psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
    psth[weights > 0] /= weights[weights > 0]
    return psth, weights  


def calculate_EV_nwb(nwb_file, unit_id, intervals_name, max_dur = None):
    # can either pass a string or an NWBFile object
    if isinstance(nwb_file, str):
        nwb_io = nwb.NWBHDF5IO(nwb_file, 'r')
        nwbfile = nwb_io.read()
    else:
        nwbfile = nwb_file

    # get intervals and spike times from database
    all_trials = nwbfile.intervals[intervals_name].to_dataframe()
    unit_spike_times = nwbfile.units[unit_id].spike_times.values[0]
    # get unit valid intervals
    all_valid_intervals = nwbfile.intervals['unit_intervals'].to_dataframe()
    unit_valid_intervals = all_valid_intervals[all_valid_intervals['unit_id'] == unit_id]

    # remove trials that are not in valid intervals
    valid_trials = all_trials.apply(lambda x: any((unit_valid_intervals.start_time < x.start_time) & (unit_valid_intervals.stop_time > x.stop_time)), axis=1)
    all_trials = all_trials[valid_trials]
    if len(all_trials) == 0:
        print('No valid trials found for unit %s' % unit_id)
        return None
    if max_dur is not None:
        all_trials['stop_time'] = np.minimum(all_trials['stop_time'], all_trials['start_time'] + max_dur)

    psth_half1 = []
    psth_half2 = []
    weights_1 = []
    weights_2 = []
    for stim_name, stim_df in all_trials.groupby('stimuli_name'):
        if len(stim_df) < 2:
            continue
        # lets split the trials into two halves
        stim = nwbfile.stimulus[stim_name]
        stim_sample_rate = stim.rate
        stim_len = stim.data.shape[0] / stim_sample_rate
        stim_dur = stim_len

        stim_df1 = stim_df.iloc[:len(stim_df):2]
        stim_df2 = stim_df.iloc[1:len(stim_df):2]
        psth1, weights1 = psth_nwb(stim_df1, unit_spike_times, stim_dur)
        psth2, weights2 = psth_nwb(stim_df2, unit_spike_times, stim_dur)
        psth_half1.append(psth1)
        psth_half2.append(psth2)
        weights_1.append(weights1)
        weights_2.append(weights2)
    psth_half1 = np.concatenate(psth_half1)
    psth_half2 = np.concatenate(psth_half2)
    weights_1 = np.concatenate(weights_1)
    weights_2 = np.concatenate(weights_2)
    wHann = windows.hann(21, sym=True)   # The 21 ms (number of points) hanning window used to smooth the PSTH
    wHann = wHann/sum(wHann)
    psth_half1 = np.convolve(psth_half1, wHann, mode='same')
    psth_half2 = np.convolve(psth_half2, wHann, mode='same')
    r12 = weighted_corr(psth_half1, psth_half2, weights_1 + weights_2)
    SNRHalf = 1/(r12*r12)-1
    SNRAll = SNRHalf*2
    R2Ceil = 1/(1+SNRAll)   # This is the ceiling value of R2 

    print('R2Ceil = %.3f'%R2Ceil)
    return R2Ceil

def preprocess_sound(raw_stim_files, raw_resp_files, preprocess_type='ft', stim_params=None,
                     output_dir=None, stim_output_pattern='preprocessed_stim_%d',
                     resp_output_pattern='preprocessed_resp_%d'):
    
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
            stim_params['fband'] = 120
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

def preprocess_response(spikeTrials, stimLength, sampleRate, multiplier=1e-3):
    # multiplier should convert spikeTrials units to seconds
    spikeIndicies = []
    for trials in spikeTrials:
        # turn spike times (ms) into indexes at response sample rate
        
        stimes = np.round(trials*multiplier * sampleRate).astype(int) - 1
        # Choose within stimulus
        stimes = stimes[stimes >= 0]
        stimes = stimes[stimes < np.round(stimLength*sampleRate)]
        spikeIndicies.append(stimes)

    psth = make_psth(spikeIndicies, np.round(stimLength*sampleRate), 1)
    resp = {
        'type': 'psth',
        'sampleRate': sampleRate,
        'rawSpikeTimes': spikeTrials,
        'rawSpikeIndicies': spikeIndicies,
        'weights': np.ones(len(psth)),
        'psth': psth
    }
    return resp


def make_psth(spikeTrialsInd, stimdur, binsize):
    nbins = int(np.round(stimdur / binsize))
    psth = np.zeros(nbins)

    ntrials = len(spikeTrialsInd)

    for trial in spikeTrialsInd:
        sindxs = np.round(trial / binsize).astype(int) 
        psth[sindxs] += 1

    psth /= ntrials
    return psth

def compute_srdata_means(srData, maxRespLen):
    # Computes the average stimulus for each spatial dimension, the average response (spikes/sample),
    # and the time-varying response in delete one format for JN statistics.

    pairCount = len(srData['datasets'])


    # compute stim and response averages
    stimSum = np.zeros((srData['nStimChannels'], ))
    stimCountSum = 0
    respSum = np.zeros((1, maxRespLen))
    meanSum = 0
    tvRespCount = np.zeros((pairCount, maxRespLen))

    # first compute all the sums
    for k in range(pairCount):

        # Stimulus mean for each spatial dimension (i.e. frequency band for spectro-temporal)
        ds = srData['datasets'][k]
        spec = ds['stim']['tfrep']['spec']
        stimSum += np.sum(spec, axis=1)
        stimCountSum += spec.shape[1]

        # Response averates
        psth = ds['resp']['psth']

        # Time varying average response
        rlen = maxRespLen - len(psth)
        nresp = np.append(psth, np.zeros((1, rlen)))
        respSum = respSum + nresp
        tvIndx = np.arange(len(psth))
        tvRespCount[k, tvIndx] = 1

        # Overall response average
        meanSum = meanSum + psth.mean()

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
        tvcnts[tvcnts < 1] = 1

        tvRespAvg[k, :] = (respSum - nresp) / tvcnts

        # smooth with hanning window
        pprod = convolve(tvRespAvg[k, :], hwin, mode='full')
        sindx = halfTau+coff
        eindx = round(len(pprod)-halfTau) + 1
        tvRespAvg[k, :] = pprod[sindx:eindx]

    stimAvg = stimSum / stimCountSum
    respAvg = meanSum / pairCount

    return stimAvg, respAvg, tvRespAvg


def split_psth(spikeTrialsInd, stimLengthMs):

    halfSize = int(len(spikeTrialsInd)/2)
    spikeTrials1 = [None]*halfSize
    spikeTrials2 = [None]*halfSize


    for j, trial in enumerate(spikeTrialsInd):
        indx = int((j + 1) // 2)
        if indx - 1 >= halfSize:
            break
        if j%2 == 0:
            spikeTrials1[indx-1] = trial
        else:
            spikeTrials2[indx-1] = trial
            
    psth = make_psth(spikeTrialsInd, stimLengthMs, 1)
    psth1 = make_psth(spikeTrials1, stimLengthMs, 1)
    psth2 = make_psth(spikeTrials2, stimLengthMs, 1)

    psthdata = {'psth': psth, 'psth_half1': psth1, 'psth_half2': psth2}
    return psthdata