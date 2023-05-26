import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve

sys.path.append("/Users/frederictheunissen/Code/crcns-kailin/module")
from strfpy.timeFreq import timefreq
import pandas as pd



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
    for trials in spikeTrials:
        # turn spike times (ms) into indexes at response sample rate
        
        stimes = np.round(trials*1e-3 * sampleRate).astype(int) - 1
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

    halfSize = round(len(spikeTrialsInd)/2)
    spikeTrials1 = [None]*halfSize
    spikeTrials2 = [None]*halfSize


    for j, trial in enumerate(spikeTrialsInd):
        indx = int((j + 1) // 2)
        if j%2 == 0:
            spikeTrials1[indx-1] = trial
        else:
            spikeTrials2[indx-1] = trial
            
    psth = make_psth(spikeTrialsInd, stimLengthMs, 1)
    psth1 = make_psth(spikeTrials1, stimLengthMs, 1)
    psth2 = make_psth(spikeTrials2, stimLengthMs, 1)

    psthdata = {'psth': psth, 'psth_half1': psth1, 'psth_half2': psth2}
    return psthdata