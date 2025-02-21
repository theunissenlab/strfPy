import os
import numpy as np
import math
from .trnDirectFit import trnDirectFit

def linInit(nIn, delays, outputNL=None, freqDom=None):
    # Set default option values
    if delays is None:
        delays = np.array([0])
    if outputNL == None:
        outputNL = 'linear'
    if freqDom == None:
        freqDom = 0
    
    strf = {}
    strf['type'] = 'lin'
    strf['nIn'] = nIn
    strf['nWts'] = (nIn*len(delays) + 1)
    strf['delays'] = delays
    
    strf['w1'] = np.zeros((nIn, len(delays)))
    strf['b1'] = 0
    
    nlSet = ['linear', 'logistic', 'softmax', 'exponential']
    if outputNL in nlSet:
        strf['outputNL'] = outputNL
    else:
        raise ValueError('Unknown Output Nonlinearity!')
    
    strf['freqDomain'] = freqDom
    
    strf['internal'] = {}
    strf['internal']['compFwd'] = 1
    strf['internal']['prevResp'] = None
    strf['internal']['prevLinResp'] = None
    strf['internal']['dataHash'] = np.nan
    
    return strf



def srdata2strflab(srData, useRaw=False, preprocOptions={}):
    if 'meanSubtractStim' not in preprocOptions:
        preprocOptions['meanSubtractStim'] = False
    if 'scaleStim' not in preprocOptions:
        preprocOptions['scaleStim'] = False
    if 'meanSubtractResp' not in preprocOptions:
        preprocOptions['meanSubtractResp'] = False

    pairCount = len(srData['datasets'])
    totalStimLength = 0
    totalRespLength = 0
    numTrials = 1e10

    for k in range(pairCount):
        ds = srData['datasets'][k]
        numTrials = min(numTrials, len(ds['resp']['trialDurations']))
        totalStimLength += ds['stim']['tfrep']['spec'].shape[1]
        totalRespLength += len(ds['resp']['psth'])

    allstim = np.zeros((totalStimLength, srData['nStimChannels']))

    if useRaw:
        allresp = [None] * numTrials
    else:
        allresp = np.zeros(totalRespLength)
    groupIndex = np.zeros(totalRespLength)

    currentIndex = 0
    for k in range(pairCount):
        ds = srData['datasets'][k]

        stim = ds['stim']['tfrep']['spec'].T
        stimLen = stim.shape[0]
        resp = ds['resp']['psth']

        if stimLen != len(resp):
            raise ValueError(f"Stim and response lengths are not the same for dataset {k}!")
        eIndx = currentIndex + len(resp) - 1
        rng = np.arange(currentIndex, eIndx + 1)

        allstim[rng, :] = stim
        groupIndex[rng] = k

        if not useRaw:
            allresp[rng] = resp
        else:
            for trialNum in range(numTrials):
                spikes = allresp[trialNum]
                st = ds['resp']['rawSpikeIndicies'][trialNum]
                st = st[st <= stimLen-1]
                st = st + currentIndex
                st = st / srData['stimSampleRate']

                allresp[trialNum] = np.concatenate((spikes, st))

        currentIndex = eIndx + 1

    if preprocOptions.get('meanSubtractStim'):
        print("Subtracting off mean stimuli...")
        allstim -= srData['stimAvg'].T

    if preprocOptions.get('scaleStim'):
        print("Scaling input by std deviation...")
        allstim /= np.std(allstim)

    if preprocOptions.get('meanSubtractResp'):
        if preprocOptions.get('tvMean'):
            print("Subtracing off time-varying mean rate from response...")
            cindx = 0
            nstims = srData['tvRespAvg'].shape[0]
            for k in range(nstims):
                tvresp = srData['tvRespAvg'][k, :]
                eindx = cindx + len(tvresp) - 1
                allresp[cindx:eindx+1] -= tvresp
                cindx = eindx + 1
        else:
            print("Subtracing off scalar mean rate from response...")
            allresp -= srData['respAvg']

    return allstim, allresp, groupIndex



def strfData(stim, resp, groupIdx=None):
    """
    Takes [stim] and [resp] from the preprocessing routines, and set them
    as global data.
    
    Parameters
    ----------
    stim : numpy.ndarray
        NxD matrix of N samples and D dimensions.
    resp : numpy.ndarray or list
        Nx1 column vector of N samples, or a list of spike time vectors.
    groupIdx : numpy.ndarray or None, optional
        Nx1 vector of integers, specifying which group the [stim] samples
        belongs. If it is None, each sample is its own group.
    
    Returns
    -------
    None, but sets the following global variable. You must have a line
    `global globDat` in your code in order to use the global variable [globDat].
    
    globDat : dict
        Global variable containing following fields:
            'stim' : same as input [stim]
            'resp' : same as input [resp]
            'nSample' : # of samples = N, the common dimension of [stim] and [resp].
            'groupIdx' : same as input [groupIdx]
            'dataHash' : hash value of the data set
    
    Raises
    ------
    Warning : If [stim] and [resp] do not have same # of rows, or if [groupIdx] does not have same length as [resp].
    """
    # Get input size
    stimSiz = stim.shape
    respLen = len(resp)
    nSample = min(stimSiz[0], respLen)
    if nSample != respLen:
        print('Warning: [stim] and [resp] must have same # of rows.')
    if groupIdx is not None and len(groupIdx) != nSample:
        print('Warning: [groupIdx] must have same length as [resp].')

    # Compute hash for the data set
    if isinstance(resp, list):
        # resp is a list of spike time vectors
        hresp = []
        for i in range(len(resp)):
            st = resp[i]
            hresp.extend(st)
        hresp = np.array(hresp)
    else:
        hresp = resp
    respHash = 100 * abs(np.nanmean(hresp.astype(float)) + np.nanmean(hresp[::11].astype(float)))
    stimHash = 100 * abs(np.nanmean(stim[::109].astype(float)))
    magdif = np.log10((respHash + 0.00012) / (stimHash + 0.00011))
    dataHash = respHash + stimHash * 10 ** magdif

    # Set global variable
    global globDat
    globDat = {
        'stim': stim,
        'resp': resp,
        'nSample': nSample,
        'groupIdx': groupIdx,
        'dataHash': dataHash,
    }

    hresp = resp
    if isinstance(resp, list):
        hresp = []
        for k in range(len(resp)):
            st = resp[k]
            hresp.extend(st)
            
    respHash = 100 * abs(np.nanmean(np.double(hresp)) + np.nanmean(np.double(hresp[::11])))
    stimHash = 100 * abs(np.nanmean(np.double(stim[::109])))
    magdif = np.log10((respHash + 0.00012) / (stimHash + 0.00011))
    dataHash = respHash + stimHash * 10**magdif
    globDat['dataHash'] = dataHash

    return globDat



def strfOpt(strf, datIdx, options,globDat,  *args):
    """
    strfOpt is a helper function which facilitates the training of
    STRFs. It calls any of the middle-layer training functions to
    optimize the parameters, and passes a dictionary to govern the
    optimization behaviour.

    vbnet
    Copy code
    Args:
        strf: model structure obtained via upper level *Init functions
        datIdx: a vector containing indices of the samples to be used in the 
                fitting.
        options: A dictionary which specifies and determines the behavior of the
                training function. (ex. options['funcName']='resampBootstrap').
                Specific values for OPTIONS depend on which algorithm
                is used.  Type help trn* for that algorithm's
                options parameters.
        *args: additional arguments to be passed to the training function

    Returns:
        strf: structure containing model fit by the training algorithm
        options: option dictionary, with any additional fields added by fitting algorithm
        *varargs: additional arguments returned by the training function
    """


    s = globals()[options['funcName']](strf, datIdx, options, globDat, 1)
   
    strf = s[0]

    if len(s) > 1:
        options = s[1]

        # If there are additional arguments, extract them
        nextra = len(s) - 2
        if nextra > 0:
            varargs = s[2:]
            return strf, options, *varargs

    return strf, options





def strfFwd(strf, datIdx, globDat):
    """
    Gets predicted response for a stimulus and STRF model.
    
    Args:
    - strf: strf model structure
    - datIdx: a vector containing indices of the samples to be used in the fitting.
    
    Returns:
    - strf: strf model structure
    - varargout: any other output generated by the STRF's fwd function (ex. response before output nonlinearity)
    """
    
    fwdstr = strf['type'] + 'Fwd'
    s = globals()[fwdstr](strf, datIdx, globDat)

    strf = s[0]

    if len(s) > 1:
        varargout = s[1:]
        return strf, varargout
    
    return strf



def linFwd(strf, datIdx, dat):
    global globDat
    globDat=dat
    samplesize = globDat['nSample']
    if strf['internal']['compFwd'] == 0 and samplesize == len(strf['internal']['prevResp']) and strf['internal']['dataHash'] == globDat['dataHash']:
        resp_strf_dat = strf['internal']['prevResp'][datIdx]
        a_dat = strf['internal']['prevLinResp'][datIdx]
        return strf, resp_strf_dat, a_dat
    
    a = np.zeros((samplesize), dtype=complex)
    for ti in range(len(strf['delays'])):
        at = globDat['stim'] @ strf['w1'][:,ti]
        thisshift = strf['delays'][ti]
        if thisshift >= 0:
            a[thisshift:] += at[:samplesize-thisshift]
        else:
            offset = thisshift % samplesize
            a[:offset] += at[-thisshift:]
    
    a += strf['b1']
    
    if strf['outputNL'] == 'linear':
        resp_strf = a
    elif strf['outputNL'] == 'logistic':
        maxcut = -math.log(np.finfo(float).eps)
        mincut = -math.log(1.0/np.finfo(float).tiny - 1)
        a = np.minimum(a, maxcut)
        a = np.maximum(a, mincut)
        resp_strf = 1.0/(1.0 + np.exp(-a))
    elif strf['outputNL'] == 'softmax':
        nout = a.shape[1]
        maxcut = math.log(np.finfo(float).max) - math.log(nout)
        mincut = math.log(np.finfo(float).tiny)
        a = np.minimum(a, maxcut)
        a = np.maximum(a, mincut)
        temp = np.exp(a)
        resp_strf = temp/(np.sum(temp, axis=1)[:,np.newaxis])
        resp_strf[resp_strf<np.finfo(float).tiny] = np.finfo(float).tiny
    elif strf['outputNL'] == 'exponential':
        resp_strf = np.exp(a)
    else:
        raise ValueError('Unknown activation function %s' % strf['outputNL'])
    
    nanmask = np.mod(strf['delays'], globDat['stim'].shape[0]+1)
    nanmask = nanmask[np.nonzero(nanmask)] # no mask for delay 0
    a[nanmask] = np.nan
    resp_strf[nanmask] = np.nan
    
    resp_strf_dat = resp_strf[datIdx]
    a_dat = a[datIdx]
    
    strf['internal']['compFwd'] = 0
    strf['internal']['prevResp'] = resp_strf
    strf['internal']['prevLinResp'] = a
    strf['internal']['dataHash'] = globDat['dataHash']
    
    return strf, resp_strf_dat, a_dat
