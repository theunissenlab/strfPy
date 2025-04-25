import tempfile
import os
import numpy as np

from scipy.signal import detrend
from scipy.signal import windows

# from .strfSetup import strflab2DS
from .DirectFit import direct_fit
from .calcAvg import df_Check_And_Load

def trnDirectFit(modelParams=None, datIdx=None, options=None, globalDat=None, *args, **kwargs):
    """
    Trains a direct fit model and sets the model parameters.

    Args:
    - modelParams: dictionary containing model parameters
    - datIdx: indices of the data to be used for training
    - options: dictionary containing options for training
    - *args: optional additional arguments

    Returns:
    - modelParams: updated dictionary of model parameters
    - options: updated dictionary of options
    """

    ## set default parameters and return if no arguments are passed
    if len(args) == 0:
        options = {}
        options['funcName'] = 'trnDirectFit'
        options['tolerances'] = [0.100, 0.050, 0.010, 0.005, 1e-03, 1e-04, 5e-05, 0]
        options['sparsenesses'] = [0, 1, 2, 3, 4, 5, 6, 7]
        options['separable'] = 0
        options['timeVaryingPSTH'] = 0
        options['timeVaryingPSTHTau'] = 31
        options['stimSampleRate'] = 1000
        options['respSampleRate'] = 1000
        options['infoFreqCutoff'] = 100
        options['infoWindowSize'] = 0.250

        tempDir = tempfile.gettempdir()
        options['outputDir'] = tempDir

        modelParams = options
        return modelParams, options

    if modelParams['type'] != 'lin' or modelParams['outputNL'] != 'linear':
        raise ValueError('trnDirectFit only works for linear models with no output nonlinearity!')
        
    if options['respSampleRate'] != options['stimSampleRate']:
        raise ValueError('trnDirectFit: Stimulus and response sampling rate must be equal!')

    global globDat
    
    globDat = globalDat
    
    os.makedirs(options['outputDir'], exist_ok=True)
    
    # convert strflab's stim/response data format to direct fit's data format
    DS = strflab2DS(globDat['stim'], globDat['resp'], globDat['weight'], globDat['groupIdx'], options['outputDir'])
    
    # set up direct fit parameters
    params = {
        'DS': DS,
        'NBAND': globDat['stim'].shape[1],
        'Tol_val': options['tolerances'],
        'setSep': options['separable'],
        'TimeLagUnit': 'frame',
        'timevary_PSTH': 0,
        'smooth_rt': options['timeVaryingPSTHTau'],
        'ampsamprate': options['stimSampleRate'],
        'respsamprate': options['respSampleRate'],
        'outputPath': options['outputDir'],
        'use_alien_space': 0,
        'alien_space_file': '',
        'TimeLag': int(np.ceil(np.max(np.abs(modelParams['delays']))))
    }
    
    # run direct fit
    strfFiles = direct_fit(params)
    
    # get computed stim and response means
    svars = np.load(os.path.join(options['outputDir'], 'stim_avg.npz'), allow_pickle=True)
    stimAvg = svars['stim_avg']
    respAvg = svars['constmeanrate']
    tvRespAvg = svars['Avg_psth']
    
    numSamples = len(DS)
    
    # compute some indices to use later
    halfIndx = params['TimeLag'] + 1
    startIndx = halfIndx + round(np.min(modelParams['delays']))
    endIndx = halfIndx + round(np.max(modelParams['delays']))
    strfRng = range(startIndx-1, endIndx)
    
    # subtract mean off of stimulus (because direct fit does this)
    for k in range(int(globDat['stim'].shape[0])):
        globDat['stim'][k,:] -= stimAvg.T

    # Make a smoothing window to calculate and R2 as well
    wHann = windows.hann(params['smooth_rt'], sym=True)  # The 31 ms (number of points) hanning window used to smooth the PSTH
    wHann = wHann / sum(wHann)

    
    # compute information values for each set of jackknifed strfs per tolerance value
    print('Finding best STRF by computing info values across sparseness and tolerance values...')
    bestInfoVal = -1
    bestStrf = -1
    bestTol = -1
    bestSparseness = -1
    spvals = options['sparsenesses']

    # Make space for R2CV
    R2CV = np.zeros((len(strfFiles), len(spvals)))

    for k in range(len(strfFiles)):    # for each tolerance value
        svars = np.load(strfFiles[k], allow_pickle=True)
        
        strfsJN = svars['STRFJN_Cell']
        strfsJN_std = svars['STRFJNstd_Cell']
        strfMean = svars['STRF_Cell']
        strfStdMean = np.mean(strfsJN_std, axis=2)
    
        for q in range(len(spvals)):
            smoothedMeanStrf = df_fast_filter_filter(strfMean, strfStdMean, spvals[q])
            smoothedMeanStrfToUse = smoothedMeanStrf[:, strfRng]

            infoSum = 0
            numJNStrfs = numSamples
            infoTBins = 0
            simple_sum_yy = 0
            simple_sum_y =  0
            simple_sum_error = 0
            simple_sum_count = 0
            for p in range(numJNStrfs):

                smoothedMeanStrfJN = df_fast_filter_filter(strfsJN[:, :, p], strfsJN_std[:, :, p], spvals[q])
                strfToUse = smoothedMeanStrfJN[:, strfRng]
                
                srRange = np.where(globDat['groupIdx'] == p)[0]
                stim = globDat['stim'][srRange, :]
                rresp = globDat['resp'][srRange]

                # Skip the very short stims to perform the coherence.
                if (len(rresp) < options['infoWindowSize']):
                    continue

                gindx = np.ones((1, stim.shape[0]))

                #compute the prediction for the held out stimulus
                mresp = conv_strf(stim, modelParams['delays'], np.real(strfToUse), gindx)

                if not options['timeVaryingPSTH']:
                    mresp = mresp + respAvg
                else:
                    mresp = mresp + tvRespAvg[p, :len(mresp)]
                
                #compute coherence and info across pairs
                # doing this for single trials
                # for (data in rawData):
                cStruct = compute_coherence_mean(mresp, rresp, options['respSampleRate'], options['infoFreqCutoff'], options['infoWindowSize'] )
                infoSum += np.real(cStruct['info'])*len(mresp)
                infoTBins += len(mresp)

                # Also calculate an R2
                # Get values to calculate R2-CV - here it is the coefficient of determination
                yw = df_Check_And_Load(DS[p]['weightfiles'])   # the index is either p or srRange...  I need to figure this out
                y = np.convolve(rresp, wHann, mode="same")
                sum_count = np.sum(yw)
                sum_y = np.sum(y*yw)
                sum_yy = np.sum(y*y*yw)
                sum_error2 = np.sum(((mresp-y)**2)*yw)

                simple_sum_yy += sum_yy
                simple_sum_y +=  sum_y
                simple_sum_error += sum_error2
                simple_sum_count += sum_count
                
        
            avgInfo = infoSum / infoTBins   # This is now normalized by the stimulus length....
            y_mean = simple_sum_y/simple_sum_count
            y_var = simple_sum_yy/simple_sum_count - y_mean**2
            y_error = simple_sum_error/simple_sum_count

            # This is a "one-trial" CV
            R2CV[k,q] = 1.0 - y_error/y_var

            print(f"Tolerance={options['tolerances'][k]}, Sparseness={spvals[q]}, Avg. Prediction Info={avgInfo}, R2CV={R2CV[k,q]}")

            # did this sparseness do better?
            if avgInfo > bestInfoVal:
                bestTol = options['tolerances'][k]
                bestSparseness = spvals[q]
                bestInfoVal = avgInfo
                bestStrf = smoothedMeanStrfToUse
                bestR2CV = R2CV[k,q]
                

        ## get best strf
    
    print('Best STRF found at tol=%f, sparseness=%d, info=%.2f bits R2=%.4f\n' % (bestTol, bestSparseness, bestInfoVal, bestR2CV))
    
    modelParams['w1'] = bestStrf
    modelParams['R2CV'] = R2CV
    
    if not options['timeVaryingPSTH']:
        modelParams['b1'] = respAvg
    else:
        modelParams['b1'] = tvRespAvg[p, :mresp.shape[1]]


     
    return modelParams, options
    




def df_fast_filter_filter(forward, forwardJN_std, nstd):
    # smooths out the filter for displaying or calculation purposes.
    # Faster than filter_filter, but less fancy.
    # Scales the filter everywhere by a sigmoid in forward/forwardJN_std, with
    # inflection point at nstd, and a dynamic range from nstd - .5 to nstd + .5
    
    if nstd > 0:
        epsilon = 10**-8 # To prevent division by 0.
        factor = (1 + np.tanh(2*(np.abs(forward)-np.abs(nstd*forwardJN_std))/(epsilon + np.abs(forwardJN_std))))/2
        s_forward = factor * forward
    else:
        s_forward = forward
    
    return s_forward



def conv_strf(allstim, delays, strf, groupIndex):
    nDatasets = len(np.unique(groupIndex))
    timeLen = allstim.shape[0]
    a = np.zeros((timeLen, 1), dtype=complex)
    
    for k in range(nDatasets):
        rng = np.where(groupIndex[0] == k+1)[0]
        soff = rng[0]
        stim = allstim[rng, :]
        for ti in range(len(delays)):
            # at = np.dot(stim, strf[:, ti])
            at = np.matmul(stim, strf[:, ti]).T

            thisshift = delays[ti]
            if thisshift >= 0:
                a[soff+thisshift:,0] += at[:timeLen-thisshift]
            else:
                offset = thisshift % timeLen
                a[soff:offset] += at[-thisshift:]
    
    return a.T



def compute_coherence_mean(modelResponse, psth, sampleRate, freqCutoff=-1, windowSize=0.500):
    
    # reshape
    modelResponse = modelResponse.T
    psth = psth.reshape(len(psth), 1)
    
    # put psths in matrix for mtchd_JN
    if len(modelResponse) != len(psth):
        minLen = min(len(modelResponse), len(psth))
        modelResponse = modelResponse[:minLen]
        psth = psth[:minLen]
    x = np.column_stack([modelResponse, psth])
    
    # compute # of time bins per FFT segment
    minFreq = round(1 / windowSize)
    numTimeBin = round(sampleRate * windowSize)
    
    # get default parameter values
    vargs = [x, numTimeBin, sampleRate]
    x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers = df_mtparam(*vargs)
    
    # compute jacknifed coherence
    y, fpxy, cxyo, cxyo_u, cxyo_l, stP = df_mtchd_JN(x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers)
    
    # normalize coherencies
    cStruct = {}
    cStruct['f'] = fpxy
    cStruct['c'] = cxyo[:, 0, 1]**2
    cStruct['cUpper'] = cxyo_u[:, 0, 1]**2
    
    clo = np.real(cxyo_l[:, 0, 1])
    closgn = np.sign(np.real(clo))
    cStruct['cLower'] = (clo**2) * closgn
    
    # restrict frequencies analyzed to the requested cutoff and minimum frequency given the window size
    if freqCutoff != -1:
        findx = np.where(cStruct['f'] < freqCutoff)[0]
        eindx = max(findx)
        indx = np.arange(eindx+1)
        
        cStruct['f'] = cStruct['f'][indx]
        cStruct['c'] = cStruct['c'][indx]
        cStruct['cUpper'] = cStruct['cUpper'][indx]
        cStruct['cLower'] = cStruct['cLower'][indx]
    
    if minFreq > 0:
        findx = np.where(cStruct['f'] >= minFreq)[0]
        sindx = min(findx)
        cStruct['f'] = cStruct['f'][sindx:]
        cStruct['c'] = cStruct['c'][sindx:]
        cStruct['cUpper'] = cStruct['cUpper'][sindx:]
        cStruct['cLower'] = cStruct['cLower'][sindx:]
    
    # compute information by integrating log of 1 - coherence
    df = cStruct['f'][1] - cStruct['f'][0]
    cStruct['minFreq'] = minFreq
    cStruct['info'] = -df * np.sum(np.log2(1 - cStruct['c']))
    cStruct['infoUpper'] = -df * np.sum(np.log2(1 - cStruct['cUpper']))
    cStruct['infoLower'] = -df * np.sum(np.log2(1 - cStruct['cLower']))
    
    return cStruct



def rv(a):
    b = a
    sz = np.shape(a)
    isvect = (sz[0] == 0) or (sz[1] == 0)
    
    if (isvect):
        if (sz[0] == 0):
            b = a.T
    return b



def df_mtparam(*varg):
    P=varg
    nargs = len(P)
    x = P[0]
    if nargs < 2 or P[1] is None:
        nFFT = 1024
    else:
        nFFT = P[1]
    if nargs < 3 or P[2] is None:
        Fs = 2
    else:
        Fs = P[2]
    if nargs < 4 or P[3] is None:
        WinLength = nFFT
    else:
        WinLength = P[3], nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers
    if nargs < 5 or P[4] is None:
        nOverlap = WinLength // 2
    else:
        nOverlap = P[4]
    if nargs < 6 or P[5] is None:
        NW = 3
    else:
        NW = P[6]
    if nargs < 7 or P[6] is None:
        Detrend = ''
    else:
        Detrend = P[6]
    if nargs < 8 or P[7] is None:
        nTapers = 2 * NW - 1
    else:
        nTapers = P[7]

    # Now do some computations that are common to all spectrogram functions
    winstep = WinLength - nOverlap

    nChannels = x.shape[1]
    nSamples = x.shape[0]

    # check for column vector input
    if nSamples == 1:
        x = x.T
        nSamples = x.shape[0]
        nChannels = 1

    # calculate number of FFTChunks per channel
    nFFTChunks = round(((nSamples - WinLength) / winstep))
    # turn this into time, using the sample frequency
    t = winstep * np.arange(nFFTChunks) / Fs

    # set up f and t arrays
    if np.all(np.isreal(x)):
        # x purely real
        if nFFT % 2:
            # nfft odd
            select = np.arange(1, (nFFT + 1) // 2)
        else:
            select = np.arange(1, nFFT // 2 + 1)
        nFreqBins = len(select)
    else:
        select = np.arange(1, nFFT + 1)
        nFreqBins = nFFT

    f = (select - 1) * Fs / nFFT

    # return x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers, nChannels, nSamples, nFFTChunks, winstep, select, nFreqBins, f, t
    return x, nFFT, Fs, WinLength, nOverlap, NW, Detrend, nTapers




def df_mtchd_JN(x, nFFT=1024, Fs=2, WinLength=None, nOverlap=None, NW=3, Detrend=None, nTapers=None):
    # if WinLength is None:
    #     WinLength = nFFT
    # if nOverlap is None:
    #     nOverlap = WinLength // 2
    # if nTapers is None:
    #     nTapers = 2 * NW - 1

    WinLength = int(WinLength)
    winstep = int(WinLength - nOverlap)
    nOverlap = int(nOverlap)
    nFFT = int(nFFT)

    nChannels = x.shape[1]
    nSamples = x.shape[0]

    # check for column vector input
    if nSamples == 1:
        x = x.T
        nSamples = x.shape[0]
        nChannels = 1

    # calculate number of FFTChunks per channel
 
    nFFTChunks = round(((nSamples - WinLength) / winstep))
    # turn this into time, using the sample frequency
    t = winstep * np.arange(nFFTChunks-1).T / Fs

    # calculate Slepian sequences. Tapers is a matrix of size [WinLength, nTapers]

    # [JN,y,stP] = make_slepian(x,WinLength,NW,nTapers,nChannels,nFFTChunks,nFFT,Detrend,winstep);
    # allocate memory now to avoid nasty surprises later
    stP = np.zeros((nFFT, nChannels, nChannels))
    varP = np.zeros((nFFT, nChannels, nChannels))
    
    from scipy.signal.windows import dpss

    Tapers, V = dpss(WinLength, NW, nTapers, return_ratios=True)
    Tapers = Tapers.T

    Periodogram = np.zeros((nFFT, nTapers, nChannels), dtype=complex)  # intermediate FFTs
    Temp1 = np.zeros((nFFT, nTapers), dtype=complex)  # Temps are particular psd or csd values for a frequency and taper
    Temp2 = np.zeros((nFFT, nTapers), dtype=complex)
    Temp3 = np.zeros((nFFT, nTapers), dtype=complex)
    eJ = np.zeros((nFFT,), dtype=complex)
    JN = np.zeros((nFFTChunks, nFFT, nChannels, nChannels), dtype=complex)
    # jackknifed cross-spectral-densities or csd. Note: JN(.,.,1,1) is
    # the power-spectral-density of time series 1 and JN(.,.,2,2) is the
    # psd of time series 2. Half-way through this code JN(.,.,1,2)
    # ceases to be the csd of 1 and 2 and becomes the abs coherency of 1
    # and 2.
    y = np.zeros((nFFT, nChannels, nChannels), dtype=complex)  # output array for csd
    Py = np.zeros((nFFT, nChannels, nChannels))  # output array for psd's

    # New super duper vectorized alogirthm
    # compute tapered periodogram with FFT
    # This involves lots of wrangling with multidimensional arrays.


    TaperingArray = np.tile(Tapers[:, :, np.newaxis], (1, 1, nChannels))
    for j in range(nFFTChunks):
        Segment = x[j*winstep:j*winstep+WinLength, :]

        if bool(Detrend):
            Segment = detrend(Segment, axis=0, type=Detrend)

        # SegmentsArray = np.tile(Segment[None, :, :], (nTapers, 1, 1))
        # SegmentsArray = np.transpose(np.tile(Segment, (1, 1, nTapers)), (0, 2, 1))
        SegmentsArray = np.tile(Segment[:, None, :], (1, nTapers, 1))

        TaperedSegments = TaperingArray * SegmentsArray

        Periodogram[:,:,:] = np.fft.fft(TaperedSegments, n=nFFT, axis=0)

        for Ch1 in range(nChannels):
            for Ch2 in range(Ch1, nChannels):
                Temp1 = Periodogram[:, :, Ch1]
                Temp2 = np.conj(Periodogram[:, :, Ch2])
                Temp3 = Temp1 * Temp2

                # eJ and eJ2 are the sum over all the tapers.
                eJ = np.sum(Temp3, axis=1) / nTapers
                JN[j, :, Ch1, Ch2] = eJ  # Here it is just the cross-power for one particular chunk.
                y[:, Ch1, Ch2] += eJ  # y is the sum of the cross-power

    # now fill other half of matrix with complex conjugate
    for Ch1 in range(nChannels):
        for Ch2 in range(Ch1+1, nChannels): # don't compute cross-spectra twice
            y[:, Ch2, Ch1] = y[:, Ch1, Ch2]
            Py[:, Ch1, Ch2] = np.arctanh(abs(y[:, Ch1, Ch2] / np.sqrt(abs(y[:, Ch1, Ch1]) * abs(y[:, Ch2, Ch2]))))

    for j in range(nFFTChunks):
        JN[j, :, :, :] = abs(y - np.squeeze(JN[j, :, :, :])) # This is where it becomes the JN quantity (the delete one)
        for Ch1 in range(nChannels):
            for Ch2 in range(Ch1+1, nChannels):
                # Calculate the transformed coherence
                JN[j, :, Ch1, Ch2] = np.arctanh(np.real(JN[j, :, Ch1, Ch2]) / np.sqrt(abs(JN[j, :, Ch1, Ch1]) * abs(JN[j, :, Ch2, Ch2])))
                # Obtain the pseudo values
                JN[j, :, Ch1, Ch2] = nFFTChunks * Py[:, Ch1, Ch2].T - (nFFTChunks-1) * np.squeeze(JN[j, :, Ch1, Ch2])

    # upper and lower bounds will be 2 standard deviations away
    
    meanP = np.mean(JN, axis=0)
    for Ch1 in range(nChannels):
        for Ch2 in range(Ch1,nChannels):
            varP[:, Ch1, Ch2] = (1/nFFTChunks) * np.var(JN[:, :, Ch1, Ch2], axis=0)

    stP = np.sqrt(varP)
    
    Pupper = meanP + 2*stP
    Plower = meanP - 2*stP
    meanP = np.tanh(meanP)
    Pupper = np.tanh(Pupper)
    Plower = np.tanh(Plower)

    # set up f array
    # if not np.any(np.any(np.imag(x))):
    if np.all(np.isreal(x)):
        # x purely real
        if nFFT % 2 == 1:  # nfft odd
            select = np.arange((nFFT + 1) // 2 + 1)
        else:
            select = np.arange(nFFT // 2 + 1)
        meanP = meanP[select, :, :]
        Pupper = Pupper[select, :, :]
        Plower = Plower[select, :, :]
        y = y[select, :, :]
    else:
        select = np.arange(nFFT)

    fo = select * Fs / nFFT
    fo = fo.reshape(len(fo), 1)


    return y, fo, meanP, Pupper, Plower, stP


def strflab2DS(allstim, allresp, allweight, groupIndex, outputPath):
    npairs = len(np.unique(groupIndex))
    DS = [None]*npairs

    for k in range(npairs):
        rng = np.where(groupIndex == k)[0]
        stim = allstim[rng, :]
        resp = allresp[rng]
        weight = allweight[rng]

        dfds = {}

        # write spectrogram to intermediate file (direct fit requires this)
        stimfile = os.path.join(outputPath, f'df_temp_stim_{k}.npy')
        outSpectrum = stim.T
        np.save(stimfile, outSpectrum)
        dfds['stimfiles'] = stimfile

        # write response to intermediate file
        rfile = os.path.join(outputPath, f'df_temp_resp_{k}.npy')
        np.save(rfile, resp)
        dfds['respfiles'] = rfile

        # write weights to intermediate file
        wfile = os.path.join(outputPath, f'df_temp_weight_{k}.npy')
        np.save(wfile, weight)
        dfds['weightfiles'] = wfile

        dfds['nlen'] = stim.shape[0]
        dfds['ntrials'] = 1

        DS[k] = dfds

    return DS

