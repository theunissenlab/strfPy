# Dependencies - General Stuff
import tempfile
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import windows, fftconvolve
# from scipy.io import wavfile
from scipy.special import genlaguerre
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import pynwb as nwb
import pickle

# Depednecies from Theunissen Lab
# from soundsig.sound import BioSound
from soundsig.sound import spec_colormap
from soundsig.sound import mps

# strfpy
from .makePSTH import SpikeResponse
from . import findDatasets, preprocSound, strfSetup, trnDirectFit, plotTfrep

# from strflab import preprocess_sound, srdata2strflab, linInit, strfData, trnDirectFit, strfOpt

import pynwb


## UTILITY FUNCTIONS


def log_downsample_spec(spec, f, nbins, fmin=200, fmax=8000):
    """Downsample the spectrogram by taking the mean of the log frequencies in each bin

    Parameters
    ----------
    spec : np.ndarray
        The spectrogram to downsample
    f : np.ndarray
        The frequencies of the spectrogram
    nbins : int
        The number of bins to downsample to
    fmin : float
        The minimum frequency to consider
    fmax : float
        The maximum frequency to consider
    """
    log_freqs = np.logspace(np.log10(fmin), np.log10(fmax), nbins + 1)
    log_midpoints = (log_freqs[1:] + log_freqs[:-1]) / 2
    # bin the frequencies into 15 log spaced bins
    f_bins = np.searchsorted(log_freqs, f) - 1
    out_log_spec = np.zeros((nbins, spec.shape[1]))
    for i in range(nbins):
        inds = f_bins == i
        if np.any(inds):
            out_log_spec[i] = np.mean(spec[inds, :], axis=0)
        else:
            out_log_spec[i] = np.zeros(spec.shape[1])
    # plt.imshow(out_log_spec, aspect='auto',origin='lower', extent=[t.min(), t.max(), 0, nbins])
    # plt.yticks(np.arange(nbins)+.5,log_midpoints.astype(int));
    return out_log_spec, log_midpoints


def laguerre(xt, amp, tau, alpha, xorder):
    """
    Computes the Laguerre function.

    Parameters:
    xt (float or array-like): The input value(s) at which to evaluate the Laguerre function.
    amp (float): Amplitude scaling factor.
    tau (float): Time constant.
    alpha (float): Shape parameter.
    xorder (int): Order of the generalized Laguerre polynomial.

    Returns:
    float or array-like: The evaluated Laguerre function at the given input value(s).
    """
    return (
        amp
        * np.exp(-xt / tau)
        * np.power(xt / tau, alpha)
        * genlaguerre(xorder, alpha)(xt / tau)
    )

def dogs(xt, meanVal, ampPos, ampNeg, tPos, tNeg, sdPos, sdNeg):
    """ 
    Computes a Difference Of GaussianS or DOGS with a DC offset: meanval. 
    The amplitudes are specified as positive numbers both for the positive and negative DOGS """

    out = meanVal + ampPos*np.exp(-0.5*(xt-tPos)**2/sdPos**2) - ampNeg*np.exp(-0.5*(xt-tNeg)**2/sdNeg**2)
    return out

def dDogsDt(xt, meanVal, ampPos, ampNeg, tPos, tNeg, sdPos, sdNeg):
    """ 
    Computes the derivative of the Difference Of GaussianS or DOGS.
     The meanVal argument is not used but it is kept to match dogs """
                
    out = -ampPos*((xt-tPos)/(sdPos**2))*np.exp(-0.5*(xt-tPos)**2/sdPos**2) + ampNeg*((xt-tNeg)/(sdNeg**2))*np.exp(-0.5*(xt-tNeg)**2/sdNeg**2)
    return out

def Gauss(xt, t, sd):
    return (np.exp(-0.5*(xt-t)**2/sd**2))

def DGauss(xt, t, sd):
    return ((-(xt-t)/(sd**2))*np.exp(-0.5*(xt-t)**2/sd**2))


def arbitrary_kernel(
    pair,
    nPoints=200,
    event_key="onoff_feature",
    event_index_key="index",
    resp_key="psth",
    mult_values=False,
):
    """
    Generate a kernel matrix for a given event in the pair data.

    Parameters:
    pair (dict): A dictionary containing 'resp' and 'events' keys. 'resp' should have a 'psth' key with the response data.
                 'events' should have keys corresponding to event names and their values.
    nPoints (int, optional): Number of points for the kernel. Defaults to 200.
    event_key (str, optional): The name of the event to be used. Defaults to 'onoff_feature'.
    event_index_key (str, optional): The key to access the event indices of events specified. Defaults to 'index'.
    mult_values (bool, optional): If True, multiply the feature values by the event values. Defaults to False.

    Returns:
    np.ndarray: A 2D array representing the kernel matrix convolved with the event data.
    """
    nT = pair["resp"][resp_key].size
    feature = pair["events"][event_key]

    # check if resp and stim are the same samplerate
    if pair['resp']['sampleRate'] != pair['stim']['sampleRate']:
        conv_factor = pair['resp']['sampleRate'] / pair['stim']['sampleRate']
    else:
        conv_factor = 1

    feature = np.round(feature * conv_factor).astype(int)

    # handle the case of only one feature, just add a dimension so the new dim is <nT, 1>
    if feature.ndim == 1:
        feature = feature[:, np.newaxis]

    num_features = feature.shape[1]
    X = np.zeros((nPoints * num_features, nT))
    if mult_values:
        values = pair["events"]["%s_values" % event_key]
    else:
        values = np.ones(feature.shape[0])
    for i in range(num_features):
        feat_inds = pair["events"][event_index_key][feature[:, i] == 1].astype(int)
        X[i * nPoints : (i + 1) * nPoints, feat_inds] = values[feature[:, i] == 1]

    # now stack the kern_mat to
    kern_mat = np.vstack([np.eye(nPoints)] * num_features)
    X = fftconvolve(X, kern_mat, axes=1, mode="full")[:, :nT]
    return X


def fit_kernel_LG (learned_conv_kernel, nPoints, nD=2):
    '''
    Fits an arbitray funcntion of nPoints with a Laguerre function expansion'''

    minDist = 20   # Minimum distance between peaks in points - set at 20 ms
    minLat = 10    # Minimu latency
    def sum_n_laguerres(xt, *args):
        tau, alpha, *w = args
        nL = len(w)
        out = np.zeros_like(xt, dtype=float)
        for iL in range(nL):
            out += w[iL] * laguerre(xt, 1.0, tau, alpha, xorder=iL) 
        return out

    nLG = 7   # Fiting tau, alpha and five amplitudes for 5 LG
    basis_args = np.zeros((nD, nLG))
    for iEventType in range(nD):

        kernel_mean  = np.mean(learned_conv_kernel[iEventType, :])

        peakInd, peakVals = find_peaks( learned_conv_kernel[iEventType, minLat:] - kernel_mean, distance=minDist, height= (None, None) )
        troughInd, troughVals = find_peaks(-(learned_conv_kernel[iEventType, minLat:] - kernel_mean), distance=minDist, height= (None, None) )

        # Take the first of the two:
        if (peakInd[0] < troughInd[0]):
            ampMax = peakVals["peak_heights"][0]
            tau =  peakInd[0] + minLat
        else:
            ampMax = troughVals["peak_heights"][0]
            tau = troughInd[0] + minLat

        ampVal = ampMax/laguerre(tau, 1, tau, 1, 0)
        
        # Fit single to have a good starting point
        try:
            pdouble, pcov = curve_fit(
                sum_n_laguerres,
                np.arange(nPoints),
                learned_conv_kernel[iEventType, :]-kernel_mean,
                p0=[tau, 1, ampVal],
                bounds=(
                    [      0,       0, -np.inf],
                    [nPoints, nPoints,  np.inf],
                ),
                method="trf"
            )
        except RuntimeError:
            basis_args[iEventType, 0:3] = [tau, 1, ampVal]
            continue
        

        # Now try 5
        try:
            popt, pcov = curve_fit(
                sum_n_laguerres,
                np.arange(nPoints),
                learned_conv_kernel[iEventType, :]-kernel_mean,
                p0=[pdouble[0], pdouble[1], pdouble[2], 0, 0, 0, 0],
                bounds=(
                    [      0,       0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                    [nPoints, nPoints,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf],
                ),
                method="trf",
            )
        except RuntimeError:
            basis_args[iEventType, 0:3] = pdouble
            continue

        basis_args[iEventType, :] = popt

    return basis_args

def fit_kernel_DG(learned_conv_kernel, nPoints, nD=2):

    '''
    Fits an arbitray funcntion of nPoints with a difference of Guassian function expansion'''

    basis_args = np.zeros((nD, 7))

    # Find starting points - the bound might be too restrictive...
    for iEventType in range(nD):
        meanVal = np.mean(learned_conv_kernel[iEventType, :])
        ampPos = np.max(learned_conv_kernel[iEventType, :])-meanVal
        tPos = np.argmax(learned_conv_kernel[iEventType, :])
        ampNeg = np.abs(np.min(learned_conv_kernel[iEventType, :])-meanVal)
        tNeg = np.argmin(learned_conv_kernel[iEventType, :])
        sdPos = sdNeg = 20
        p0=[meanVal, ampPos, ampNeg, tPos, tNeg, sdPos, sdNeg]
        popt, pcov = curve_fit(
            dogs,
            np.arange(nPoints),
            learned_conv_kernel[iEventType, :],
            p0=p0,
            bounds=(
            [np.min(learned_conv_kernel[iEventType, :]), 0, 0, 0, 0, 0, 0],
            [np.max(learned_conv_kernel[iEventType, :]), np.inf, np.inf, nPoints, nPoints, nPoints, nPoints],
            ),
            method="trf",
        )
        basis_args[iEventType, :] = popt

    return basis_args


def generate_dogs_features(
    pair,
    feature_key,
    event_index_key="index",
    resp_key="psth",
    dogs_args=np.zeros((2, 7)),
    nPoints=300
):
    """
    Generate Laguerre features for a given pair of data.

    Parameters:
    pair (dict): Dictionary containing 'resp' and 'events' keys.
    feature_key (str): Key to access the features in the 'events' dictionary.
    event_index_key (str): Key to access the event indices in the 'events' dictionary. Default is 'index'.
    resp_key (str): Key to access the response data in the 'resp' dictionary. Default is 'psth'.
    dogs_args (numpy.ndarray): Numpy array of shape (nEventsTypes, 7) containing the DOGS parameters
                                    (meanVal, ampPos, ampNeg, tPos, tNeg, sdPos, sdNeg) for each event type. Default is np.zeros((2,7)).
    nPoints (int): Number of points for the DOGS functions. Default is 300.

    Returns:
    numpy.ndarray: A 2D array of shape (nFeatures*nLaguerre, nT) containing the generated Laguerre features.
    """
    # we will generate X and Y for each pair
    # dogs_args should be a numpy array of size nEventsTypes x 7
    nEventsTypes = dogs_args.shape[0]
    nT = pair["resp"][resp_key].size
    feature = pair["events"][feature_key]
    if feature.ndim == 1:
        feature = feature[:, np.newaxis]
    nFeatures = feature.shape[1]
    assert (
        nFeatures % nEventsTypes == 0
    )  # we expect the number of features to be a multiple of the number of event types
    nFeaturesPerEventType = nFeatures // nEventsTypes
    # the features are organized in the following way:
    # the first nFeatures/nEventTypes features are for the first event type, the second nFeatures/nEventTypes are for the second event type, ...
    nDOGS = 5
    X = np.zeros((nFeatures * nDOGS, nT))

    # DOGS functions for each order
    # the order of the matrix's rows are: Laguerre order 0-nDOGS for feature 1, Laguerre order 0-nDOGS for feature 2, ...
    dogs_mat = np.zeros((nFeatures * nDOGS, nPoints))
    
    x_t = np.arange(nPoints)
    for iEventType in range(nEventsTypes):
        meanVal, ampPos, ampNeg, tPos, tNeg, sdPos, sdNeg = dogs_args[iEventType]
        for iDOG in range(nDOGS):
            lag_start_ind = (
                iDOG * nFeaturesPerEventType * nEventsTypes
                + iEventType * nFeaturesPerEventType
            )
            if (iDOG == 0) :
                y = Gauss(x_t, tPos, sdPos)
            elif (iDOG == 1) : 
                y = DGauss(x_t, tPos, sdPos)
            elif (iDOG == 2) : 
                y = -Gauss(x_t, tNeg, sdNeg)
            elif (iDOG ==3) :
                y = -DGauss(x_t, tNeg, sdNeg)
            elif (iDOG ==4):
                y = np.ones(nPoints)
          
            y = y / np.sqrt(np.sum(y**2))
            dogs_mat[lag_start_ind : lag_start_ind + nFeaturesPerEventType, :] = y[
                np.newaxis, :
            ]

    # for each event we will convolve the guassian functions and derivatives with the feature value
    X[:, pair["events"][event_index_key]] = np.hstack([feature] * nDOGS).T
    # now convolve the laguerre function with the feature value
    X = fftconvolve(X, dogs_mat, axes=1, mode="full")[:, :nT]  # LG1: 20 pcONset, 20 pcOffset, LG1: 20 pcOnset, 20 pcOffset, ...
    return X


def generate_laguerre_features(
    pair,
    feature_key,
    event_index_key="index",
    resp_key="psth",
    laguerre_args=np.zeros((2, 2)),
    nLaguerrePoints=300,
    nLaguerre=5,
):
    """
    Generate Laguerre features for a given pair of data.

    Parameters:
    pair (dict): Dictionary containing 'resp' and 'events' keys.
    feature_key (str): Key to access the features in the 'events' dictionary.
    event_index_key (str): Key to access the event indices in the 'events' dictionary. Default is 'index'.
    resp_key (str): Key to access the response data in the 'resp' dictionary. Default is 'psth'.
    laguerre_args (numpy.ndarray): Numpy array of shape (nEventsTypes, 3) containing Laguerre function parameters
                                    (amplitude, tau, alpha) for each event type. Default is np.zeros((2,3)).
    nLaguerrePoints (int): Number of points for the Laguerre functions. Default is 300.
    nLaguerre (int): Number of Laguerre functions to generate. Default is 5.

    Returns:
    numpy.ndarray: A 2D array of shape (nFeatures*nLaguerre, nT) containing the generated Laguerre features.
    """
    # we will generate X and Y for each pair
    # laguerre_args should be a numpy array of size nEventsTypes x 3
    nEventsTypes = laguerre_args.shape[0]
    nT = pair["resp"][resp_key].size
    feature = pair["events"][feature_key]
    if feature.ndim == 1:
        feature = feature[:, np.newaxis]
    nFeatures = feature.shape[1]
    assert (
        nFeatures % nEventsTypes == 0
    )  # we expect the number of features to be a multiple of the number of event types
    nFeaturesPerEventType = nFeatures // nEventsTypes
    # the features are organized in the following way:
    # the first nFeatures/nEventTypes features are for the first event type, the second nFeatures/nEventTypes are for the second event type, ...
    X = np.zeros((nFeatures * nLaguerre, nT))

    # laguerre functions for each order
    # the order of the matrix's rows are: Laguerre order 0-nLaguerre for feature 1, Laguerre order 0-nLaguerre for feature 2, ...
    laguerre_mat = np.zeros((nFeatures * nLaguerre, nLaguerrePoints))
    laguerre_amp = 1.0
    x_t = np.arange(nLaguerrePoints)
    for iEventType in range(nEventsTypes):
        laguerre_tau, laguerre_alpha = laguerre_args[iEventType]
        for iLaguerre in range(nLaguerre):
            lag_start_ind = (
                iLaguerre * nFeaturesPerEventType * nEventsTypes
                + iEventType * nFeaturesPerEventType
            )
            y = laguerre(
                x_t,
                amp=laguerre_amp,
                tau=laguerre_tau,
                alpha=laguerre_alpha,
                xorder=iLaguerre,
            )
            y = y / np.sqrt(np.sum(y**2))
            laguerre_mat[lag_start_ind : lag_start_ind + nFeaturesPerEventType, :] = y[
                np.newaxis, :
            ]

    # for each event we will convolve the laguerre function with the feature value
    X[:, pair["events"][event_index_key]] = np.hstack([feature] * nLaguerre).T
    # now convolve the laguerre function with the feature value
    X = fftconvolve(X, laguerre_mat, axes=1, mode="full")[:, :nT]  # LG1: 20 pcONset, 20 pcOffset, LG1: 20 pcOnset, 20 pcOffset, ...
    return X


def get_simple_prediction_r2_Values(pair, ridge_conv_filter, nPoints, ntrials: int, smWindow = 31, mult_values=False):
# Returns all componnents needed to calculate the r2 for the segmented model
    
    if ~isinstance(ntrials, int):
        try:
            ntrials = int(ntrials)
        except ValueError:
            raise ValueError("ntrials argument must be an integer or convertible")
        
    resp = pair['resp']
    nactual = len(resp['trialDurations'])

    if (nactual < ntrials):
        return 0, 0, 0, 0, 0, 0
    elif (nactual > ntrials):
        # The Hanning window
        wHann = windows.hann(
            smWindow, sym=True
        )  # The 31 ms (number of points) hanning window used to smooth the PSTH
        wHann = wHann / sum(wHann)
        # recalculate the psth_smooth
        bin_size = 1
        nbins = len(resp['psth'])

        weights = (resp['trialDurations'][0:ntrials] >= np.arange(nbins)[:, None]).sum(axis=1)
        psth_idx, counts = np.unique(np.round(np.concatenate(resp['rawSpikeTimes'][0:ntrials]) * 1000 / bin_size).astype(int), return_counts=True)
        psth = np.zeros(nbins)
        psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
        psth[weights > 0] /= weights[weights > 0]
        y = np.convolve(psth, wHann, mode="same")
    else :
        y = pair["resp"]["psth_smooth"]

    x = arbitrary_kernel(pair, nPoints=nPoints, resp_key='psth_smooth', mult_values=mult_values)
    y_pred = ridge_conv_filter.predict(x.T)
    
    return np.sum(y), np.sum(y*y), np.sum(y_pred), np.sum(y_pred*y_pred), np.sum(y*y_pred), len(y_pred)


def get_prediction_r2_Values(pair, ridge, feature, laguerre_args, ridge_conv_filter, nPoints, ntrials: int, smWindow = 31, nLaguerre=5):
# Returns all componnents needed to calculate the r2 for the segmented + Identification model
    
    if ~isinstance(ntrials, int):
        try:
            ntrials = int(ntrials)
        except ValueError:
            raise ValueError("ntrials argument must be an integer or convertible")
        
    resp = pair['resp']
    nactual = len(resp['trialDurations'])

    if (nactual < ntrials):
        return 0, 0, 0, 0, 0, 0
    elif (nactual > ntrials):
        # The Hanning window
        wHann = windows.hann(
            smWindow, sym=True
        )  # The 31 ms (number of points) hanning window used to smooth the PSTH
        wHann = wHann / sum(wHann)
        # recalculate the psth_smooth
        bin_size = 1
        nbins = len(resp['psth'])

        weights = (resp['trialDurations'][0:ntrials] >= np.arange(nbins)[:, None]).sum(axis=1)
        psth_idx, counts = np.unique(np.round(np.concatenate(resp['rawSpikeTimes'][0:ntrials]) * 1000 / bin_size).astype(int), return_counts=True)
        psth = np.zeros(nbins)
        psth[psth_idx[psth_idx < nbins]] = counts[psth_idx < nbins]
        psth[weights > 0] /= weights[weights > 0]
        y = np.convolve(psth, wHann, mode="same")
    else :
        y = pair["resp"]["psth_smooth"]

    y_pred = generate_prediction(pair, ridge, feature, laguerre_args, nPoints, nLaguerre)

    return np.sum(y), np.sum(y*y), np.sum(y_pred), np.sum(y_pred*y_pred), np.sum(y*y_pred), len(y_pred)



# Slated for removal
# This function is not used?
def gen_y_avg_laguerre(pair, laguerre_args, nPts):
    x = generate_laguerre_features(
        pair, feature_key="onoff_feature", resp_key='psth_smooth', laguerre_args=laguerre_args, nLaguerrePoints=nPts, nLaguerre=5
    )
    y = pair["resp"]["psth_smooth"]
    ridge = RidgeCV()
    ridge.fit(x.T, y)
    return ridge.predict(x.T)


def gen_y_avg(pair, ridge_conv_filter, nPoints=200, mult_values=False):
    x = arbitrary_kernel(pair, nPoints=nPoints, mult_values=mult_values)
    y_pred = ridge_conv_filter.predict(x.T)
    return y_pred

# end slated for removal


def generate_prediction(
    pair, ridge, feature, basis_args, nPoints=200, nLaguerre=5
):
    if (nLaguerre > 0) :
        x = generate_laguerre_features(
        pair,
        feature_key="pca_%s" % feature,
        resp_key='psth_smooth',
        laguerre_args=basis_args[:,0:2],
        nLaguerrePoints=nPoints,
        nLaguerre=nLaguerre,
        )
    else:
        x = generate_dogs_features(
        pair,
        feature_key="pca_%s" % feature,
        resp_key='psth_smooth',
        dogs_args=basis_args,
        nPoints=nPoints
        )

    y_pred = ridge.predict(x.T)
    y_pred[y_pred < 0] = 0
    return y_pred

def generate_x(pair, feature, basis_args = None, xGen = 'Kernel', nPoints=200, nLaguerre=5):
    # xGen is a string determining which x matrix to generate: Must be "Kernel", "LG", or "DG"
    # "Kernel" is a time-varying impulse function at events for segmentation
    # "LG" is Laguerre Polynomials
    # "DG" is difference of Guassians

    if (xGen == 'Kernel') | (xGen == 'Kernel2') | (xGen == 'Kernel0') :
        x = arbitrary_kernel(
        pair,
        nPoints=nPoints,
        event_key=feature,
        event_index_key="index",
        resp_key="psth",
        mult_values=False,
    )
    elif xGen == 'LG':
        x = generate_laguerre_features(
        pair,
        feature_key='pca_%s'%feature,
        resp_key='psth_smooth',
        laguerre_args=basis_args[:,0:2],
        nLaguerrePoints=nPoints,
        nLaguerre=nLaguerre,
        )
    elif xGen == 'DG':
        x = generate_dogs_features(
        pair,
        feature_key= 'pca_%s'%feature,
        resp_key='psth_smooth',
        dogs_args=basis_args,
        nPoints=nPoints
        )
    else:
        print('Non valid key word for stimulus generation')
        x = None

    return x



def generate_predictionV2 (pair, model, feature, basis_args=None, xGen = 'Kernel', nPoints=200, nLaguerre=5):

    # xGen is the model used for generating the x matrix
    x = generate_x(pair, feature, basis_args = basis_args, xGen = xGen, nPoints=nPoints, nLaguerre=nLaguerre)

    # The Linear model
    y_pred = model['weights']@ x + model['bias']
    # This should give the same result
    # y_pred2 = model['weights']@ (x-model['xavg']) + model['yavg']
                                 
    y_pred[y_pred < 0] = 0
    return y_pred


def generate_pred_score(
    pair, ridge, feature, basis_args, ridge_conv_filter, nPoints=200, nLaguerre=5
):
    if (nLaguerre > 0) :
        x = generate_laguerre_features(
        pair,
        feature_key="pca_%s" % feature,
        resp_key='psth_smooth',
        laguerre_args=basis_args[:,0:2],
        nLaguerrePoints=nPoints,
        nLaguerre=nLaguerre,
        )
    else :
        x = generate_dogs_features(
        pair,
        feature_key="pca_%s" % feature,
        resp_key='psth_smooth',
        dogs_args=basis_args,
        nPoints=nPoints
        )

    y = pair["resp"]["psth_smooth"]
    return ridge.score(x.T, y)


# preproc function


def preprocess_srData(srData, plot=False, respChunkLen=150, segmentBuffer=25, tdelta=0, plotFlg = False, seg_spec_lookup=None, smWindow=31):
    """
    Preprocesses stimulus-response data by segmenting the stimulus based on its envelope, calculating the spectrogram,
    PSTH (Peri-Stimulus Time Histogram), and MPS (Modulation Power Spectrum).
    Parameters:
    srData (dict): Dictionary containing stimulus-response data.
    plot (bool, optional): If True, plots the results. Default is False.
    respChunkLen (int, optional): Total chunk length (including segment buffer) in number of points. Default is 150.
    segmentBuffer (int, optional): Number of points on each side of segment for response and MPS. Default is 25.
    tdelta (int, optional): Time delta to offset the events. Default is 0.
    seg_spec_lookup (dict, optional): Dictionary containing the spectrogram for each stimulus to be used for segmentation
    smWindow (int, optional): Size of the smoothing window used to get a smoothed PSTH
    Returns:
    None: The function modifies the srData dictionary in place, adding preprocessed data to it.
    """
    # PREPROCESSING
    # - Segmentation of the stimulus based on the envelope
    # - Calculation of the spectrogram
    # - Calculation of the smoothed PSTH
    # - Calculation of the MPS

    # Segmentation based on derivative of the envelope
    # ampThresh = 20.0  # Threshold in dB where 50 is max

    minSound = 25  # Minimum distance between peaks or troffs
    derivativeThresh = 0.5  # Threshold derivative 0.5 dB per ms.
    # segmentBuffer = 30 # Number of points on each side of segment for response and MPS - time units given by stim sample rate
    # respChunkLen = 150 # Total chunk length (including segment buffer) in number of points
    # DBNOISE = 50  # Set a baseline for everything below 50 dB from max - done in preprocSound
    
    # Colormap for plotting spectrograms
    spec_colormap()   # defined in sound.py
    cmap = plt.get_cmap('SpectroColorMap')


    wHann = windows.hann(
        smWindow, sym=True
    )  # The 31 ms (number of points) hanning window used to smooth the PSTH
    wHann = wHann / sum(wHann)

    pairCount = len(srData["datasets"])  # number of stim/response pairs

    for iSet in range(pairCount):
        events = dict(
            {
                "index": [],
                "feature": [[]],
            }
        )

        # Stimulus wave file and amplitude enveloppe from spectrogram
        # waveFile = srData['datasets'][iSet]['stim']['rawFile']
        # fs , soundStim = wavfile.read(waveFile)
        # soundLen = soundStim.size
        spectro = np.copy(srData["datasets"][iSet]["stim"]["tfrep"]["spec"])
        # if we have a spec_lookup, we will use it to get the spectrogram
        if seg_spec_lookup is not None:
            seg_spectro = np.asarray(seg_spec_lookup[srData["datasets"][iSet]["stim"]["rawFile"]].data).T
        else:
            seg_spectro = spectro
        
        # This normalization is done at the srData level so that the relative amplitude of the stims is preserved. 
        # dBMax = spectro.max()
        # spectro[spectro < dBMax - DBNOISE] = dBMax - DBNOISE

        # set the y ticks to freq
        nFreqs = len(srData["datasets"][iSet]["stim"]["tfrep"]["f"])

        ampenv = np.mean(seg_spectro, axis=0)
        ampfs = srData["datasets"][iSet]["stim"]["sampleRate"]

        # nSound = int((respChunkLen)*fs/ampfs)   # number of time points in sound chunks - should be the same for all stimulus-response pairs

        ampenv = np.convolve(ampenv, wHann, mode="same")
        ampdev = ampenv[1:] - ampenv[0:-1]

        # Find peaks and troughs
        peakInd, peakVals = find_peaks(
            ampdev, height=derivativeThresh, distance=minSound
        )
        troughInd, troughVals = find_peaks(
            -ampdev, height=derivativeThresh, distance=minSound
        )
        events["index"].extend(peakInd)
        events["onoff_feature"] = np.vstack(
            [np.ones(len(peakInd), dtype=int), np.zeros(len(peakInd), dtype=int)]
        ).T  # these are event 0
        events["index"].extend(troughInd)
        events["onoff_feature"] = np.concatenate(
            [
                events["onoff_feature"],
                np.vstack(
                    [
                        np.zeros(len(troughInd), dtype=int),
                        np.ones(len(troughInd), dtype=int),
                    ]
                ).T,
            ]
        )  # these are event 1
        events["onoff_feature_values"] = np.concatenate(
            [
                peakVals["peak_heights"] / peakVals["peak_heights"].max(),
                troughVals["peak_heights"] / troughVals["peak_heights"].max(),
            ]
        )
        # sort the features by index
        events["onoff_feature"] = events["onoff_feature"][np.argsort(events["index"])]
        events["onoff_feature_values"] = events["onoff_feature_values"][
            np.argsort(events["index"])
        ]
        events["index"] = np.sort(events["index"])

        # now get the features from the spectrogram

        # pad the spectrogram with zeros to make sure we have enough points for windowing
        padded_spect = np.pad(
            spectro,
            ((0, 0), (respChunkLen, respChunkLen)),
            "constant",
            constant_values=(0, 0),
        )
        # now get the sliding window view
        spect_windows = np.lib.stride_tricks.sliding_window_view(
            padded_spect, respChunkLen, axis=1
        )
        # now get the indices of the start of the spectrograms
        # for ON features we take the index of the peak and subtract the segment buffer
        # for OFF features we take the index of the trough and subtract the respChunkLen
        # spect_inds = events['index'] + respChunkLen - segmentBuffer * events['onoff_feature'][:,0] - (respChunkLen - segmentBuffer) * events['onoff_feature'][:,1]
        spect_inds = events["index"] + respChunkLen - segmentBuffer
        events["spect_windows"] = spect_windows[:, spect_inds, :].swapaxes(0, 1)
        events["spect_inds"] = spect_inds

        # now create the features
        events["spect_windows_nfeats"] = (
            events["spect_windows"].shape[1] * events["spect_windows"].shape[2]
        )
        spect_feats = events["spect_windows"].reshape(
            (len(events["index"]), events["spect_windows_nfeats"])
        )

        # now get the features from the log spectrogram
        events["log_spect_windows"] = []
        for i in range(len(events["index"])):
            spect = events["spect_windows"][i]
            log_spect, log_freqs = log_downsample_spec(
                spect, srData["datasets"][iSet]["stim"]["tfrep"]["f"], 18
            )
            events["log_spect_windows"].append(log_spect)
            events["log_spect_freqs"] = log_freqs
        events["log_spect_windows"] = np.asarray(events["log_spect_windows"])
        events["log_spect_windows_nfeats"] = (
            events["log_spect_windows"].shape[1] * events["log_spect_windows"].shape[2]
        )

        # and finally create features for the mps
        events["mps_windows"] = []
        for i in range(len(events["index"])):
            spect = events["spect_windows"][i]
            window = 0.05
            Norm = True
            wf, wt, mps_powAvg = mps(
                spect,
                srData["datasets"][iSet]["stim"]["tfrep"]["f"],
                np.arange(respChunkLen) / ampfs,
                window=window,
                Norm=Norm,
            )

            indwt = np.argwhere((wt > -100) & (wt < 100))
            indwf = np.argwhere((wf >= 0) & (wf < 6e-3))
            mps_powAvg = mps_powAvg[indwf[:, 0], :][:, indwt[:, 0]]

            events["mps_windows"].append(mps_powAvg)
            events["mps_windows_freqs"] = wf[indwf[:, 0]]
            events["mps_windows_times"] = wt[indwt[:, 0]]
        events["mps_windows"] = np.asarray(events["mps_windows"])
        events["mps_windows_nfeats"] = (
            events["mps_windows"].shape[1] * events["mps_windows"].shape[2]
        )

        # offset the events by tdelta
        events["index"] = events["index"] + tdelta
        events["index"][events["index"] < 0] = 0

        # now get the response
        srData["datasets"][iSet]["events"] = events

        # smooth the psth
        if "resp" in srData["datasets"][iSet]:
            srData["datasets"][iSet]["resp"]["psth_smooth"] = np.convolve(
                srData["datasets"][iSet]["resp"]["psth"], wHann, mode="same"
            )

        if plotFlg:
            plt.figure(figsize=(8, 2), dpi=100)
            # plt.plot(ampdev)

            # plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(spectro, aspect="auto", cmap=cmap, origin="lower")
            

            for soundStart in peakInd:
                soundFinish = np.argwhere(troughInd >= soundStart + minSound)
                if soundFinish.shape[0]:
                    dt = troughInd[soundFinish[0][0]] - soundStart
                    # print (soundStart, soundEnd[soundFinish[0][0]], dt)
                    if dt < minSound:
                        print(
                            "WARNING - not expecting short sound here",
                            int(soundStart),
                            int(troughInd[soundFinish[0][0]] + 1),
                        )

            for soundStart in peakInd:
                plt.plot([soundStart, soundStart], [0, np.max(ampdev) * 1.1], "r")

            for soundEnd in troughInd:
                plt.plot([soundEnd, soundEnd], [0, -np.max(ampdev) * 1.1], "b")
            xlim = plt.xlim()
            plt.subplot(2, 1, 2)
            plt.plot(ampdev)
            plt.axhline(derivativeThresh, color="k")
            plt.axhline(-derivativeThresh, color="k")
            plt.ylim((derivativeThresh*-3.0, derivativeThresh*3.0))
            plt.xlim(xlim)

            plt.figure(figsize=(16, 2), dpi=100)
            nEvents = events["index"].shape[0]
            nOn = sum(events['onoff_feature'][:,0] == 1 )
            nOff = sum(events['onoff_feature'][:,0] == 0 )
            nMax = max((nOn, nOff))
            iOn = 1
            iOff = nMax+1

            for iEvent in range(nEvents):
                if (events['onoff_feature'][iEvent,0] == 1):
                    plt.subplot(2, nMax, iOn)
                    iOn += 1
                else:
                    plt.subplot(2, nMax, iOff)
                    iOff += 1
                
                plt.imshow(events['spect_windows'][iEvent,:,:], aspect="auto", cmap=cmap, origin="lower")
                plt.axis('off')


    # lets generate zero-mean mps
    all_mps = np.concatenate(
        [
            srData["datasets"][iSet]["events"]["mps_windows"]
            for iSet in range(pairCount)
        ],
        axis=0,
    )
    mean_mps = np.mean(all_mps, axis=0)
    for iSet in range(pairCount):
        srData["datasets"][iSet]["events"]["mps_windows"] = (
            srData["datasets"][iSet]["events"]["mps_windows"] - mean_mps
        )

from numba import njit

@njit
def nearDiagInv_optim(diagA, u, s, v, tol=0.0):
    """
    Fast Sherman-Morrison-based inverse of near-diagonal matrix using Numba.

    Parameters:
        diagA : (n, n) ndarray
            Diagonal matrix (full 2D form, assumed to be diagonal).
        u : (n, k) ndarray
            Left singular vectors.
        s : (k,) ndarray
            Singular values.
        v : (k, n) ndarray
            Right singular vectors.
        tol : float
            Regularization term for stability.

    Returns:
        Ainv : (n, n) ndarray
            Approximate inverse of A = diagA + u @ diag(s) @ v.
    """
    n = diagA.shape[0]
    k = s.shape[0]

    # Precompute D⁻¹
    Ainv = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        Ainv[i, i] = 1.0 / diagA[i, i]

    # uc[:, i] = s[i] * u[:, i]
    for i in range(k):
        weight = s[i] / (s[i] + tol)
        ue = np.zeros((n, 1))
        ve = np.zeros((1, n))

        for j in range(n):
            ue[j, 0] = s[i] * u[j, i]
            ve[0, j] = v[i, j]

        # denom = 1 + ve @ Ainv @ ue
        temp1 = np.zeros((1, 1))
        for a in range(n):
            for b in range(n):
                temp1[0, 0] += ve[0, a] * Ainv[a, b] * ue[b, 0]

        denom = 1.0 + temp1[0, 0]

        # Only apply update if denom is not zero
        if denom != 0.0:
            # temp2 = Ainv @ ue
            temp2 = np.zeros((n, 1))
            for a in range(n):
                for b in range(n):
                    temp2[a, 0] += Ainv[a, b] * ue[b, 0]

            # temp3 = ve @ Ainv
            temp3 = np.zeros((1, n))
            for a in range(n):
                for b in range(n):
                    temp3[0, b] += ve[0, a] * Ainv[a, b]

            # Ainv -= weight * (temp2 @ temp3) / denom
            for a in range(n):
                for b in range(n):
                    Ainv[a, b] -= weight * temp2[a, 0] * temp3[0, b] / denom

    return Ainv

def nearDiagInv(diagA, u, s, v, tol=0):
    ''' Calculates the regularized inverse of a near diagonal matrix, decomposed into its diagonal element and an SVD of its off-diagonal terms
    diag A: The diagonal componnent of the diagonal matrix in its full matrix form
    u, s, v: The singular value decomposition of the offdiagonal terms
    tol: The regularization parameter
    '''
    
    # The dimensionality of square matrices - could check for dimensions here
    nDim = diagA.shape[0]

    # Initialize the inverse
    Ainv = np.diag(1/np.diag(diagA))
    uc = u@np.diag((s))

    # Calculate the inverse by iteration using the Sherman-Morisson formula
    for ie in range(nDim):
        ue = np.reshape(uc[:,ie], (uc.shape[0], 1))
        ve = np.reshape(v[ie,:], (1, v.shape[1]))
        Ainv = Ainv - (s[ie]/(s[ie]+tol)) * (Ainv @ ue @ ve @ Ainv)/(1+ve@Ainv@ue)

    return Ainv

def nearDiagInv2(Cxx, diagCxx, tol=0):
    '''Alternative to neadDiagInv2 that interporlates between the inverse of Cxx and of diagCxx'''

    Ainv = np.linalg.inv(Cxx*(1/(1+tol)) + diagCxx*(tol/(1+tol)))
    return Ainv

def generate_event_pca_feature(srData, event_types, feature, pca = None, npcs=20):

    # first use PCA to reduce dim of the features'
    nEventTypes = srData["datasets"][0]["events"][event_types].shape[1]
    nfeats = srData["datasets"][0]["events"]["%s_nfeats" % feature]
    pairCount = len(srData["datasets"])

    # FIt the pca on the features - notice that this is not weighted.
    if (pca == None) :
        all_spect_windows = np.concatenate(
        [
            np.asarray(srData["datasets"][iSet]["events"][feature]).reshape(
                (len(srData["datasets"][iSet]["events"]["index"]), nfeats)
            )
         for iSet in range(pairCount)
        ],
        axis=0,
        )
        pca = PCA(n_components=npcs)
        pca.fit(all_spect_windows)



    # Calculate and store the PC coefficients
    for iSet in range(pairCount):
        events = srData["datasets"][iSet]["events"][event_types]
        n_events = len(srData["datasets"][iSet]["events"]["index"])
        spect_pca_features = pca.transform(
            srData["datasets"][iSet]["events"][feature].reshape((n_events, nfeats))
        )

        srData["datasets"][iSet]["events"]["pca_%s" % feature] = np.zeros(
            (n_events, nEventTypes * npcs)
        )

        for iEventType in range(events.shape[1]):
            srData["datasets"][iSet]["events"]["pca_%s" % feature][
                events[:, iEventType] == 1, iEventType * npcs : (iEventType + 1) * npcs
            ] = spect_pca_features[events[:, iEventType] == 1, :]

    return pca
    

# fitting funcitons
def fit_seg(
    srData, nPoints, x_feature, y_feature = 'psth_smooth', y_R2feature = None, kernel = 'Kernel', basis_args = [], nD=2, pair_train_set=None, tol = np.array([0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001, 0]),
store_error = False):
    """
    Fits a segmented model to the given data using ridge regresseion and leave one out cross-validation
    Parameters:
    srData (dict): The dataset containing events and responses.
    nPoints (int): The number of points for the convolutional Kernel.
    event_types (str): The type of events that define the segmentation
    feature (str): The feature to use. If a pca is used this field will be pca_feature
    kernel (str): a string specifying the kernel type.  Options are 'Kernel', 'LG', 'DG'.
                'Kernel' : arbitrary impulse function ('Kernel2' uses a different reg, 'Kernel0' is straight ridge)
                'DG' : difference of gaussians convolutional kernel
                'LG' : Laguerre polynomials convolutional kernel
    basis_args: arguments for the DG or LG kernels
    nD (int): The number of kernel functions for the convolutional models. 2 is the number for 'kernel', 5 is the number for 'DG' and 20 is a good option for 'LG' kernel.
    pair_train_set (list): List of dataset indices to use for training.
    tol: the ridge hyperparameter expressed as a scale of the stimulus auto-correlation 
    Returns:
        segMpdel: The ridge regression model for segmentation/identification - currently a dictionary. To be maade into a class
    """


    # If a training subset is not defined use the entire data set
    if pair_train_set is None:
        nSets = len(srData["datasets"])
        pair_train_set = np.arange(nSets)
    else:
        nSets = len(pair_train_set) 

    # Find the dimensionality of the x feature space.  For kernel it is just the number of points
    if (kernel == 'Kernel') | (kernel =='Kernel2') | (kernel == 'Kernel0'):
        nFeatures = nPoints
    else:
        pair = srData["datasets"][pair_train_set[0]]
        x = pair["events"]["pca_%s" % x_feature]
        if x.ndim == 1:
            x = x[:, np.newaxis]
        nFeatures = x.shape[1]
  
    # 1. Calculate the averages to zero out data
    all_x = []
    all_y = []
    all_yw = []

    xavg = np.zeros((nSets,nD*nFeatures, 1))
    count = np.zeros(nSets)
    yavg = np.zeros(nSets)

    # We are not going to store the feature matrix, x, but recalculate them to save RAM
    for iS, iSet in enumerate(pair_train_set):
        # Get the x and y
        pair = srData["datasets"][iSet]
        x = generate_x(pair, x_feature, basis_args = basis_args, xGen = kernel, nPoints=nPoints, nLaguerre=nD)

        y = pair["resp"][y_feature]  
        if "weights" not in pair["resp"]:
            yw = np.ones_like(y)
        else:
            yw = pair["resp"]["weights"][0:len(y)]   # The processed fatures might be shorter

        # Eliminate entres with zero weight - this is not needed but should make smaller x and y
        x = x[:, 0:len(y)]
        x = x[:, yw> 0]
        y = y[yw > 0]
        yw = yw[yw > 0]

        all_x.append(x)
        all_y.append(y)
        all_yw.append(yw)


        xavg[iS, :, :] = np.sum(x*yw.T, axis=1, keepdims=True)
        count[iS] = np.sum(yw)
        yavg[iS] = np.sum(y*yw)

    # 2 Calculate the leave one out average X and average Y
    xsumAll = np.sum(xavg, axis=0, keepdims=True)
    countAll = np.sum(count)
    ysumAll = np.sum(yavg)
    for iS in range(nSets):
        xavg[iS,:,:] = (xsumAll - xavg[iS,:,:])/(countAll - count[iS])
        yavg[iS] = (ysumAll - yavg[iS])/(countAll - count[iS])

     
    # 3 Calculate the leave one out auto-covariance (XX.t) and cross-covariance (XY), and the norm of XX.t
    Cxx = np.zeros((nSets,nD*nFeatures,nD*nFeatures))
    Cxy = np.zeros((nSets,nD*nFeatures))

    for iS, iSet in enumerate(pair_train_set):
        # pair = srData["datasets"][iSet] 
        # x = generate_x(pair, x_feature, basis_args = basis_args, xGen = kernel, nPoints=nPoints, nLaguerre=nD)      
        # y = pair["resp"][y_feature]  
        # if "weights" not in pair["resp"]:
        #     yw = np.ones_like(y)
        # else:
        #     yw = pair["resp"]["weights"][0:len(y)]

        # x = x[:, 0:len(y)]
        # x = x[:, yw> 0]
        # y = y[yw > 0]
        # yw = yw[yw>0]
        x = all_x[iS]
        y = all_y[iS]
        yw = all_yw[iS]
        
        # Auto-Covariance and Cross-Covariances matrices, the square roots multiply to give a weight to the squares
        Cxx[iS,:,:] = ((x-xavg[iS,:,:])*(np.sqrt(yw)).T) @ ((x-xavg[iS,:,:])*(np.sqrt(yw)).T).T
        # Cxy[iS,:] = ((x-xavg[iS,:,:])*(np.sqrt(yw)).T) @ ((y-yavg[iS])*(np.sqrt(yw)))
        Cxy[iS,:] = (x-xavg[iS,:,:]) @ ((y-yavg[iS])*yw)  # This matches how we do strfs

    CxxAll = np.sum(Cxx, axis=0)
    CxyAll = np.sum(Cxy, axis=0)
    CxxNorm = np.zeros(nSets)
    
    for iS in range(nSets):
        Cxx[iS,:,:] = (CxxAll - Cxx[iS,:,:])/(countAll - count[iS])
        Cxy[iS,:] = (CxyAll - Cxy[iS,:])/(countAll - count[iS])
        if ( (kernel == 'Kernel') | (kernel == 'Kernel2') ):
            CxxNorm[iS] = np.linalg.norm(np.squeeze(Cxx[iS, :, :]-np.diag(np.diag(np.squeeze(Cxx[iS, :, :])))))
        else:
            CxxNorm[iS] = np.linalg.norm(np.squeeze(Cxx[iS, :, :]))
            

    ranktol = tol * np.max(CxxNorm)

    # 4. Calculate the ridge regression by hand.

    # 4a. Invert all auto-correlation matrices
    u = np.zeros(Cxx.shape)
    v = np.zeros(Cxx.shape)
    s = np.zeros(Cxy.shape)     # This is just the diagonal
    hJN = np.zeros(Cxy.shape)
    nb = Cxx.shape[1]

    if ( kernel == 'Kernel' ):
        for iS in range(nSets):
            diagCxx = np.diag(np.diag(np.squeeze(Cxx[iS,:,:])))
            u[iS,:,:],s[iS,:],v[iS,:,:] = np.linalg.svd(Cxx[iS,:,:]-diagCxx)
    else:
        for iS in range(nSets):
            u[iS,:,:],s[iS,:],v[iS,:,:] = np.linalg.svd(Cxx[iS,:,:])

    R2CV = np.zeros(ranktol.shape[0])

    for it, tolval in enumerate(ranktol):

        simple_sum_yy = 0
        simple_sum_y =  0
        simple_sum_error = 0
        simple_sum_count = 0

        for iS, iSet in enumerate(pair_train_set):
            
            if (kernel == 'Kernel'):
                diagCxx = np.diag(np.diag(np.squeeze(Cxx[iS,:,:])))
                Cxx_inv = nearDiagInv_optim(diagCxx, u[iS,:,:], s[iS,:], v[iS,:,:], tol=tolval)
                hJN[iS,:] = Cxx_inv @ Cxy[iS,:]
            elif (kernel == 'Kernel2'):
                diagCxx = np.diag(np.diag(np.squeeze(Cxx[iS,:,:])))
                Cxx_inv = nearDiagInv2(np.squeeze(Cxx[iS,:,:]), diagCxx,  tol=tolval)
                hJN[iS,:] = Cxx_inv @ Cxy[iS,:]
            else:
                is_mat = np.zeros((nb, nb))
                # ridge regression - regularized normal equation
                for ii in range(nb):
                    is_mat[ii,ii] = 1.0/(s[iS, ii] + tolval)       
                hJN[iS,:] = v[iS, :, :].T @ is_mat @ (u[iS, :, :].T @ Cxy[iS,:])

            # Find x and actual y for leave out set to asses fit
            pair = srData["datasets"][iSet]

            # x = generate_x(pair, x_feature, basis_args = basis_args, xGen = kernel, nPoints=nPoints, nLaguerre=nD)      
            # y = pair["resp"][y_feature]  
            # if "weights" not in pair["resp"]:
            #     yw = np.ones_like(y)
            # else:
            #     yw = pair["resp"]["weights"][0:len(y)]

            # x = x[:, 0:len(y)]
            # x = x[:, yw> 0]
            # y = y[yw > 0]
            # yw = yw[yw >0]

            x = all_x[iS]
            y = all_y[iS]
            yw = all_yw[iS]
    
            # Get the prediciton
            ypred = hJN[iS, :]@ (x - xavg[iS]) + yavg[iS]

            # Store it if asked
            if (store_error):
                pair['resp']['error_%s' % kernel] = y - ypred

            if (y_R2feature is None):
                yr2 = y
            else:
                yr2 = pair['resp'][y_R2feature][0:len(y)]
                yr2 = yr2[yw>0]
                ypred += (yr2 - y)

            # Rectify - this should be a flag
            ypred[ypred<0] = 0

            # Get values to calculate R2-CV - here it is the coefficient of determination
            sum_count = np.sum(yw)
            sum_y = np.sum(yr2*yw)
            sum_yy = np.sum(yr2*yr2*yw)
            sum_error2 = np.sum(((ypred-yr2)**2)*yw)

            simple_sum_yy += sum_yy
            simple_sum_y +=  sum_y
            simple_sum_error += sum_error2
            simple_sum_count += sum_count
        

        y_mean = simple_sum_y/simple_sum_count
        y_var = simple_sum_yy/simple_sum_count - y_mean**2
        y_error = simple_sum_error/simple_sum_count

        # This is not a "one-trial" CV
        R2CV[it] = 1.0 - y_error/y_var
     
    # Find the best tolerance level, i.e. the ridge penalty hyper-parameter
    segModel = {}
    itMax = np.argmax(R2CV)
    segModel['Tol'] = tol
    segModel['R2CV'] = R2CV
    if ( (itMax == 0) | (itMax == len(tol)-1 )):
        print('fit_seg() warning: Max prediction found for %f. Extend range of tolerance values' % tol[itMax])

    # Calculate one filter at the best tolerance
    CxxAll /= countAll
    CxyAll /= countAll
    if (kernel == 'Kernel'):
        diagCxx = np.diag(np.diag(np.squeeze(CxxAll)))
        uAll,sAll,vAll = np.linalg.svd(CxxAll-diagCxx)
        CxxAll_inv = nearDiagInv_optim(diagCxx, uAll, sAll, vAll, tol=ranktol[itMax])
        hJNAll = CxxAll_inv @ CxyAll
    elif (kernel == 'Kernel2'):
        diagCxx = np.diag(np.diag(np.squeeze(CxxAll)))
        CxxAll_inv = nearDiagInv2(np.squeeze(CxxAll), diagCxx, tol=ranktol[itMax])
        hJNAll = CxxAll_inv @ CxyAll
    else:
        uAll,sAll,vAll = np.linalg.svd(CxxAll)
        for ii in range(nb):
            is_mat[ii,ii] = 1.0/(sAll[ii] + ranktol[itMax])
        hJNAll = vAll.T @ is_mat @ (uAll.T @ CxyAll)

    # The bias term
    b0 = -hJNAll @ (xsumAll/countAll) + (ysumAll/countAll)
    segModel['weights'] = hJNAll
    segModel['b0'] = b0[0,0]
    segModel['yavg'] = ysumAll/countAll
    segModel['xavg'] = xsumAll/countAll
    segModel['Cxy'] = CxyAll
    segModel['Cxx'] = CxxAll


    # 6. Return segmented model
    return segModel

   

def fit_seg_segId_model(
    srData, nLaguerre, nPoints, event_types, feature, pair_train_set=None, pca=None, tol = np.array([0.6, 0.5, 0.4, 0.2, 0.15, 0.100, 0.08, 0.050, 0.010, 0.005, 1e-03])
):
    """
    Fits a segmented model to the given data.
    Parameters:
    srData (dict): The dataset containing events and responses.
    nLaguerre (int): The number of Laguerre functions to use. If equal to -1 DOGS fits to segmented kernels and their derivatives are used.
    nPoints (int): The number of points for the convolutional kernel.
    event_types (str): The type of events to consider.
    feature (str): The feature to use for PCA.
    pair_train_set (list): List of dataset indices to use for training.
    pca (PCA, optional): Pre-fitted PCA object. If None, a new PCA will be fitted. Default is None.
    Returns:
    tuple: A tuple containing:
        - pca (PCA): The fitted PCA object for the input features
        - segIDModel: The ridge regression model for segmentation/identification - currently a dictionary. To be maade into a class
        - ridgeS: The ridge regression model for the segmentation only model - currently a dictionary.
        - basis_args (ndarray): The fitted parameters for the basis set used in the SI model - either Laguerre or DOGS.
    """
    nEventTypes = srData["datasets"][0]["events"][event_types].shape[1]
    nfeats = srData["datasets"][0]["events"]["%s_nfeats" % feature]
    pairCount = len(srData["datasets"])


    if pair_train_set is None:
        pair_train_set = np.arange(pairCount)

    nSets = len(pair_train_set) 


    # 1. first use PCA to reduce dim of the features'
    print("Fitting PCA to Feature")
    if pca is None:
        npcs = 20
        all_spect_windows = np.concatenate(
            [
                np.asarray(srData["datasets"][iSet]["events"][feature]).reshape(
                    (len(srData["datasets"][iSet]["events"]["index"]), nfeats)
                )
                for iSet in range(pairCount)
            ],
            axis=0,
        )
        pca = PCA(n_components=npcs)
        pca.fit(all_spect_windows)
            # Clear some memory
        all_spect_windows = None
    else:
        npcs = pca.n_components_

    # This pca transform might not be needed?  Maybe it is stored? or it could be done on the fly?
    for iSet in range(pairCount):
        events = srData["datasets"][iSet]["events"][event_types]
        n_events = len(srData["datasets"][iSet]["events"]["index"])
        spect_pca_features = pca.transform(
            srData["datasets"][iSet]["events"][feature].reshape((n_events, nfeats))
        )
        srData["datasets"][iSet]["events"]["pca_%s" % feature] = np.zeros(
            (n_events, nEventTypes * npcs)
        )
        for iEventType in range(events.shape[1]):
            srData["datasets"][iSet]["events"]["pca_%s" % feature][
                events[:, iEventType] == 1, iEventType * npcs : (iEventType + 1) * npcs
            ] = spect_pca_features[events[:, iEventType] == 1, :]
    


    # 2. now fit the onsets with a convolutional kernel
    #       This removes the average response to onsets and offsets
    print("Fitting convolutional kernel")
    xavg = np.zeros((nSets,nPoints*2, 1))
    count = np.zeros(nSets)
    yavg = np.zeros(nSets)

    for iS, iSet in enumerate(pair_train_set):
        pair = srData["datasets"][iSet]
        x = generate_x(pair, feature = 'onoff_feature', xGen = 'Kernel', nPoints=nPoints)

        if "weights" not in pair["resp"]:
            yw = np.ones_like(pair["resp"]["psth_smooth"])
        else:
            yw = pair["resp"]["weights"]
        y = pair["resp"]["psth_smooth"]
        x = x[:, yw > 0]
        y = y[yw > 0]
        yw = yw[yw > 0]

        xavg[iS, :, :] = np.sum(x*yw.T, axis=1, keepdims=True)
        count[iS] = np.sum(yw)
        yavg[iS] = np.sum(y*yw)

    # 2b. Calculate the leave one out average stimulus and average response
    xsumAll = np.sum(xavg, axis=0, keepdims=True)
    countAll = np.sum(count)
    ysumAll = np.sum(yavg)

    for iS in range(nSets):
        xavg[iS,:,:] = (xsumAll - xavg[iS,:,:])/(countAll - count[iS])
        yavg[iS] = (ysumAll - yavg[iS])/(countAll - count[iS])

     
    # 2c. Calculate auto-covariance and cross-covariance
    
    Cxx = np.zeros((nSets,nPoints*2,nPoints*2))
    Cxy = np.zeros((nSets,nPoints*2))

    for iS, iSet in enumerate(pair_train_set):
        pair = srData["datasets"][iSet]
        x = generate_x(pair, feature = 'onoff_feature', xGen = 'Kernel', nPoints=nPoints)
        y = pair["resp"]["psth_smooth"]  
        if "weights" not in pair["resp"]:
            yw = np.ones_like(y)
        else:
            yw = pair["resp"]["weights"]

        x = x[:, yw > 0]
        y = y[yw > 0]
        yw = yw[yw > 0]

        # Auto-Covariance and Cross-Covariances matrices, the square roots multiply to give a weight to the squares
        Cxx[iS,:,:] = ((x-xavg[iS,:,:])*(np.sqrt(yw[yw>0])).T) @ ((x-xavg[iS,:,:])*(np.sqrt(yw[yw>0])).T).T
        Cxy[iS,:] = ((x-xavg[iS,:,:])*(np.sqrt(yw[yw>0])).T) @ ((y-yavg[iS])*(np.sqrt(yw[yw>0])))

    # 2d. Calculate the leave one out covariances and the norm of CXX
    CxxAll = np.sum(Cxx, axis=0)
    CxyAll = np.sum(Cxy, axis=0)
    CxxNorm = np.zeros(nSets)
    
    for iS in range(nSets):
        Cxx[iS,:,:] = (CxxAll - Cxx[iS,:,:])/(countAll - count[iS])
        Cxy[iS,:] = (CxyAll - Cxy[iS,:])/(countAll - count[iS])
        CxxNorm[iS] = np.linalg.norm(np.squeeze(Cxx[iS, :, :]))

    ranktol = tol * np.max(CxxNorm)

    # 3. Calculate the ridge regression by hand.

    # 3a. Invert all auto-correlation matrices
    u = np.zeros(Cxx.shape)
    v = np.zeros(Cxx.shape)
    s = np.zeros(Cxy.shape)     # This is just the diagonal
    hJN = np.zeros(Cxy.shape)
    nb = Cxx.shape[1]

    for iS in range(nSets):
        u[iS,:,:],s[iS,:],v[iS,:,:] = np.linalg.svd(Cxx[iS,:,:])

    R2CV = np.zeros(ranktol.shape[0])

    for it, tolval in enumerate(ranktol):

        simple_sum_yy = 0
        simple_sum_y =  0
        simple_sum_error = 0
        simple_sum_count = 0

        for iS, iSet in enumerate(pair_train_set):

            is_mat = np.zeros((nb, nb))
            # ridge regression - regularized normal equation
            for ii in range(nb):
                is_mat[ii,ii] = 1.0/(s[iS, ii] + tolval)
        
            hJN[iS,:] = v[iS, :, :] @ is_mat @ (u[iS, :, :] @ Cxy[iS,:])

            # Find x and actual y:
            pair = srData["datasets"][iSet]
            x = generate_x(pair, feature = 'onoff_feature', xGen = 'Kernel', nPoints=nPoints)

            y = pair["resp"]["psth_smooth"] 
            if "weights" not in pair["resp"]:
                yw = np.ones_like(y)
            else:
                yw = pair["resp"]["weights"]

            x = x[:, yw > 0]
            y = y[yw > 0]
            yw = yw[yw > 0]
          
            # Get the prediciton
            ypred = hJN[iS, :]@ (x - xavg[iS]) + yavg[iS]

            # Get values to calculate R2-CV - here it is the coefficient of determination
            sum_count = np.sum(yw)
            sum_y = np.sum(y*yw)
            sum_yy = np.sum(y*y*yw)
            sum_error2 = np.sum(((ypred-y)**2)*yw)

            simple_sum_yy += sum_yy
            simple_sum_y +=  sum_y
            simple_sum_error += sum_error2
            simple_sum_count += sum_count
        

        y_mean = simple_sum_y/simple_sum_count
        y_var = simple_sum_yy/simple_sum_count - y_mean**2
        y_error = simple_sum_error/simple_sum_count

        # This is a "one-trial" CV
        R2CV[it] = 1.0 - y_error/y_var
    
    # Find the best tolerance level, i.e. the ridge penalty hyper-parameter
    segModel = {}
    itMax = np.argmax(R2CV)
    segModel['Tolerances'] = tol
    segModel['BestITolInd'] = itMax
    segModel['R2CV'] = R2CV

    # Calculate one filter at the best tolerance
    uAll,sAll,vAll = np.linalg.svd(CxxAll/countAll)
    
    is_mat = np.zeros((nb, nb))
    for ii in range(nb):
        is_mat[ii,ii] = 1.0/(sAll[ii] + ranktol[itMax])
    
    hJNAll = vAll @ is_mat @ (uAll @ (CxyAll/countAll))
    # The bias term
    b0 = -hJNAll @ (xsumAll/countAll) + (ysumAll/countAll)
    segModel['weights'] = hJNAll
    segModel['bias'] = b0[0,0]
    segModel['xavg'] = xsumAll/countAll
    segModel['yavg'] = ysumAll/countAll
    learned_conv_kernel = hJNAll.reshape(2, nPoints)

    # This is how we used to do it, using the sicikit learn ridge
    #SegModel_ridge = RidgeCV()
    #SegModel_ridge.fit(X.T, Y, sample_weight=Y_weights)
    # learned_conv_kernel = SegModel_ridge.coef_.reshape(2, nPoints)



    # 3. now fit the laguerre parameters to the convolutional kernel
    print("Fitting laguerre or DOGS parameters using Ridge")

    def sum_n_laguerres(xt, *args):
        tau, alpha, *w = args
        nL = len(w)
        out = np.zeros_like(xt, dtype=float)
        for iL in range(nL):
            out += w[iL] * laguerre(xt, 1.0, tau, alpha, xorder=iL) 
        return out

    if ( nLaguerre > 0 ):
        print("Fitting Laguerre parameters")
        basis_args = np.zeros((nEventTypes, 7))
        for iEventType in range(nEventTypes):
            popt, pcov = curve_fit(
                sum_n_laguerres,
                np.arange(nPoints),
                learned_conv_kernel[iEventType, :]-np.mean(learned_conv_kernel[iEventType, :]),
                p0=[15, 5, 1, 1, 1, 1, 1],
                bounds=(
                [0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                ),
                method="trf",
            )
            basis_args[iEventType, :] = popt
    else:
        print("Fitting DOGS parameters")
        basis_args = np.zeros((nEventTypes, 7))

        # Find starting points - the bound might be too restrictive...
        for iEventType in range(nEventTypes):
            meanVal = np.mean(learned_conv_kernel[iEventType, :])
            ampPos = np.max(learned_conv_kernel[iEventType, :])-meanVal
            tPos = np.argmax(learned_conv_kernel[iEventType, :])
            ampNeg = np.abs(np.min(learned_conv_kernel[iEventType, :])-meanVal)
            tNeg = np.argmin(learned_conv_kernel[iEventType, :])
            sdPos = sdNeg = 20
            p0=[meanVal, ampPos, ampNeg, tPos, tNeg, sdPos, sdNeg]
            popt, pcov = curve_fit(
                dogs,
                np.arange(nPoints),
                learned_conv_kernel[iEventType, :]-np.mean(learned_conv_kernel[iEventType, :]),
                p0=p0,
                bounds=(
                [np.min(learned_conv_kernel[iEventType, :]), 0, 0, 0, 0, 0, 0],
                [np.max(learned_conv_kernel[iEventType, :]), np.inf, np.inf, nPoints, nPoints, nPoints, nPoints],
            ),
            method="trf",
            )
            basis_args[iEventType, :] = popt

    # 4. now fit the response to onsets and offsets using the features

    # 4a. First calculate the weighted sums per pair_train_set to get average response and stim:
    # Obrain the size of the feature space
    feature_key= "pca_%s" % feature
    pair = srData["datasets"][pair_train_set[0]]
    x = pair["events"][feature_key]
    if x.ndim == 1:
        x = x[:, np.newaxis]
    nFeatures = x.shape[1]

    nDOGS = 5
    nSets = len(pair_train_set) 
    if (nLaguerre > 0 ):
        nD = nLaguerre
    else:
        nD = nDOGS

  
    xavg = np.zeros((nSets,nD*nFeatures, 1))

    count = np.zeros(nSets)
    yavg = np.zeros(nSets)

    # We are not going to store the feature matrix, x, but recalculate them to save RAM
    for iS, iSet in enumerate(pair_train_set):
        # Get the x and y
        pair = srData["datasets"][iSet]
        if (nLaguerre > 0 ):
            x = generate_x(pair, feature, basis_args = basis_args, xGen = 'LG', nPoints=nPoints, nLaguerre=nLaguerre)
        else:
            x = generate_x(pair, feature, basis_args = basis_args, xGen = 'DG', nPoints=nPoints)

        y = pair["resp"]["psth_smooth"]  
        if "weights" not in pair["resp"]:
            yw = np.ones_like(y)
        else:
            yw = pair["resp"]["weights"]

        # Eliminate stimulus with no data - this is not needed but should make smaller x and y
        x = x[:, yw > 0]
        y = y[yw > 0]
        

        # subract result from segmented model
        xs = generate_x(pair, feature = 'onoff_feature', xGen = 'Kernel', nPoints=nPoints)
        ys_pred = segModel['weights']@ xs[:, yw > 0] + segModel['bias']
        y = y-ys_pred
        yw = yw[yw > 0]

        xavg[iS, :, :] = np.sum(x*yw.T, axis=1, keepdims=True)
        count[iS] = np.sum(yw)
        yavg[iS] = np.sum(y*yw)

    # 4b. Calculate the leave one out average stimulus and average response
    xsumAll = np.sum(xavg, axis=0, keepdims=True)
    countAll = np.sum(count)
    ysumAll = np.sum(yavg)
    for iS in range(nSets):
        xavg[iS,:,:] = (xsumAll - xavg[iS,:,:])/(countAll - count[iS])
        yavg[iS] = (ysumAll - yavg[iS])/(countAll - count[iS])

     
    # 4c. Calculate auto-covariance and cross-covariance
    nSets = len(pair_train_set) 
    Cxx = np.zeros((nSets,nD*nFeatures,nD*nFeatures))
    Cxy = np.zeros((nSets,nD*nFeatures))

    for iS, iSet in enumerate(pair_train_set):
        pair = srData["datasets"][iSet]       
        if (nLaguerre > 0 ):
            x = generate_x(pair, feature, basis_args = basis_args, xGen = 'LG', nPoints=nPoints, nLaguerre=nLaguerre)
        else:
            x = generate_x(pair, feature, basis_args = basis_args, xGen = 'DG', nPoints=nPoints)

        y = pair["resp"]["psth_smooth"]  
        if "weights" not in pair["resp"]:
            yw = np.ones_like(y)
        else:
            yw = pair["resp"]["weights"]

        x = x[:, yw > 0]
        y = y[yw > 0]
        

        # subract result from segmented model
        xs = generate_x(pair, feature = 'onoff_feature', xGen = 'Kernel', nPoints=nPoints)
        ys_pred = segModel['weights']@ xs[:, yw>0] + segModel['bias']
        y = y-ys_pred
        yw = yw[yw > 0]

        # Auto-Covariance and Cross-Covariances matrices, the square roots multiply to give a weight to the squares
        Cxx[iS,:,:] = ((x-xavg[iS,:,:])*(np.sqrt(yw[yw>0])).T) @ ((x-xavg[iS,:,:])*(np.sqrt(yw[yw>0])).T).T
        Cxy[iS,:] = ((x-xavg[iS,:,:])*(np.sqrt(yw[yw>0])).T) @ ((y-yavg[iS])*(np.sqrt(yw[yw>0])))

    # 4d. Calculate the leave one out covariances and the norm of CXX
    CxxAll = np.sum(Cxx, axis=0)
    CxyAll = np.sum(Cxy, axis=0)
    CxxNorm = np.zeros(nSets)
    
    for iS in range(nSets):
        Cxx[iS,:,:] = (CxxAll - Cxx[iS,:,:])/(countAll - count[iS])
        Cxy[iS,:] = (CxyAll - Cxy[iS,:])/(countAll - count[iS])
        CxxNorm[iS] = np.linalg.norm(np.squeeze(Cxx[iS, :, :]))

    ranktol = tol * np.max(CxxNorm)

    # 5. Calculate the ridge regression by hand.

    # 5a. Invert all auto-correlation matrices
    u = np.zeros(Cxx.shape)
    v = np.zeros(Cxx.shape)
    s = np.zeros(Cxy.shape)     # This is just the diagonal
    hJN = np.zeros(Cxy.shape)
    nb = Cxx.shape[1]

    for iS in range(nSets):
        u[iS,:,:],s[iS,:],v[iS,:,:] = np.linalg.svd(Cxx[iS,:,:])

    R2CV = np.zeros(ranktol.shape[0])

    for it, tolval in enumerate(ranktol):

        simple_sum_yy = 0
        simple_sum_y =  0
        simple_sum_error = 0
        simple_sum_count = 0

        for iS, iSet in enumerate(pair_train_set):

            is_mat = np.zeros((nb, nb))
            # ridge regression - regularized normal equation
            for ii in range(nb):
                is_mat[ii,ii] = 1.0/(s[iS, ii] + tolval)
        
            hJN[iS,:] = v[iS, :, :] @ is_mat @ (u[iS, :, :] @ Cxy[iS,:])

            # Find x and actual y:
            pair = srData["datasets"][iSet]
            if (nLaguerre > 0 ):
                x = generate_x(pair, feature, basis_args = basis_args, xGen = 'LG', nPoints=nPoints, nLaguerre=nLaguerre)
            else:
                x = generate_x(pair, feature, basis_args = basis_args, xGen = 'DG', nPoints=nPoints)
            y = pair["resp"]["psth_smooth"] 
            if "weights" not in pair["resp"]:
                yw = np.ones_like(y)
            else:
                yw = pair["resp"]["weights"]

            x = x[:, yw > 0]
            y = y[yw > 0]
    
            # Get the prediciton
            ypred = hJN[iS, :]@ (x - xavg[iS]) + yavg[iS]

            # add the contribution from segmented model
            xs = generate_x(pair, feature = 'onoff_feature', xGen = 'Kernel', nPoints=nPoints)
            ys_pred = segModel['weights']@ xs[:, yw>0] + segModel['bias']
            ypred += ys_pred
            yw = yw[yw > 0]

            # Get values to calculate R2-CV - here it is the coefficient of determination
            sum_count = np.sum(yw)
            sum_y = np.sum(y*yw)
            sum_yy = np.sum(y*y*yw)
            sum_error2 = np.sum(((ypred-y)**2)*yw)

            simple_sum_yy += sum_yy
            simple_sum_y +=  sum_y
            simple_sum_error += sum_error2
            simple_sum_count += sum_count
        

        y_mean = simple_sum_y/simple_sum_count
        y_var = simple_sum_yy/simple_sum_count - y_mean**2
        y_error = simple_sum_error/simple_sum_count

        # This is a "one-trial" CV
        R2CV[it] = 1.0 - y_error/y_var
     
    # Find the best tolerance level, i.e. the ridge penalty hyper-parameter
    segIDModel = {}
    itMax = np.argmax(R2CV)
    segIDModel['Tolerances'] = tol
    segIDModel['BestITolInd'] = itMax
    segIDModel['R2CV'] = R2CV

    # Calculate one filter at the best tolerance
    uAll,sAll,vAll = np.linalg.svd(CxxAll/countAll)
    for ii in range(nb):
        is_mat[ii,ii] = 1.0/(sAll[ii] + ranktol[itMax])
    
    hJNAll = vAll @ is_mat @ (uAll @ (CxyAll/countAll))
    # The bias term
    b0 = -hJNAll @ (xsumAll/countAll) + (ysumAll/countAll)
    segIDModel['weights'] = hJNAll
    segIDModel['bias'] = b0[0,0]
    segIDModel['yavg'] = ysumAll/countAll
    segIDModel['xavg'] = xsumAll/countAll


    # 6. Return the ridge model for the residuals, the ridge model for the onsets, and the laguerre parameters
    return pca, segIDModel, segModel, basis_args


# high level function
def process_unit_strf(nwb_file, unit_name, model_dir=None, trials_type='playback_trials'):
    respChunkLen = 100 # ms of stim to use in each chunk of feature space
    segmentBuffer = 30 # ms to add at the beginning of each segment
    strfLength = 200 # number of points in the STRF in sampling rate - 200 ms for Theunissen data
    smooth_rt = 31 # smoothing window for the R2 calculation for the strf.  The segmented model is also fitted on a smooth_psth with the same time window.
    all_models = dict()

    # Calculate spectrogram, smooth psth and make a new object the stimulus-response Data: srData
    srData = preprocSound.preprocess_sound_nwb(nwb_file, trials_type, unit_name, preprocess_type='ft')
    preprocess_srData(srData, plot=False, respChunkLen=respChunkLen, segmentBuffer=segmentBuffer, tdelta=0, plotFlg = False)

    # Estimate the single trial SNR for this data set
    snr = preprocSound.estimate_SNR(srData)
    evOne= snr/(snr + 1)     # The expected variance (R2-ceiling) for one trial

    # The Classic STRF

    # Initialize the linear time invariant model. Here we choose 200 delay points 
    nStimChannels = srData['nStimChannels']
    strfDelays = np.arange(strfLength)
    modelParams = strfSetup.linInit(nStimChannels, strfDelays)

    # Convert srData into a format that strflab understands
    allstim, allresp, allweights, groupIndex = strfSetup.srdata2strflab(srData, useRaw = False)
    globDat = strfSetup.strfData(allstim, allresp, allweights, groupIndex)
    
    # Additional model options
    modelParams['Tol_val'] = [0.100, 0.050, 0.010, 0.005, 1e-03, 1e-04, 5e-05, 0]  # These are the same as the default in fit_seg
    modelParams['sparsenesses'] = [0, 1, 2, 3, 4, 5, 6, 7]   # The sparseness is a lasso like regularization
    modelParams['timevary_PSTH'] = 0           # This is to calculate a time-varying mean across all stim
    modelParams['smooth_rt'] = smooth_rt
    modelParams['ampsamprate'] = srData['stimSampleRate']
    modelParams['respsamprate'] = srData['respSampleRate']
    modelParams['infoFreqCutoff'] = 100        # For the coherence-based Info calculation this is the frequency CutOff in Hz
    modelParams['infoWindowSize'] = 0.250      # Window size in s for the coherence estimate
    modelParams['TimeLagUnit'] = 'frame'       # Can be set to 'frame' or 'msec'
    modelParams['outputPath'] = os.path.join(tempfile.gettempdir(), srData['UUID'])  # Temporary path to store the results
    modelParams['TimeLag'] =  int(np.ceil(np.max(np.abs(modelParams['delays']))))


    # Run direct fit optimization on all of the data
    modelParams = trnDirectFit.trnDirectFit(modelParams, globDat)
    r2STRF = modelParams['R2CV'].max()
    if isinstance(nwb_file, nwb.NWBFile):
        identifier = nwb_file.identifier
    else:
        identifier = nwb_file
    all_models['strfModel'] = modelParams

    # we will save the models to the model directory
    if model_dir is not None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("Saving Models to %s" % model_dir)
        unit_model_dir = os.path.join(model_dir, f"{identifier}_{unit_name}")
        if not os.path.exists(unit_model_dir):
            os.makedirs(unit_model_dir)
        # Save the models as pickle files
        for model in all_models.keys():
            model_path = os.path.join(unit_model_dir, f"{int(unit_name)}_{model}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(all_models[model], f)
        print("Models saved successfully.")
    result = {
        'nwb_file_identifier': identifier,
        'unit' : unit_name,
        'r2Ceil' : evOne,
        'r2STRF' : r2STRF
    }
    
    return result

def process_unit(nwb_file, unit_name, model_dir=None, trials_type='playback_trials'):
    ''' Fits using ridge or modified ridge 4 encoding models using the data structure in srData.  The four encoding models are:
            a segmentation model,
            a segmenation + identification using LG expansion
            a segmentation + identification using DOGS expansion
            a classic STRF
            
    '''

    # Segmentation and fit parameters - this could be a dictionary of options
    respChunkLen = 100 # ms of stim to use in each chunk of feature space
    segmentBuffer = 30 # ms to add at the beginning of each segment
    nLaguerre = 25 # number of laguerre functions to use
    feature = 'spect_windows'
    feature2 = 'mps_windows'
    event_types = 'onoff_feature'
    nPoints = 150 # number of points to use in the kernel
    strfLength = 200 # number of points in the STRF in sampling rate - 200 ms for Theunissen data
    nPCs = 20
    nDOGS = 5
    smooth_rt = 31 # smoothing window for the R2 calculation for the strf.  The segmented model is also fitted on a smooth_psth with the same time window.
    all_models = dict()

    # Calculate spectrogram, smooth psth and make a new object the stimulus-response Data: srData
    srData = preprocSound.preprocess_sound_nwb(nwb_file, trials_type, unit_name, preprocess_type='ft')
    preprocess_srData(srData, plot=False, respChunkLen=respChunkLen, segmentBuffer=segmentBuffer, tdelta=0, plotFlg = False)

    # Estimate the single trial SNR for this data set
    snrEst, f, snrEstf, cumInfo, totWeight  = preprocSound.estimate_SNR(srData)
    evOne= snrEst/(snrEst + 1)     # The expected variance (R2-ceiling) for one trial - not used here
    r2num = 0
    r2den = 0
    for pair in srData["datasets"]:
        yw = pair["resp"]["weights"]
        r2num += np.sum(snrEst*yw)
        r2den += np.sum(1+snrEst*yw)

    EV = r2num/r2den

    # Fit the segmentation (on-off here) kernel (impulse response)
    segModel = fit_seg(srData, nPoints, x_feature = event_types, y_feature = 'psth_smooth', kernel = 'Kernel0', nD=2, tol=np.array([0.1, 0.01, 0.001, 0.0001]), store_error = True  )
    learned_conv_kernel = segModel['weights'].reshape(2, nPoints)
    r2segModel = np.max(segModel['R2CV'])
    all_models['segModel'] = segModel

    # Fit the on-off kernels with laguerre and DOGs
    laguerre_args = fit_kernel_LG(learned_conv_kernel, nPoints, nD=2)
    DOGS_args = fit_kernel_DG(learned_conv_kernel, nPoints, nD=2)

    # first use PCA to reduce dim of the features'
    pca_spect = generate_event_pca_feature(srData, event_types, feature, pca = None, npcs=nPCs)
    pca_mps = generate_event_pca_feature(srData, event_types, feature2, pca = None, npcs=nPCs)
    all_models['pca_spect'] = pca_spect
    all_models['pca_mps'] = pca_mps

    # Calculate the segmented encoding models for spectrograms, LGs and DOGS.
    segIDModelLG = fit_seg(srData, nPoints, feature, y_feature = 'error_Kernel0', y_R2feature = 'psth_smooth', kernel = 'LG', basis_args =laguerre_args, nD=nLaguerre) 
    r2segIDModelLG = np.max(segIDModelLG['R2CV'])
    segIDModelDG = fit_seg(srData, nPoints, feature, y_feature = 'error_Kernel0', y_R2feature = 'psth_smooth', kernel = 'DG', basis_args =DOGS_args, nD=nDOGS)
    r2segIDModelDG = np.max(segIDModelDG['R2CV'])
    all_models['segIDModelLG'] = segIDModelLG
    all_models['segIDModelDG'] = segIDModelDG

    # Repeat using the MPS
    segIDModelLGMPS = fit_seg(srData, nPoints, feature2, y_feature = 'error_Kernel0', y_R2feature = 'psth_smooth', kernel = 'LG', basis_args =laguerre_args, nD=nLaguerre) 
    r2segIDModelLGMPS = np.max(segIDModelLGMPS['R2CV'])
    segIDModelDGMPS = fit_seg(srData, nPoints, feature2, y_feature = 'error_Kernel0', y_R2feature = 'psth_smooth', kernel = 'DG', basis_args =DOGS_args, nD=nDOGS)
    r2segIDModelDGMPS = np.max(segIDModelDGMPS['R2CV'])
    all_models['segIDModelLGMPS'] = segIDModelLGMPS
    all_models['segIDModelDGMPS'] = segIDModelDGMPS

    # The Classic STRF

    # Initialize the linear time invariant model. Here we choose 200 delay points 
    nStimChannels = srData['nStimChannels']
    strfDelays = np.arange(strfLength)
    modelParams = strfSetup.linInit(nStimChannels, strfDelays)

    # Convert srData into a format that strflab understands
    allstim, allresp, allweights, groupIndex = strfSetup.srdata2strflab(srData, useRaw = False)
    globDat = strfSetup.strfData(allstim, allresp, allweights, groupIndex)
    
    # Additional model options
    modelParams['Tol_val'] = [0.100, 0.050, 0.010, 0.005, 1e-03, 1e-04, 5e-05, 0]  # These are the same as the default in fit_seg
    modelParams['sparsenesses'] = [0, 1, 2, 3, 4, 5, 6, 7]   # The sparseness is a lasso like regularization
    modelParams['timevary_PSTH'] = 0           # This is to calculate a time-varying mean across all stim
    modelParams['smooth_rt'] = smooth_rt
    modelParams['ampsamprate'] = srData['stimSampleRate']
    modelParams['respsamprate'] = srData['respSampleRate']
    modelParams['infoFreqCutoff'] = 100        # For the coherence-based Info calculation this is the frequency CutOff in Hz
    modelParams['infoWindowSize'] = 0.250      # Window size in s for the coherence estimate
    modelParams['TimeLagUnit'] = 'frame'       # Can be set to 'frame' or 'msec'
    modelParams['outputPath'] = os.path.join(tempfile.gettempdir(), srData['UUID'])  # Temporary path to store the results
    modelParams['TimeLag'] =  int(np.ceil(np.max(np.abs(modelParams['delays']))))


    # Run direct fit optimization on all of the data
    modelParams = trnDirectFit.trnDirectFit(modelParams, globDat)
    r2STRF = modelParams['R2CV'].max()
    if isinstance(nwb_file, nwb.NWBFile):
        identifier = nwb_file.identifier
    else:
        identifier = nwb_file
    all_models['strfModel'] = modelParams

    # we will save the models to the model directory
    if model_dir is not None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("Saving Models to %s" % model_dir)
        unit_model_dir = os.path.join(model_dir, f"{identifier}_{unit_name}")
        if not os.path.exists(unit_model_dir):
            os.makedirs(unit_model_dir)
        # Save the models as pickle files
        for model in all_models.keys():
            model_path = os.path.join(unit_model_dir, f"{int(unit_name)}_{model}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(all_models[model], f)
        print("Models saved successfully.")
    result = {
        'nwb_file_identifier': identifier,
        'unit' : unit_name,
        'r2Ceil' : EV,
        'snr' : snrEst,
        'f' : f,
        'snrEstf' : snrEstf,
        'cumInfo' : cumInfo,
        'totWeight' : totWeight,
        'segModel' : segModel,
        'r2segModel' : r2segModel,
        'Laguerre_args' :  laguerre_args,
        'dogs_args' : DOGS_args,
        'r2segIDModelLG' : r2segIDModelLG,
        'r2segIDModelDG' : r2segIDModelDG,
        'r2segIDModelLGMPS' : r2segIDModelLGMPS,
        'r2segIDModelDG' : r2segIDModelDGMPS,
        'r2STRF' : r2STRF
    }
    
    return result

def process_unit_nostrf(nwb_file, unit_name, model_dir=None, trials_type='playback_trials'):
    ''' Fits using ridge or modified ridge 4 encoding models using the data structure in srData.  The four encoding models are:
            a segmentation model,
            a segmenation + identification using LG expansion
            a segmentation + identification using DOGS expansion
    This is a version that does not fit the STRF, so it is faster.            
    '''

    # Segmentation and fit parameters - this could be a dictionary of options
    respChunkLen = 100 # ms of stim to use in each chunk of feature space
    segmentBuffer = 30 # ms to add at the beginning of each segment
    nLaguerre = 25 # number of laguerre functions to use
    feature = 'spect_windows'
    feature2 = 'mps_windows'
    event_types = 'onoff_feature'
    nPoints = 150 # number of points to use in the kernel
    nPCs = 20
    nDOGS = 5
    all_models = dict()

    # Calculate spectrogram, smooth psth and make a new object the stimulus-response Data: srData
    srData = preprocSound.preprocess_sound_nwb(nwb_file, trials_type, unit_name, preprocess_type='ft')
    preprocess_srData(srData, plot=False, respChunkLen=respChunkLen, segmentBuffer=segmentBuffer, tdelta=0, plotFlg = False)

    # Estimate the single trial SNR for this data set
    snr = preprocSound.estimate_SNR(srData)
    evOne= snr/(snr + 1)     # The expected variance (R2-ceiling) for one trial

    # Fit the segmentation (on-off here) kernel (impulse response)
    segModel = fit_seg(srData, nPoints, x_feature = event_types, y_feature = 'psth_smooth', kernel = 'Kernel', nD=2, tol=np.array([0.1, 0.01, 0.001, 0.0001, 0.00001, 0]), store_error = True  )
    learned_conv_kernel = segModel['weights'].reshape(2, nPoints)
    r2segModel = np.max(segModel['R2CV'])
    all_models['segModel'] = segModel

    # Fit the on-off kernels with laguerre and DOGs
    laguerre_args = fit_kernel_LG(learned_conv_kernel, nPoints, nD=2)
    DOGS_args = fit_kernel_DG(learned_conv_kernel, nPoints, nD=2)

    # first use PCA to reduce dim of the features'
    pca_spect = generate_event_pca_feature(srData, event_types, feature, pca = None, npcs=nPCs)
    pca_mps = generate_event_pca_feature(srData, event_types, feature2, pca = None, npcs=nPCs)
    all_models['pca_spect'] = pca_spect
    all_models['pca_mps'] = pca_mps

    # Calculate the segmented encoding models for spectrograms, LGs and DOGS.
    segIDModelLG = fit_seg(srData, nPoints, feature, y_feature = 'psth_smooth', kernel = 'LG', basis_args =laguerre_args, nD=nLaguerre) # 'error_Kernel', y_R2feature =
    r2segIDModelLG = np.max(segIDModelLG['R2CV'])
    segIDModelDG = fit_seg(srData, nPoints, feature, y_feature = 'psth_smooth', kernel = 'DG', basis_args =DOGS_args, nD=nDOGS)
    r2segIDModelDG = np.max(segIDModelDG['R2CV'])
    all_models['segIDModelLG'] = segIDModelLG
    all_models['segIDModelDG'] = segIDModelDG

    # Repeat using the MPS
    segIDModelLGMPS = fit_seg(srData, nPoints, feature2, y_feature = 'psth_smooth', kernel = 'LG', basis_args =laguerre_args, nD=nLaguerre) 
    r2segIDModelLGMPS = np.max(segIDModelLGMPS['R2CV'])
    segIDModelDGMPS = fit_seg(srData, nPoints, feature2, y_feature = 'psth_smooth', kernel = 'DG', basis_args =DOGS_args, nD=nDOGS)
    r2segIDModelDGMPS = np.max(segIDModelDGMPS['R2CV'])
    all_models['segIDModelLGMPS'] = segIDModelLGMPS
    all_models['segIDModelDGMPS'] = segIDModelDGMPS
    
    if isinstance(nwb_file, nwb.NWBFile):
        identifier = nwb_file.identifier
    else:
        identifier = nwb_file

    # we will save the models to the model directory
    if model_dir is not None:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("Saving Models to %s" % model_dir)
        unit_model_dir = os.path.join(model_dir, f"{identifier}_{unit_name}")
        if not os.path.exists(unit_model_dir):
            os.makedirs(unit_model_dir)
        # Save the models as pickle files
        for model in all_models.keys():
            model_path = os.path.join(unit_model_dir, f"{int(unit_name)}_{model}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(all_models[model], f)
        print("Models saved successfully.")
    result = {
        'nwb_file_identifier': identifier,
        'unit' : unit_name,
        'r2Ceil' : evOne,
        'r2segModel' : r2segModel,
        'Laguerre_args' :  laguerre_args,
        'dogs_args' : DOGS_args,
        'r2segIDModelLG' : r2segIDModelLG,
        'r2segIDModelDG' : r2segIDModelDG,
        'r2segIDModelLGMPS' : r2segIDModelLGMPS,
        'r2segIDModelDG' : r2segIDModelDGMPS,
    }
    
    return result
    
