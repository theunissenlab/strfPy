# Dependencies - General Stuff
import sys
import os
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from glob import glob
from scipy.signal import windows, fftconvolve
# from scipy.io import wavfile
from scipy.special import genlaguerre
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import seaborn as sns
from functools import partial
from statistics import mode

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

def generate_predictionV2(
    pair, model, feature, basis_args, nPoints=200, nLaguerre=5
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

    y_pred = model['weights']@ x + model['bias']
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
    derivativeThresh = 0.2  # Threshold derivative 0.5 dB per ms.
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


# fitting funcitons


def fit_seg_model(
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
    all_spect_windows = np.concatenate(
        [
            np.asarray(srData["datasets"][iSet]["events"][feature]).reshape(
                (len(srData["datasets"][iSet]["events"]["index"]), nfeats)
            )
            for iSet in range(pairCount)
        ],
        axis=0,
    )

    if pair_train_set is None:
        pair_train_set = np.arange(pairCount)


    # 1. first use PCA to reduce dim of the features'
    print("Fitting PCA to Feature")
    if pca is None:
        npcs = 20
        pca = PCA(n_components=npcs)
        pca.fit(all_spect_windows)
    else:
        npcs = pca.n_components_

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
    
    # Clear some memory
    all_spect_windows = None


    # 2. now fit the onsets with a convolutional kernel
    #       This removes the average response to onsets and offsets
    print("Fitting convolutional kernel")
    X = None
    Y = None
    Y_weights = None

    for iSet in pair_train_set:
        pair = srData["datasets"][iSet]
        x = arbitrary_kernel(pair, nPoints=nPoints)
        if "weights" not in pair["resp"]:
            yw = np.ones_like(pair["resp"]["psth_smooth"])
        else:
            yw = pair["resp"]["weights"]
        y = pair["resp"]["psth_smooth"]
        x = x[:, yw > 0]
        y = y[yw > 0]
        yw = yw[yw > 0]
        # yw = yw / np.max(yw)
        if X is None:
            X = x
        else:
            X = np.hstack([X, x])
        if Y is None:
            Y = y
            Y_weights = yw
        else:
            Y = np.hstack([Y, y])
            Y_weights = np.hstack([Y_weights, yw])

    SegModel_ridge = RidgeCV()
    SegModel_ridge.fit(X.T, Y, sample_weight=Y_weights)
    learned_conv_kernel = SegModel_ridge.coef_.reshape(2, nPoints)

    # Clear some memory
    X = None
    Y = None

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
    feature = pair["events"][feature_key]
    if feature.ndim == 1:
        feature = feature[:, np.newaxis]
    nFeatures = feature.shape[1]

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
            x = generate_laguerre_features(
                pair,
                feature_key=feature_key,
                resp_key="psth_smooth",
                laguerre_args=basis_args[:,0:2],
                nLaguerrePoints=nPoints,
                nLaguerre=nLaguerre
            )
        else:
            x = generate_dogs_features(
                pair,
                feature_key=feature_key,
                resp_key="psth_smooth",
                dogs_args=basis_args,
                nPoints=nPoints
            )
        

        y = pair["resp"]["psth_smooth"]  
        if "weights" not in pair["resp"]:
            yw = np.ones_like(y)
        else:
            yw = pair["resp"]["weights"]

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
            x = generate_laguerre_features(
                pair,
                feature_key=feature_key,
                resp_key="psth_smooth",
                laguerre_args=basis_args[:,0:2],
                nLaguerrePoints=nPoints,
                nLaguerre=nLaguerre,
            )
        else:
            x = generate_dogs_features(
                pair,
                feature_key=feature_key,
                resp_key="psth_smooth",
                dogs_args=basis_args,
                nPoints=nPoints,
            )

        y = pair["resp"]["psth_smooth"]  
        if "weights" not in pair["resp"]:
            yw = np.ones_like(y)
        else:
            yw = pair["resp"]["weights"]

        x = x[:, yw > 0]

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
                x = generate_laguerre_features(
                    pair,
                    feature_key=feature_key,
                    resp_key="psth_smooth",
                    laguerre_args=basis_args[:,0:2],
                    nLaguerrePoints=nPoints,
                    nLaguerre=nLaguerre,
                )
            else:
                x = generate_dogs_features(
                    pair,
                    feature_key=feature_key,
                    resp_key="psth_smooth",
                    dogs_args=basis_args,
                    nPoints=nPoints
                )
            y = pair["resp"]["psth_smooth"] 
            if "weights" not in pair["resp"]:
                yw = np.ones_like(y)
            else:
                yw = pair["resp"]["weights"]
          
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
    b0 = hJNAll @ (xsumAll/countAll) + (ysumAll/countAll)
    segIDModel['weights'] = hJNAll
    segIDModel['bias'] = b0[0,0]


    # 6. Return the ridge model for the residuals, the ridge model for the onsets, and the laguerre parameters
    return pca, segIDModel, SegModel_ridge, basis_args


# high level function


def process_unit(
    srData,
    feature="log_spect_windows",
    nPoints=200,
    nLaguerre=5,                          # Set to -1 to use DOGS and their derivatives
    pca=None,
    LOO=True,                             # LeaveOneOut (LOO) cross-validation. Set to False for speed and testing but you won't get cross-validation data
                                          # LOO is performed by leaving out all datasets obtained for a particular stim name.
):

    # parameters for fit
    event_types = "onoff_feature"
    pairCount = len(srData["datasets"])


    # First fit the full model with the entire data set
    pair_train_set = np.arange(pairCount)
    pca, segIDModel, ridge_conv_filter, basis_args = fit_seg_model(
        srData,
        nLaguerre,
        nPoints,
        event_types,
        feature,
        pair_train_set,
        pca=pca,
    )

    # Get predictions and calculate the single R2 for segment only model called here simple for the entire data set used for training and testing
    simple_sum_xy = 0
    simple_sum_xx = 0
    simple_sum_yy = 0
    simple_sum_x =  0
    simple_sum_y =  0
    simple_sum_count = 0
    ntrials = np.zeros(len(pair_train_set))
    for iSet in pair_train_set:
        resp = srData['datasets'][iSet]['resp']
        ntrials[iSet] = len(resp['trialDurations'])
    
    ntrialsR2 = int(mode(ntrials))
    print('Max trials %d Min trials %d  Mode %d' % (ntrials.max(), ntrials.min(), ntrialsR2) )

    for iSet in pair_train_set:
        if (ntrials[iSet] >= ntrialsR2) :
            sum_x, sum_xx, sum_y, sum_yy, sum_xy, sum_count = get_simple_prediction_r2_Values(srData["datasets"][iSet], ridge_conv_filter, nPoints, ntrialsR2, smWindow=31)
            simple_sum_xy += sum_xy
            simple_sum_xx += sum_xx
            simple_sum_yy += sum_yy
            simple_sum_x +=  sum_x
            simple_sum_y +=  sum_y
            simple_sum_count += sum_count

    x_mean = simple_sum_x/simple_sum_count
    y_mean = simple_sum_y/simple_sum_count
    simple_r2_train = (simple_sum_xy/simple_sum_count - x_mean*y_mean)**2/((simple_sum_xx/simple_sum_count - x_mean**2)*(simple_sum_yy/simple_sum_count - y_mean**2))

     
    all_r2_train = 0
    all_r2_test = segIDModel['R2CV'][segIDModel['BestITolInd']]
    
    LOO = False
    if LOO:
    # Now do Leave One Out Cross-Validation (LOOCV)
        print('Starting cross-validation calculations')

    # Reset all counters
        simple_sum_xy = 0
        simple_sum_xx = 0
        simple_sum_yy = 0
        simple_sum_x =  0
        simple_sum_y =  0
        simple_sum_count = 0

        all_sum_xy = 0
        all_sum_xx = 0
        all_sum_yy = 0
        all_sum_x =  0
        all_sum_y =  0
        all_sum_count = 0

        # Find unique stimuli
        stim_names = []
        for ds in srData['datasets']:
            stim_names.append(ds['stim']['rawFile'])

        unique_stims = np.unique(stim_names)
    
        for stimLO in unique_stims:

            pair_test_set = []
            for ids, ds in enumerate(srData['datasets']):
                if (ds['stim']['rawFile'] == stimLO):
                    pair_test_set.append(ids)
            
            pair_test_set = np.array(pair_test_set)
            pair_train_set = np.setdiff1d(range(pairCount), pair_test_set)
        
            # Fit model on training set
            pca_loo, ridge_loo, ridge_conv_filter_loo, basis_args_loo = fit_seg_model(
                srData,
                nLaguerre,
                nPoints,
                event_types,
                feature,
                pair_train_set,
                pca=pca,
            )

            # Get Prediction on left out set
            # First for segmented model
            for iLO in pair_test_set:
                sum_x, sum_xx, sum_y, sum_yy, sum_xy, sum_count =get_simple_prediction_r2_Values (srData["datasets"][iLO], ridge_conv_filter_loo, nPoints, ntrialsR2, smWindow=31)
                simple_sum_xy += sum_xy
                simple_sum_xx += sum_xx
                simple_sum_yy += sum_yy
                simple_sum_x +=  sum_x
                simple_sum_y +=  sum_y
                simple_sum_count += sum_count

                # Next for segmented + Indentification model
                if (nLaguerre > 0):
                    sum_x, sum_xx, sum_y, sum_yy, sum_xy, sum_count = get_prediction_r2_Values(srData["datasets"][iLO],
                                                        ridge_loo, feature, basis_args_loo[:,0:2], ridge_conv_filter_loo, nPoints, ntrialsR2, nLaguerre=nLaguerre)
                else:
                    sum_x, sum_xx, sum_y, sum_yy, sum_xy, sum_count = get_prediction_r2_Values(srData["datasets"][iLO],
                                                        ridge_loo, feature, basis_args_loo, ridge_conv_filter_loo, nPoints, ntrialsR2, nLaguerre=nLaguerre)

                all_sum_xy += sum_xy
                all_sum_xx += sum_xx
                all_sum_yy += sum_yy
                all_sum_x +=  sum_x
                all_sum_y +=  sum_y
                all_sum_count += sum_count

        # Now calculate R2 values
        x_mean = simple_sum_x/simple_sum_count
        y_mean = simple_sum_y/simple_sum_count
        simple_r2_test = (simple_sum_xy/simple_sum_count - x_mean*y_mean)**2/((simple_sum_xx/simple_sum_count - x_mean**2)*(simple_sum_yy/simple_sum_count - y_mean**2))
    
        x_mean = all_sum_x/all_sum_count
        y_mean = all_sum_y/all_sum_count    
        all_r2_test = (all_sum_xy/all_sum_count - x_mean*y_mean)**2/((all_sum_xx/all_sum_count - x_mean**2)*(all_sum_yy/all_sum_count - y_mean**2))
    else:
        simple_r2_test = 0.0
        # all_r2_test = 0.0

    return (
        srData,
        pca,
        segIDModel,
        ridge_conv_filter,
        basis_args,
        [
            simple_r2_train,
            simple_r2_test,
            all_r2_train,
            all_r2_test,
            ntrialsR2
        ]
    )  # , R2Ceil
