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
from scipy.io import wavfile
from scipy.special import genlaguerre
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
import seaborn as sns
from functools import partial

# Depednecies from Theunissen Lab
from soundsig.sound import BioSound
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


def arbitrary_kernel(
    pair, nPoints=200, event_key="onoff_feature", resp_key="psth", mult_values=False
):
    """
    Generate a kernel matrix for a given event in the pair data.

    Parameters:
    pair (dict): A dictionary containing 'resp' and 'events' keys. 'resp' should have a 'psth' key with the response data.
                 'events' should have keys corresponding to event names and their values.
    event_name (str, optional): The name of the event to be used. Defaults to 'onoff_feature'.
    nPoints (int, optional): Number of points for the kernel. Defaults to 200.
    mult_values (bool, optional): If True, multiply the feature values by the event values. Defaults to False.

    Returns:
    np.ndarray: A 2D array representing the kernel matrix convolved with the event data.
    """
    nT = pair["resp"][resp_key].size
    feature = pair["events"][event_key]

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
        X[i * nPoints : (i + 1) * nPoints, pair["events"]["index"]] = (
            feature[:, i] * values
        )

    # now stack the kern_mat to
    kern_mat = np.vstack([np.eye(nPoints)] * num_features)
    X = fftconvolve(X, kern_mat, axes=1, mode="full")[:, :nT]
    return X


def generate_laguerre_features(
    pair, feature_key, laguerre_args=np.zeros((2, 3)), nLaguerrePoints=300, nLaguerre=5
):
    """
    Generate Laguerre features for a given pair of data.

    Parameters:
    pair (dict): Dictionary containing 'resp' and 'events' keys. 'resp' should have a 'psth' key with response data.
                    'events' should have 'index' and feature_key keys.
    feature_key (str): Key to access the features in the 'events' dictionary.
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
    nT = pair["resp"]["psth"].size
    nFeatures = pair["events"][feature_key].shape[1]
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
    x_t = np.arange(nLaguerrePoints)
    for iEventType in range(nEventsTypes):
        laguerre_amp, laguerre_tau, laguerre_alpha = laguerre_args[iEventType]
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
    inds = pair["events"][
        "index"
    ]  # + int(laguerre_dt_s*srData['datasets'][iSet]['resp']['sampleRate'])
    # inds = np.clip(inds, 0, nT-1)
    X[:, pair["events"]["index"]] = np.hstack(
        [pair["events"][feature_key]] * nLaguerre
    ).T
    # now convolve the laguerre function with the feature value
    X = fftconvolve(X, laguerre_mat, axes=1, mode="full")[:, :nT]
    return X


# Slated for removal
def get_simple_prediction_r2(pair, ridge_conv_filter, nPoints, mult_values=False):
    y = pair["resp"]["psth_smooth"]
    x = arbitrary_kernel(pair, nPoints=nPoints, mult_values=mult_values)
    y_pred = ridge_conv_filter.predict(x.T)
    return np.corrcoef(y, y_pred)[0, 1] ** 2


def get_prediction_r2(
    pair, ridge, feature, laguerre_args, ridge_conv_filter, nPoints, nLaguerre=5
):
    y = pair["resp"]["psth_smooth"]
    y_pred = generate_prediction(
        pair, ridge, feature, laguerre_args, ridge_conv_filter, nPoints, nLaguerre
    )
    return np.corrcoef(y, y_pred)[0, 1] ** 2


# end slated for removal


def gen_y_avg_laguerre(pair, laguerre_args, nPts):
    x = generate_laguerre_features(
        pair, "onoff_feature", laguerre_args, nLaguerrePoints=nPts, nLaguerre=5
    )
    y = pair["resp"]["psth_smooth"]
    ridge = RidgeCV()
    ridge.fit(x.T, y)
    return ridge.predict(x.T)


def gen_y_avg(pair, ridge_conv_filter, nPoints=200, mult_values=False):
    x = arbitrary_kernel(pair, nPoints=nPoints, mult_values=mult_values)
    y_pred = ridge_conv_filter.predict(x.T)
    return y_pred


def generate_prediction(
    pair, ridge, feature, laguerre_args, ridge_conv_filter, nPoints=200, nLaguerre=5
):
    x = generate_laguerre_features(
        pair,
        "pca_%s" % feature,
        laguerre_args,
        nLaguerrePoints=nPoints,
        nLaguerre=nLaguerre,
    )
    y_pred = ridge.predict(x.T)
    y_pred[y_pred < 0] = 0
    return y_pred


def generate_pred_score(
    pair, ridge, feature, laguerre_args, ridge_conv_filter, nPoints=200, nLaguerre=5
):
    x = generate_laguerre_features(
        pair,
        "pca_%s" % feature,
        laguerre_args,
        nLaguerrePoints=nPoints,
        nLaguerre=nLaguerre,
    )
    # y_avg = gen_y_avg_laguerre(pair, laguerre_args, nPoints)
    y_avg = gen_y_avg(pair, ridge_conv_filter, nPoints)
    y = pair["resp"]["psth_smooth"]
    return ridge.score(x.T, y)


# preproc function


def preprocess_srData(srData, plot=False, respChunkLen=150, segmentBuffer=25, tdelta=0):
    """
    Preprocesses stimulus-response data by segmenting the stimulus based on its envelope, calculating the spectrogram,
    PSTH (Peri-Stimulus Time Histogram), and MPS (Modulation Power Spectrum).
    Parameters:
    srData (dict): Dictionary containing stimulus-response data.
    plot (bool, optional): If True, plots the results. Default is False.
    respChunkLen (int, optional): Total chunk length (including segment buffer) in number of points. Default is 150.
    segmentBuffer (int, optional): Number of points on each side of segment for response and MPS. Default is 25.
    tdelta (int, optional): Time delta to offset the events. Default is 0.
    Returns:
    None: The function modifies the srData dictionary in place, adding preprocessed data to it.
    """
    # PREPROCESSING
    # - Segmentation of the stimulus based on the envelope
    # - Calculation of the spectrogram
    # - Calculation of the PSTH
    # - Calculation of the MPS

    # Segmentation based on derivative of the envelope
    ampThresh = 20.0  # Threshold in dB where 50 is max

    minSound = 25  # Minimum distance between peaks or troffs
    derivativeThresh = 0.2  # Threshold derivative 0.5 dB per ms.
    # segmentBuffer = 30 # Number of points on each side of segment for response and MPS - time units given by stim sample rate
    # respChunkLen = 150 # Total chunk length (including segment buffer) in number of points
    DBNOISE = 50  # Set a baseline for everything below 70 dB from max

    wHann = windows.hann(
        21, sym=True
    )  # The 21 ms (number of points) hanning window used to smooth the PSTH
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
        dBMax = spectro.max()
        spectro[spectro < dBMax - DBNOISE] = dBMax - DBNOISE
        # set the y ticks to freq
        nFreqs = len(srData["datasets"][iSet]["stim"]["tfrep"]["f"])

        ampenv = np.mean(spectro, axis=0)
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
            constant_values=(dBMax - DBNOISE, dBMax - DBNOISE),
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

        if plot:
            plt.figure(figsize=(8, 2), dpi=100)
            # plt.plot(ampdev)

            # plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(spectro, aspect="auto", cmap=spec_colormap(), origin="lower")
            plt.axhline(derivativeThresh, color="k")

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
            plt.xlim(xlim)

    # lets normalize all the mps
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
    srData, nLaguerre, nPoints, event_types, feature, pair_train_set=None, pca=None
):
    """
    Fits a segmented model to the given data.
    Parameters:
    srData (dict): The dataset containing events and responses.
    nLaguerre (int): The number of Laguerre functions to use.
    nPoints (int): The number of points for the convolutional kernel.
    event_types (str): The type of events to consider.
    feature (str): The feature to use for PCA.
    pair_train_set (list): List of dataset indices to use for training.
    pca (PCA, optional): Pre-fitted PCA object. If None, a new PCA will be fitted. Default is None.
    Returns:
    tuple: A tuple containing:
        - pca (PCA): The fitted PCA object.
        - ridge (RidgeCV): The ridge regression model for the response residuals.
        - ridge_conv_filter (RidgeCV): The ridge regression model for the convolutional kernel.
        - laguerre_args (ndarray): The fitted Laguerre parameters.
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
    print("Fitting PCA")
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

    # 3. now fit the laguerre parameters to the convolutional kernel
    print("Fitting laguerre parameters")
    partial_laguerre = partial(laguerre, xorder=0)

    def sum_n_laguerres(xt, *args):
        amp, tau, alpha, *w = args
        nL = len(w)
        out = np.zeros_like(xt, dtype=float)
        for iL in range(nL):
            out += w[iL] * laguerre(xt, amp, tau, alpha, xorder=iL)  # TODO FIX
        return out

    laguerre_args = np.zeros((nEventTypes, 3))
    for iEventType in range(nEventTypes):
        popt, pcov = curve_fit(
            sum_n_laguerres,
            np.arange(nPoints),
            learned_conv_kernel[iEventType, :],
            p0=[2, 6, 5, 1, 1, 1, 1, 1],
            bounds=(
                [-np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            ),
            method="trf",
        )
        laguerre_args[iEventType, :] = popt[:3]

    # 4. now fit the response to onsets and offsets using the features
    print("Removing average response to onsets and offsets")
    Y_avg_removed = None
    Y_weights = None
    X = None
    for iSet in pair_train_set:
        pair = srData["datasets"][iSet]
        x = generate_laguerre_features(
            pair,
            "pca_%s" % feature,
            laguerre_args,
            nLaguerrePoints=nPoints,
            nLaguerre=nLaguerre,
        )
        # y = pair['resp']['psth_smooth'] - gen_y_avg_laguerre(pair, laguerre_args, nPoints)
        y = pair["resp"]["psth_smooth"]  # - gen_y_avg(pair, ridge_conv_filter, nPoints)
        if "weights" not in pair["resp"]:
            yw = np.ones_like(y)
        else:
            yw = pair["resp"]["weights"]
        x = x[:, yw > 0]
        y = y[yw > 0]
        yw = yw[yw > 0]
        if X is None:
            X = x
        else:
            X = np.hstack([X, x])
        if Y_avg_removed is None:
            Y_avg_removed = y
            Y_weights = yw
        else:
            Y_avg_removed = np.hstack([Y_avg_removed, y])
            Y_weights = np.hstack([Y_weights, yw])

    # 5. now fit the laguerre features to the response residual
    print("Fit the laguerre features to the response residual")
    SIModel_ridge = RidgeCV()
    SIModel_ridge.fit(X.T, Y_avg_removed, sample_weight=Y_weights)

    # 6. Return the ridge model for the residuals, the ridge model for the onsets, and the laguerre parameters
    return pca, SIModel_ridge, SegModel_ridge, laguerre_args


# high level function


def process_unit(
    srData,
    feature="log_spect_windows",
    nPoints=200,
    mult_values=False,
    plot=False,
    nLaguerre=5,
    pca=None,
    do_test_set=False,
):

    # parameters for fit
    event_types = "onoff_feature"
    pairCount = len(srData["datasets"])

    if do_test_set:
        pair_test_set = np.random.choice(range(pairCount), 2, replace=False)
        pair_train_set = np.setdiff1d(range(pairCount), pair_test_set)
    else:
        pair_train_set = np.arange(pairCount)
        pair_test_set = []

    pca, ridge, ridge_conv_filter, laguerre_args = fit_seg_model(
        srData,
        nLaguerre,
        nPoints,
        event_types,
        feature,
        pair_train_set,
        mult_values=mult_values,
        pca=pca,
    )

    simple_r2_train = [
        get_simple_prediction_r2(srData["datasets"][iSet], ridge_conv_filter, nPoints)
        for iSet in pair_train_set
    ]
    simple_r2_test = [
        get_simple_prediction_r2(srData["datasets"][iSet], ridge_conv_filter, nPoints)
        for iSet in pair_test_set
    ]

    all_r2_train = [
        get_prediction_r2(
            srData["datasets"][iSet],
            ridge,
            feature,
            laguerre_args,
            ridge_conv_filter,
            nPoints,
            nLaguerre,
        )
        for iSet in pair_train_set
    ]
    all_r2_test = [
        get_prediction_r2(
            srData["datasets"][iSet],
            ridge,
            feature,
            laguerre_args,
            ridge_conv_filter,
            nPoints,
            nLaguerre,
        )
        for iSet in pair_test_set
    ]

    # R2Ceil = preprocSound.calculate_EV(srData, nPoints, mult_values)
    if plot:
        plt.figure()
        # make a paired box plot of the r2 values from simple to all
        plt.scatter(simple_r2_train, all_r2_train, label="Train")
        plt.scatter(simple_r2_test, all_r2_test, label="Test")
        plt.xlabel("Event R2")
        plt.ylabel("Fit R2")
        # and plot the line y=x
        plt.plot([0, 1], [0, 1], color="black")

    return (
        srData,
        pca,
        ridge,
        ridge_conv_filter,
        laguerre_args,
        [
            np.mean(simple_r2_train),
            np.mean(simple_r2_test),
            np.mean(all_r2_train),
            np.mean(all_r2_test),
        ],
    )  # , R2Ceil
