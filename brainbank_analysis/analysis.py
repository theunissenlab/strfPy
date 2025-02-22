import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import pickle
from himalaya.backend import set_backend, get_backend
import gc
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import utils
import torch
from strfpy import findDatasets, preprocSound
from strfpy.calcSegmentedModel import (
    arbitrary_kernel,
    generate_laguerre_features,
    laguerre,
)
from functools import partial
from sklearn.preprocessing import StandardScaler, LabelEncoder
import glob
import joblib

set_backend("torch_cuda")
from himalaya.ridge import RidgeCV, GroupRidgeCV, ColumnTransformerNoStack

print("Using backend: ", get_backend())
torch.cuda.empty_cache()


def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_config(config, config_path="config.json"):
    """Save configuration to a JSON file."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def get_conv_kernel_input(
    data_dict,
    nPoints,
    resp_key="lfp_downsampled_demeaned",
    event_key="onoff_feature",
    word_buffer=0,
):
    """
    Create convolved kernel X and response Y for a given dataset

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the dataset
    nPoints : int
        Number of points to use for the kernel
    resp_key : str
        Key in data_dict containing the response
    event_key :
        Key in data_dict containing the event
    word_buffer : int
        Buffer time in seconds to consider before word onset

    Returns
    -------
    X : np.array
        Convolved kernel
    Y : np.array
        Response
    sentence_boundaries : np.array
        Index of sentence boundaries
    """
    print(f"Processing data for {resp_key}_{nPoints}")
    X = None
    Y = None
    Y_list = []

    if event_key == "onoff_feature":
        event_index_key = "index"
        buffered_event_index_key = "index_word-buffered"

    for key in data_dict.keys():
        batch_x = arbitrary_kernel(
            data_dict[key],
            nPoints=nPoints,
            event_key=event_key,
            event_index_key=buffered_event_index_key,
            resp_key=resp_key,
        )
        batch_y = data_dict[key]["resp"][f"{resp_key}"]
        Y_list.extend([batch_y])
        if X is None:
            X = batch_x
        else:
            X = np.hstack([X, batch_x])

        if Y is None:
            Y = batch_y
        else:
            Y = np.hstack([Y, batch_y])

    sentence_boundaries = np.cumsum([len(y) for y in Y_list])

    return X, Y, sentence_boundaries


def create_kernel_and_fit(
    model_file_path,
    resp_key,
    event_key,
    nPoints,
    word_buffer,
    data_dict,
    rerun_model=False,
):
    if (not os.path.exists(model_file_path)) or rerun_model:
        print(f"Computing new kernel for {resp_key}_{nPoints}")
        X, Y, sentence_boundaries = get_conv_kernel_input(
            data_dict, nPoints, resp_key, event_key, word_buffer
        )
        X, Y = X.astype(np.float32), Y.astype(np.float32)
        torch.cuda.empty_cache()
        alphas = np.logspace(-2, 5, 10)
        model = RidgeCV(
            alphas=alphas,
            fit_intercept=True,
            solver_params={"n_targets_batch": 10, "n_alphas_batch": 5},
        )
        model.fit(X.T, Y)
        gc.collect()
        torch.cuda.empty_cache()
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
    else:
        print(f"Loading model from {model_file_path}")
        with open(model_file_path, "rb") as f:
            model = pickle.load(f)
    return model


def plot_prediction_vs_actual(
    model, nPoints, event_key, resp_key, data_dict, save_file_name
):
    """
    Plot a segment of the prediction vs actual neural response during events of random sentence.
    """
    keys = list(data_dict.keys())
    random_key = np.random.choice(keys)
    x = arbitrary_kernel(
        data_dict[random_key],
        nPoints=nPoints,
        event_key=event_key,
        event_index_key="index_word-buffered",
        resp_key=resp_key,
    )
    y = data_dict[random_key]["resp"][f"{resp_key}"]
    coef = model.coef_
    if torch.is_tensor(coef):
        coef = coef.cpu().numpy()
    yhat = model.predict(x.T)

    # Print the R^2 score
    r2 = model.score(x.T, y)
    print(f"R^2 Score: {r2}")

    plt.figure(figsize=(12, 6))
    # Plot onsets of events
    event_idx = data_dict[random_key]["events"]["index"]
    plt.plot(y, label="Actual Response")
    plt.plot(yhat, label="Predicted Response")
    for idx in event_idx:
        plt.axvline(idx, color="k", linestyle="--", alpha=0.5)
    plt.legend()
    plt.title(f"Predicted vs Actual Neural Response Sentence {random_key}")
    plt.savefig(save_file_name)
    plt.close()


def plot_learned_kernel(learned_conv_kernel, word_buffer, plot_file):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    ax[0].plot(learned_conv_kernel[0])
    ax[0].set_title("Onset Kernel")
    ax[1].plot(learned_conv_kernel[1])
    ax[1].set_title("Offset Kernel")

    for i in range(2):
        ax[i].axvline(word_buffer * 100, color="k", linestyle="--", alpha=0.5)
    plt.savefig(plot_file)


def process_and_save_kernels(
    config, train_test_split=False, test_size=0.2, rerun_model=False
):
    """
    Process and save kernels using memory-efficient batching.
    """

    (
        base_dir,
        sub_num,
        trial_num,
        electrode_num,
        resp_key,
        event_key,
        nPoints,
        word_buffer,
        chunk_buffer,
        subset_value,
    ) = config.values()

    if train_test_split:
        train_data_dict_path = utils.generate_multiple_feature_file_name(
            **config, save_item="train_data_dict"
        )
        test_data_dict_path = utils.generate_multiple_feature_file_name(
            **config, save_item="test_data_dict"
        )
        train_data_dict, test_data_dict = utils.load_or_create_data_dict_train_test(
            train_data_dict_path,
            test_data_dict_path,
            sub_num,
            trial_num,
            electrode_num,
            chunk_buffer,
            word_buffer,
            subset_value,
            train_test_split=train_test_split,
            test_size=test_size,
            rerun_model=rerun_model,
        )
        model_file_path = utils.generate_multiple_feature_file_name(
            **config, save_item="model"
        )
        model = create_kernel_and_fit(
            model_file_path,
            resp_key,
            event_key,
            nPoints,
            word_buffer,
            train_data_dict,
            rerun_model=rerun_model,
        )
        prediction_plot_file_path = utils.generate_multiple_feature_file_name(
            **config, save_item="prediction_plot", extension="png"
        )
        plot_prediction_vs_actual(
            model,
            nPoints,
            event_key,
            resp_key,
            test_data_dict,
            prediction_plot_file_path,
        )

        learned_conv_kernel = model.coef_.reshape(2, nPoints)
        plot_file = utils.generate_multiple_feature_file_name(
            **config, save_item="learned_conv_kernel", extension="png"
        )
        plot_learned_kernel(learned_conv_kernel, word_buffer, plot_file)

    else:
        data_dict_path = utils.generate_multiple_feature_file_name(
            **config, save_item="data_dict"
        )
        data_dict = utils.load_or_create_data_dict(
            data_dict_path,
            sub_num,
            electrode_num,
            chunk_buffer,
            word_buffer,
            subset_value,
            rerun_model=rerun_model,
        )
        model_file_path = utils.generate_multiple_feature_file_name(
            **config, save_item="model"
        )
        model = create_kernel_and_fit(
            model_file_path,
            resp_key,
            event_key,
            nPoints,
            word_buffer,
            data_dict,
            rerun_model=rerun_model,
        )

        prediction_plot_file_path = utils.generate_multiple_feature_file_name(
            **config, save_item="prediction_plot", extension="png"
        )
        plot_prediction_vs_actual(
            model, nPoints, event_key, resp_key, data_dict, prediction_plot_file_path
        )

        learned_conv_kernel = model.coef_.reshape(2, nPoints)
        plot_file = utils.generate_multiple_feature_file_name(
            **config, save_item="learned_conv_kernel", extension="png"
        )
        plot_learned_kernel(learned_conv_kernel, word_buffer, plot_file)

    return model, learned_conv_kernel


def spect_PCA(data_dict, n_components=20):
    nEventTypes = 2
    mel_values = None
    for k in data_dict.keys():
        mel = data_dict[k]["stim"]["mel"]
        if mel_values is None:
            mel_values = mel
        else:
            mel_values = np.vstack([mel_values, mel])

    pca = PCA(n_components=n_components)
    mel_values = StandardScaler().fit_transform(mel_values)
    pca.fit(mel_values)

    for k in data_dict.keys():
        events = data_dict[k]["events"]["onoff_feature"]
        n_events = len(data_dict[k]["events"]["index"])
        mel_pca_features = pca.transform(data_dict[k]["stim"]["mel_aligned"])
        data_dict[k]["events"]["pca_mel"] = np.zeros(
            (n_events, nEventTypes * n_components)
        )
        for iEventType in range(events.shape[1]):
            data_dict[k]["events"]["pca_mel"][
                events[:, iEventType] == 1,
                iEventType * n_components : (iEventType + 1) * n_components,
            ] = mel_pca_features[events[:, iEventType] == 1, :]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(np.cumsum(pca.explained_variance_ratio_))
    ax[0].set_title("Explained Variance")
    ax[0].set_xlabel("PC")
    ax[0].set_ylabel("Cumulative Explained Variance")
    pc_ind = np.random.randint(0, n_components, 1)
    ax[1].imshow(
        pca.components_[pc_ind].reshape((1, 128)),
        aspect="auto",
        origin="lower",
        cmap="inferno",
    )
    ax[1].set_title("PC %d" % pc_ind)


def convert_speaker_labels(data_dict, speaker_mapping_file_path, test_data_dict=None):
    label_encoder = LabelEncoder()

    # Collect all speaker names from both data_dict and test_data_dict if provided
    all_speakers = []
    for k in data_dict.keys():
        all_speakers.extend(data_dict[k]["stim"]["speaker"])

    if test_data_dict is not None:
        for k in test_data_dict.keys():
            all_speakers.extend(test_data_dict[k]["stim"]["speaker"])

    label_encoder.fit(all_speakers)

    # Transform the speaker names to numerical values and save the mapping
    speaker_mapping = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

    # Save the mapping to a file
    with open(speaker_mapping_file_path, "wb") as f:
        pickle.dump(speaker_mapping, f)

    # Apply the transformation to the data_dict
    for k in data_dict.keys():
        data_dict[k]["stim"]["speaker_label"] = label_encoder.transform(
            data_dict[k]["stim"]["speaker"]
        )

    # Apply the transformation to the test_data_dict if provided
    if test_data_dict is not None:
        for k in test_data_dict.keys():
            test_data_dict[k]["stim"]["speaker_label"] = label_encoder.transform(
                test_data_dict[k]["stim"]["speaker"]
            )
        return data_dict, test_data_dict, speaker_mapping
    else:
        return data_dict, speaker_mapping


def add_speaker_event_labels(data_dict, top_speakers_mapping):
    for k in data_dict.keys():
        # Initialize speaker event labels with top_speakers_mapping
        for top_speaker in top_speakers_mapping:
            data_dict[k]["stim"][f"speaker_{top_speaker}"] = np.zeros(
                len(data_dict[k]["events"]["on_index"])
            )

        # Add speaker event label for all other speakers
        data_dict[k]["stim"]["speaker_rest"] = np.zeros(
            len(data_dict[k]["events"]["on_index"])
        )

        for i, event_idx in enumerate(data_dict[k]["events"]["on_index"]):
            speaker_label = data_dict[k]["stim"]["speaker_label"][i]
            if speaker_label in top_speakers_mapping:
                data_dict[k]["stim"][f"speaker_{speaker_label}"][i] = 1
            else:
                data_dict[k]["stim"]["speaker_rest"][i] = 1

    return data_dict


def add_sentence_position_labels(data_dict):
    """
    Add sentence position labels to data_dict
    0: first word
    1: second word
    2: third word
    3: second to last word
    4: third to last word
    5: last word
    """
    for key in data_dict.keys():
        stim_data = data_dict[key]["stim"]

        sentence_indices = stim_data["sentence_idx"]
        idx_in_sentence = stim_data["idx_in_sentence"]

        position_labels = np.zeros_like(idx_in_sentence)

        unique_sentences = np.unique(sentence_indices)

        for sentence in unique_sentences:
            sentence_mask = sentence_indices == sentence
            max_idx = np.max(idx_in_sentence[sentence_mask])

            if max_idx >= 0:
                position_labels[sentence_mask & (idx_in_sentence == 0)] = 0
            if max_idx >= 1:
                position_labels[sentence_mask & (idx_in_sentence == max_idx)] = 5
            if max_idx >= 2:
                position_labels[sentence_mask & (idx_in_sentence == 1)] = 1
            if max_idx >= 3:
                position_labels[sentence_mask & (idx_in_sentence == 2)] = 2
            if max_idx >= 4:
                position_labels[sentence_mask & (idx_in_sentence == max_idx - 1)] = 3
            if max_idx >= 5:
                position_labels[sentence_mask & (idx_in_sentence == max_idx - 2)] = 4

        stim_data["sentence_position"] = position_labels

    return data_dict


def add_sentence_position_events(data_dict):
    for k in data_dict.keys():
        for i in range(6):
            data_dict[k]["stim"][f"sentence_position_{i}"] = np.zeros(
                len(data_dict[k]["events"]["on_index"])
            )

        for i, event_idx in enumerate(data_dict[k]["events"]["on_index"]):
            sentence_position = data_dict[k]["stim"]["sentence_position"][i]
            data_dict[k]["stim"][f"sentence_position_{sentence_position}"][i] = 1
    return data_dict


def add_features(data_dict, feature_keys):
    nEventTypes = 2
    for feature_key in feature_keys:
        for k in data_dict.keys():
            events = data_dict[k]["events"]["onoff_feature"]
            n_events = len(data_dict[k]["events"]["index"])
            feature = data_dict[k]["stim"][feature_key]
            data_dict[k]["events"][feature_key] = np.zeros((n_events, nEventTypes))
            for iEventType in range(events.shape[1]):
                data_dict[k]["events"][feature_key][
                    events[:, iEventType] == 1, iEventType
                ] = feature
    return data_dict


def sum_n_laguerres(xt, *args):
    amp, tau, alpha, *w = args
    nL = len(w)
    out = np.zeros_like(xt, dtype=float)
    for iL in range(nL):
        out += w[iL] * laguerre(xt, amp, tau, alpha, xorder=iL)  # TODO FIX
    return out


if __name__ == "__main__":
    for electrode_num in range(69, 70):
        config_path = glob.glob(
            f"/nfs/zdrive/sjshim/code/brain_treebank_analysis/sub-3/electrode-"
            + str(electrode_num)
            + "/*config.json"
        )[0]
        trial_num = 0
        if os.path.exists(config_path):
            config = load_config(config_path)
            sub_id = config["sub_num"]
            chunk_buffer = config["chunk_buffer"]
            word_buffer = config["word_buffer"]
            subset_value = config["subset_value"]
            nPoints = config["nPoints"]
            print("Fitting kernel for electrode", electrode_num)
            model, learned_conv_kernel = process_and_save_kernels(
                config, train_test_split=True, test_size=0.2, rerun_model=False
            )

            train_data_dict_path = utils.generate_multiple_feature_file_name(
                **config, save_item="train_data_dict"
            )
            test_data_dict_path = utils.generate_multiple_feature_file_name(
                **config, save_item="test_data_dict"
            )
            print("Loading data dicts")
            train_data_dict = utils.load_or_create_data_dict(
                train_data_dict_path,
                sub_id,
                trial_num,
                electrode_num,
                chunk_buffer,
                word_buffer,
                subset_value,
                rerun_model=False,
            )
            test_data_dict = utils.load_or_create_data_dict(
                test_data_dict_path,
                sub_id,
                trial_num,
                electrode_num,
                chunk_buffer,
                word_buffer,
                subset_value,
                rerun_model=False,
            )

            X_test, Y_test, sentence_boundaries_test = get_conv_kernel_input(
                test_data_dict,
                nPoints,
                "lowpassed_downsampled_demeaned",
                "onoff_feature",
                word_buffer=word_buffer,
            )

            R2_onset_offset = model.score(X_test.T, Y_test)
            print("R2 onset offset", R2_onset_offset)

            train_data_dict = add_sentence_position_labels(train_data_dict)
            test_data_dict = add_sentence_position_labels(test_data_dict)

            train_data_dict = add_sentence_position_events(train_data_dict)
            test_data_dict = add_sentence_position_events(test_data_dict)
            # make features
            feature_keys = [
                # "gpt2_surprisal",
                "sentence_position_0",
                "sentence_position_1",
                "sentence_position_2",
                "sentence_position_3",
                "sentence_position_4",
                "sentence_position_5",
            ]

            train_data_dict = add_features(train_data_dict, feature_keys)
            test_data_dict = add_features(test_data_dict, feature_keys)

            laguerre_file_path = utils.generate_file_name(
                **config, save_item="laguerre_args", extension="npy"
            )

            if os.path.exists(laguerre_file_path):
                laguerre_args = np.load(laguerre_file_path)
                fit_laguerre_success = True
            else:
                fit_laguerre_success = False
            # nEventTypes = 2
            # print("Fitting laguerre parameters")
            # partial_laguerre = partial(laguerre, xorder=0)
            # laguerre_args = np.zeros((nEventTypes, 3))
            # all_args = np.zeros((nEventTypes, 8))

            # fit_laguerre_success = True
            # for iEventType in range(nEventTypes):
            #     try:
            #         popt, pcov = curve_fit(
            #             sum_n_laguerres,
            #             np.arange(nPoints),
            #             learned_conv_kernel[iEventType, :],
            #             p0=[2, 6, 5, 1, 1, 1, 1, 1],
            #             bounds=(
            #                 [
            #                     -np.inf,
            #                     -np.inf,
            #                     0,
            #                     -np.inf,
            #                     -np.inf,
            #                     -np.inf,
            #                     -np.inf,
            #                     -np.inf,
            #                 ],
            #                 [
            #                     np.inf,
            #                     np.inf,
            #                     np.inf,
            #                     np.inf,
            #                     np.inf,
            #                     np.inf,
            #                     np.inf,
            #                     np.inf,
            #                 ],
            #             ),
            #             method="trf",
            #             max_nfev=20000,
            #         )
            #         laguerre_args[iEventType, :] = popt[:3]
            #         all_args[iEventType, :] = popt
            #     except RuntimeError:
            #         fit_laguerre_success = False
            #         print("Failed to fit laguerre parameters")
            #         break

            if not fit_laguerre_success:
                print(f"Skipping electrode {electrode_num} due to failed Laguerre fit.")
                continue

            # laguerre_file_path = utils.generate_file_name(
            #     **config, save_item="laguerre_args", extension="npy"
            # )
            # np.save(laguerre_file_path, laguerre_args)

            nLaguerre = 10

            start_idx = 0
            transformations = []
            sentence_position_keys = [
                key for key in feature_keys if key.startswith("sentence_position_")
            ]

            for i, event_key in enumerate(feature_keys):
                if event_key in train_data_dict[0]["events"].keys():
                    # Get the shape for current event
                    if len(train_data_dict[0]["events"][event_key].shape) == 1:
                        current_shape = 1
                    else:
                        current_shape = train_data_dict[0]["events"][event_key].shape[1]

                    if event_key.startswith("sentence_position_"):
                        # Skip adding individual speaker keys
                        continue

                    # Create slice for this group, multiplied by nPoints
                    end_idx = (start_idx + current_shape) * nLaguerre
                    start_idx_scaled = start_idx * nLaguerre

                    # Add transformation tuple for this group
                    transformations.append(
                        (event_key, StandardScaler(), slice(start_idx_scaled, end_idx))
                    )

                    # Update start index for next group
                    start_idx = start_idx + current_shape

            if sentence_position_keys:
                start_idx_scaled = start_idx * nLaguerre
                end_idx = (start_idx + len(sentence_position_keys) * 2) * nLaguerre
                transformations.append(
                    (
                        "sentence_position_group",
                        StandardScaler(),
                        slice(start_idx_scaled, end_idx),
                    )
                )
                start_idx = start_idx + len(sentence_position_keys) * 2

            Y_avg_removed = None
            X = None
            nLaguerre = 10
            for feature_key in feature_keys:
                feature_x = None
                for k in train_data_dict.keys():
                    pair = train_data_dict[k]
                    x = generate_laguerre_features(
                        pair,
                        feature_key,
                        event_index_key="index_word-buffered",
                        resp_key="lowpassed_downsampled_demeaned",
                        laguerre_args=laguerre_args,
                        nLaguerrePoints=nPoints,
                        nLaguerre=nLaguerre,
                    )
                    if feature_x is None:
                        feature_x = x
                    else:
                        feature_x = np.hstack([feature_x, x])
                if X is None:
                    X = feature_x
                else:
                    X = np.vstack([X, feature_x])

            for k in train_data_dict.keys():
                y = train_data_dict[k]["resp"]["lowpassed_downsampled_demeaned"]
                if Y_avg_removed is None:
                    Y_avg_removed = y
                else:
                    Y_avg_removed = np.hstack([Y_avg_removed, y])

            # Create the ColumnTransformerNoStack
            ct = ColumnTransformerNoStack(transformations)
            print("X shape", X.shape)
            X = X.astype(np.float32)
            Y_avg_removed = Y_avg_removed.astype(np.float32)
            model = GroupRidgeCV(
                groups="input",
                solver_params={
                    "n_iter": 100,
                    "n_alphas_batch": 1,
                },
            )
            pipe = make_pipeline(ct, model)
            pipe.fit(X.T, Y_avg_removed)

            Y_avg_removed_test = None
            X_test = None
            nLaguerre = 10
            for feature_key in feature_keys:
                feature_x = None
                for k in test_data_dict.keys():
                    pair = test_data_dict[k]
                    x = generate_laguerre_features(
                        pair,
                        feature_key,
                        event_index_key="index_word-buffered",
                        resp_key="lowpassed_downsampled_demeaned",
                        laguerre_args=laguerre_args,
                        nLaguerrePoints=nPoints,
                        nLaguerre=nLaguerre,
                    )
                    if feature_x is None:
                        feature_x = x
                    else:
                        feature_x = np.hstack([feature_x, x])
                if X_test is None:
                    X_test = feature_x
                else:
                    X_test = np.vstack([X_test, feature_x])

            for k in test_data_dict.keys():
                y = test_data_dict[k]["resp"]["lowpassed_downsampled_demeaned"]
                if Y_avg_removed_test is None:
                    Y_avg_removed_test = y
                else:
                    Y_avg_removed_test = np.hstack([Y_avg_removed_test, y])

            pipe_file_name = utils.generate_file_name(
                **config, save_item="group_ridge_model", extension="pkl"
            )
            train_X_file_name = utils.generate_file_name(
                **config, save_item="train_X", extension="npy"
            )
            train_Y_file_name = utils.generate_file_name(
                **config, save_item="train_Y", extension="npy"
            )
            test_X_file_name = utils.generate_file_name(
                **config, save_item="test_X", extension="npy"
            )
            test_Y_file_name = utils.generate_file_name(
                **config, save_item="test_Y", extension="npy"
            )

            # Save the pipeline
            joblib.dump(pipe, pipe_file_name)
            print("Model saved to", pipe_file_name)

            # Save matrices
            np.save(train_X_file_name, X)
            np.save(train_Y_file_name, Y_avg_removed)
            np.save(test_X_file_name, X_test)
            np.save(test_Y_file_name, Y_avg_removed_test)
