import numpy as np
import h5py
import pandas as pd
import utils
from soundsig.signal import bandpass_filter, highpass_filter
from soundsig.sound import temporal_envelope
from scipy import signal, stats
from scipy.signal import resample
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def resample_ecog(data, sample_rate, resample_rate):
    lensound = len(data)
    t = (np.array(range(lensound), dtype=float)) / sample_rate
    lenresampled = int(round(float(lensound) * resample_rate / sample_rate))
    (srectresampled, tresampled) = resample(
        data, lenresampled, t=t, axis=0, window=None
    )
    return (srectresampled, tresampled)


def get_data_dict_info(
    data_df,
    chunk_buffer,
    word_buffer,
    samp_frequency,
    electrode_data,
    fs_conversion_factor,
    electrode_data_lowpassed_downsampled,
    electrode_data_highpassed_rectified_lowpassed_downsampled,
):
    data_dict = {}
    resp_end_idx_adding = 0
    for chunk_id, chunk_df in data_df.groupby("chunk_idx"):
        # Get word count in chunk
        word_count = len(chunk_df)

        chunk_onset_idx = chunk_df["est_idx"].min()
        chunk_offset_idx = chunk_df["est_end_idx"].max()

        # Get the neural data for the sentence
        raw_start_idx = int(max(0, chunk_onset_idx - word_buffer * samp_frequency))
        raw_end_idx = int(
            min(
                chunk_offset_idx + chunk_buffer * samp_frequency,
                len(electrode_data),
            )
        )
        resp_start_idx = int(raw_start_idx * fs_conversion_factor)
        resp_end_idx = int(raw_end_idx * fs_conversion_factor)

        # neural_data_lfp = electrode_data_lfped[raw_start_idx:raw_end_idx]
        # neural_data_hfp = electrode_data_hfped[raw_start_idx:raw_end_idx]
        neural_data_lowpassed_downsampled = electrode_data_lowpassed_downsampled[
            resp_start_idx:resp_end_idx
        ]
        neural_data_highpassed_rectified_lowpassed_downsampled = (
            electrode_data_highpassed_rectified_lowpassed_downsampled[
                resp_start_idx:resp_end_idx
            ]
        )

        resp_end_idx_adding = resp_end_idx - resp_start_idx
        resp_end_idx_adding = resp_end_idx_adding + resp_end_idx - resp_start_idx
        # Get the events for the sentence
        onset_events_ind = (chunk_df["est_idx"].values * fs_conversion_factor).astype(
            int
        ) - resp_start_idx
        offset_events_ind = (
            chunk_df["est_end_idx"].values * fs_conversion_factor
        ).astype(int) - resp_start_idx

        onset_features = np.array([1, 0] * len(onset_events_ind)).reshape(-1, 2)
        offset_features = np.array([0, 1] * len(offset_events_ind)).reshape(-1, 2)

        events = np.concatenate([onset_events_ind, offset_events_ind])
        on_events = np.concatenate([onset_events_ind])
        off_events = np.concatenate([offset_events_ind])

        features = np.concatenate([onset_features, offset_features])
        on_features = np.concatenate([onset_features])
        off_features = np.concatenate([offset_features])

        mel_values = np.array(chunk_df["mel"].apply(ast.literal_eval).tolist())

        # Create duplicate mel values for onset and offset events
        mel_values_onset = mel_values.copy()
        mel_values_offset = mel_values.copy()

        # Combine all events data
        combined_data = []
        for i in range(len(onset_events_ind)):
            # Add onset event
            combined_data.append(
                {
                    "time": onset_events_ind[i],
                    "is_onset": True,
                    "features": [1, 0],
                    "mel": mel_values_onset[i],
                }
            )
            # Add offset event
            combined_data.append(
                {
                    "time": offset_events_ind[i],
                    "is_onset": False,
                    "features": [0, 1],
                    "mel": mel_values_offset[i],
                }
            )

        # Sort by event time
        combined_data.sort(key=lambda x: x["time"])

        # Extract sorted data
        events = np.array([x["time"] for x in combined_data])
        features = np.array([x["features"] for x in combined_data])
        mel_values_aligned = np.array([x["mel"] for x in combined_data])

        # Separate onset and offset events while maintaining their original order
        on_events = np.array([x["time"] for x in combined_data if x["is_onset"]])
        on_features = np.array([x["features"] for x in combined_data if x["is_onset"]])
        on_mel_values = np.array([x["mel"] for x in combined_data if x["is_onset"]])

        off_events = np.array([x["time"] for x in combined_data if not x["is_onset"]])
        off_features = np.array(
            [x["features"] for x in combined_data if not x["is_onset"]]
        )
        off_mel_values = np.array(
            [x["mel"] for x in combined_data if not x["is_onset"]]
        )

        data_dict[chunk_id] = {}
        data_dict[chunk_id]["stim"] = {}
        data_dict[chunk_id]["stim"]["word_idx"] = chunk_df["word_idx"].values
        data_dict[chunk_id]["stim"]["mel"] = mel_values  # Original mel values
        data_dict[chunk_id]["stim"][
            "mel_aligned"
        ] = mel_values_aligned  # Aligned with all events
        data_dict[chunk_id]["stim"][
            "mel_onset"
        ] = on_mel_values  # Aligned with onset events
        data_dict[chunk_id]["stim"][
            "mel_offset"
        ] = off_mel_values  # Aligned with offset events

        data_dict[chunk_id]["resp"] = {}
        data_dict[chunk_id]["resp"][
            "lowpassed_downsampled"
        ] = neural_data_lowpassed_downsampled
        data_dict[chunk_id]["resp"]["lowpassed_downsampled_demeaned"] = (
            neural_data_lowpassed_downsampled
            - np.mean(neural_data_lowpassed_downsampled)
        )
        data_dict[chunk_id]["resp"][
            "highpassed_rectified_lowpassed_downsampled_demeaned"
        ] = neural_data_highpassed_rectified_lowpassed_downsampled - np.mean(
            neural_data_highpassed_rectified_lowpassed_downsampled
        )
        data_dict[chunk_id]["resp"]["pre_filt"] = electrode_data[
            raw_start_idx:raw_end_idx
        ]
        data_dict[chunk_id]["resp"]["sentence_end_idx"] = resp_end_idx_adding

        data_dict[chunk_id]["events"] = {}
        data_dict[chunk_id]["events"]["index"] = events.astype(int)
        data_dict[chunk_id]["events"]["onoff_feature"] = features
        data_dict[chunk_id]["events"]["on_index"] = on_events.astype(int)
        data_dict[chunk_id]["events"]["on_feature"] = on_features
        data_dict[chunk_id]["events"]["off_index"] = off_events.astype(int)
        data_dict[chunk_id]["events"]["off_feature"] = off_features
        data_dict[chunk_id]["events"]["word_count"] = word_count
        data_dict[chunk_id]["stim"]["rms"] = chunk_df["rms"].values
        data_dict[chunk_id]["stim"]["pitch"] = chunk_df["pitch"].values
        data_dict[chunk_id]["stim"]["magnitude"] = chunk_df["magnitude"].values
        data_dict[chunk_id]["stim"]["speaker"] = chunk_df["speaker"].values
        data_dict[chunk_id]["stim"]["gpt2_surprisal"] = chunk_df[
            "gpt2_surprisal_zscore"
        ].values
        data_dict[chunk_id]["stim"]["idx_in_sentence"] = chunk_df[
            "idx_in_sentence"
        ].values
        data_dict[chunk_id]["stim"]["sentence_idx"] = chunk_df["sentence_idx"].values

    return data_dict


def create_data_dict(
    sub_id,
    trial_num,
    electrode_num,
    chunk_buffer=1,
    word_buffer=0.2,
    subset=None,
    train_test_split=False,
    test_size=0.2,
):
    """
    Create a dictionary with the neural data, event indices, and on/off features.
    Chunks are separated by at least 1 second in between word onsets.

    Inputs:
    sub_id - subject id
    electrode_num - electrode number
    chunk_buffer - buffer after each chunk in seconds
    subset - randomly select subset number of sentences
    train_test_split - whether to split the data into training and testing sets
    test_size - proportion of data (in words) to use for testing

    Output:
    A dictionary with the neural data, event indices, and on/off features.
    If train_test_split is True, returns two dictionaries for training and testing data.
    """
    # Path to the data
    data_path = "/nfs/zdrive/sjshim/brain_treebank_data/"
    samp_frequency = 2048
    resp_frequency = 100
    trial_num = str(trial_num)
    # Load the neural data
    neural_data_file = data_path + f"sub_{sub_id}_trial00{trial_num}.h5"
    h5f = h5py.File(neural_data_file, "r")
    data = h5f["data"]

    # Extract the data from the dataset
    electrode_data = data[f"electrode_{electrode_num}"]
    for f in [60, 120, 180, 240, 300, 360]:
        electrode_data = utils.notch_filter(electrode_data, f)

    # # Bands we are interested in
    # lfp_band = [1, 25]
    # hfp_band = [25, 60]

    # # Filter the data
    # electrode_data_lfped = bandpass_filter(
    #     electrode_data, samp_frequency, *lfp_band, filter_order=20
    # )
    # electrode_data_hfped = bandpass_filter(
    #     electrode_data, samp_frequency, *hfp_band, filter_order=20
    # )

    electrode_data_lowpassed = bandpass_filter(
        electrode_data, samp_frequency, 1, 50, filter_order=10
    )
    electrode_data_lowpassed_downsampled, _ = resample_ecog(
        electrode_data_lowpassed, samp_frequency, resp_frequency
    )

    electrode_data_highpassed = highpass_filter(
        electrode_data, samp_frequency, 50, filter_order=10
    )
    electrode_data_highpassed_rectified = np.abs(
        electrode_data_highpassed - np.mean(electrode_data_highpassed)
    )
    electrode_data_highpassed_rectified_lowpassed = bandpass_filter(
        electrode_data_highpassed_rectified, samp_frequency, 1, 50, filter_order=10
    )
    electrode_data_highpassed_rectified_lowpassed_downsampled, _ = resample_ecog(
        electrode_data_highpassed_rectified_lowpassed, samp_frequency, resp_frequency
    )

    # electrode_data_lfped_downsampled, _ = resample_ecog(
    #     electrode_data_lfped, samp_frequency, resp_frequency
    # )
    # electrode_data_hfped_downsampled, _ = resample_ecog(
    #     electrode_data_hfped, samp_frequency, resp_frequency
    # )

    # Downsample the data
    # electrode_data_lfped_downsampled, _ = temporal_envelope(
    #     electrode_data_lfped,
    #     samp_frequency,
    #     resample_rate=resp_frequency,
    #     cutoff_freq=None,
    # )
    # electrode_data_hfped_downsampled, _ = temporal_envelope(
    #     electrode_data_hfped,
    #     samp_frequency,
    #     resample_rate=resp_frequency,
    #     cutoff_freq=None,
    # )
    # Get the conversion factor to go from the original data to the downsampled data
    fs_conversion_factor = resp_frequency / samp_frequency

    # Get features data
    features_file = data_path + f"sub_{sub_id}_trial00{trial_num}_words.csv"
    features_df = pd.read_csv(features_file)

    # Z-score features_df['gpt2_surprisal']
    features_df["gpt2_surprisal_zscore"] = stats.zscore(features_df["gpt2_surprisal"])

    if subset is not None:
        assert subset <= len(features_df["sentence_idx"].unique())

        # Take first subset number of sentences
        features_df = features_df[features_df["sentence_idx"] < subset]

    df = features_df.copy()
    starts = df.start.values
    ends = df.end.values
    time_between_words = starts[1:] - ends[:-1]
    time_between_words = np.concatenate([[0], time_between_words])
    chunk_start = np.where(time_between_words > 1)[0]
    chunk_start = np.concatenate([[0], chunk_start])
    chunk_end = np.where(time_between_words > 1)[0] - 1
    chunk_end = np.concatenate([chunk_end, [len(time_between_words) - 1]])

    # add chunk_idx to df
    chunk_idx = np.zeros(len(df))
    for i, (start, end) in enumerate(zip(chunk_start, chunk_end)):
        chunk_idx[start : end + 1] = i
    chunk_idx = chunk_idx.astype(int)
    features_df["chunk_idx"] = chunk_idx

    # Divide into training and testing sets
    if train_test_split:
        chunk_word_count = 0
        subset_chunk_idx = []
        np.random.seed(0)
        while chunk_word_count < len(features_df) * test_size:
            chunk_id = np.random.choice(features_df["chunk_idx"].unique())
            if chunk_id not in subset_chunk_idx:
                subset_chunk_idx.append(chunk_id)
                chunk_word_count += len(
                    features_df[features_df["chunk_idx"] == chunk_id]
                )
        print(f"Number of words in test set: {chunk_word_count}")
        test_df = features_df[features_df["chunk_idx"].isin(subset_chunk_idx)]
        train_df = features_df[~features_df["chunk_idx"].isin(subset_chunk_idx)]
        train_data_dict = get_data_dict_info(
            train_df,
            chunk_buffer,
            word_buffer,
            samp_frequency,
            electrode_data,
            fs_conversion_factor,
            electrode_data_lowpassed_downsampled,
            electrode_data_highpassed_rectified_lowpassed_downsampled,
        )
        test_data_dict = get_data_dict_info(
            test_df,
            chunk_buffer,
            word_buffer,
            samp_frequency,
            electrode_data,
            fs_conversion_factor,
            electrode_data_lowpassed_downsampled,
            electrode_data_highpassed_rectified_lowpassed_downsampled,
        )
        return train_data_dict, test_data_dict

    else:
        data_dict = get_data_dict_info(
            features_df,
            chunk_buffer,
            word_buffer,
            samp_frequency,
            electrode_data,
            fs_conversion_factor,
            electrode_data_lowpassed_downsampled,
            electrode_data_highpassed_rectified_lowpassed_downsampled,
        )
        return data_dict
