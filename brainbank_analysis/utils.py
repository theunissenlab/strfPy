from scipy import signal
import os
import pickle
import preprocess_data


def notch_filter(data, freq, sampling_freq=2048, Q=30):
    """
    Run notch filtering on a given signal.

    Inputs:
    data - pre-filtered signals
    freq - frequency band to filter out

    Output:
    The original data without the given frequency componant.
    """
    w0 = freq / (sampling_freq / 2)  # Normalized Frequency
    b, a = signal.iirnotch(w0, Q)  # Design notch filter
    y = signal.lfilter(b, a, data, axis=0)  # remove frequancy
    return y


def generate_multiple_feature_file_name(
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
    train_test_split=False,
    extension="pkl",
    save_item="data_dict",
):
    """
    Generate a standardized file name for saving models, kernels, or plots.
    """
    os.makedirs(
        os.path.join(
            base_dir, "sub-" + str(sub_num), "electrode-" + str(electrode_num)
        ),
        exist_ok=True,
    )
    save_dir = os.path.join(
        base_dir, "sub-" + str(sub_num), "electrode-" + str(electrode_num)
    )

    file_name = (
        f"events-{event_key}_{resp_key}_nPoints-{nPoints}_word-buffer_{word_buffer}"
    )

    if subset_value is not None and subset_value > 0:
        file_name += f"_subset-{subset_value}"

    if train_test_split:
        file_name += "_train-test-split"
    file_name += f"_{save_item}"

    return os.path.join(
        base_dir,
        f"sub-{sub_num}",
        f"electrode-{electrode_num}",
        f"{file_name}.{extension}",
    )


def generate_file_name(
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
    extension="pkl",
    save_item="data_dict",
):
    """
    Generate a standardized file name for saving models, kernels, or plots.
    """
    os.makedirs(
        os.path.join(
            base_dir, "sub-" + str(sub_num), "electrode-" + str(electrode_num)
        ),
        exist_ok=True,
    )
    save_dir = os.path.join(
        base_dir, "sub-" + str(sub_num), "electrode-" + str(electrode_num)
    )
    file_name = (
        f"events-{event_key}_{resp_key}_nPoints-{nPoints}_word-buffer_{word_buffer}"
    )

    if subset_value is not None and subset_value > 0:
        file_name += f"_subset-{subset_value}"

    file_name += f"_{save_item}"

    return os.path.join(
        base_dir,
        f"sub-{sub_num}",
        f"electrode-{electrode_num}",
        f"{file_name}.{extension}",
    )


def load_or_save_pickle(file_path, create_func, *args, **kwargs):
    """
    Load a pickle file if it exists, otherwise create it using create_func and save it.
    """
    if os.path.exists(file_path):
        print(f"Loading from {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Creating new file at {file_path}")
        data = create_func(*args, **kwargs)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        return data


def load_or_create_data_dict(
    file_path,
    sub_num,
    trial_num,
    electrode_num,
    chunk_buffer,
    word_buffer,
    subset_value,
    rerun_model=False,
):
    if subset_value == None:
        print("no subset")
        if os.path.exists(file_path) and not rerun_model:
            print("loading existing data_dict")
            with open(file_path, "rb") as f:
                data_dict = pickle.load(f)
        else:
            print("creating new data_dict")
            data_dict = preprocess_data.create_data_dict(
                sub_num,
                trial_num,
                electrode_num,
                chunk_buffer=chunk_buffer,
                word_buffer=word_buffer,
            )
            data_dict, _ = add_word_buffer(data_dict, "index", word_buffer)

            with open(file_path, "wb") as f:
                pickle.dump(data_dict, f)
    else:
        if os.path.exists(file_path) and not rerun_model:
            with open(file_path, "rb") as f:
                data_dict = pickle.load(f)
        else:
            data_dict = preprocess_data.create_data_dict(
                sub_num,
                trial_num,
                electrode_num,
                chunk_buffer,
                word_buffer,
                subset_value,
            )
            data_dict, _ = add_word_buffer(data_dict, "index", word_buffer)
            with open(file_path, "wb") as f:
                pickle.dump(data_dict, f)
    return data_dict


def load_or_create_data_dict_train_test(
    train_file_path,
    test_file_path,
    sub_num,
    trial_num,
    electrode_num,
    chunk_buffer,
    word_buffer,
    subset_value,
    train_test_split=True,
    test_size=0.2,
    rerun_model=False,
):
    if subset_value == None:
        print("no subset")
        if (
            os.path.exists(train_file_path)
            and os.path.exists(test_file_path)
            and not rerun_model
        ):
            with open(train_file_path, "rb") as f:
                train_data_dict = pickle.load(f)
            with open(test_file_path, "rb") as f:
                test_data_dict = pickle.load(f)
        else:
            train_data_dict, test_data_dict = preprocess_data.create_data_dict(
                sub_num,
                trial_num,
                electrode_num,
                chunk_buffer=chunk_buffer,
                word_buffer=word_buffer,
                train_test_split=train_test_split,
                test_size=test_size,
            )
            train_data_dict, _ = add_word_buffer(train_data_dict, "index", word_buffer)
            test_data_dict, _ = add_word_buffer(test_data_dict, "index", word_buffer)
            with open(train_file_path, "wb") as f:
                pickle.dump(train_data_dict, f)
            with open(test_file_path, "wb") as f:
                pickle.dump(test_data_dict, f)
    else:
        if (
            os.path.exists(train_file_path)
            and os.path.exists(test_file_path)
            and not rerun_model
        ):
            with open(train_file_path, "rb") as f:
                train_data_dict = pickle.load(f)
            with open(test_file_path, "rb") as f:
                test_data_dict = pickle.load(f)
        else:
            train_data_dict, test_data_dict = preprocess_data.create_data_dict(
                sub_num,
                trial_num,
                electrode_num,
                chunk_buffer=chunk_buffer,
                word_buffer=word_buffer,
                subset=subset_value,
                train_test_split=train_test_split,
                test_size=test_size,
            )
            with open(train_file_path, "wb") as f:
                pickle.dump(train_data_dict, f)
            with open(test_file_path, "wb") as f:
                pickle.dump(test_data_dict, f)
    return train_data_dict, test_data_dict


def add_word_buffer(data_dict, event_index_key, word_buffer):
    """
    Add a word buffer to the event index key

    Parameters
    ----------
    data_dict : dict
        Dictionary of data
    event_index_key : str
        Key for the event index
    word_buffer : float
        Second buffer to add to the event index key"""

    word_buffer = word_buffer * 100
    buffered_event_index_key = f"{event_index_key}_word-buffered"
    for key in data_dict.keys():
        data_dict[key]["events"][buffered_event_index_key] = (
            data_dict[key]["events"][event_index_key] - word_buffer
        ).astype(int)
    return data_dict, buffered_event_index_key
