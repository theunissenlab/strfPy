import json
import os
from utils import generate_file_name


def save_config(config, config_path="config.json"):
    """Save configuration to a JSON file."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


base_dir = "/nfs/zdrive/sjshim/code/brain_treebank_analysis"
sub_num = 3
trial_num = 0
resp_key = "lowpassed_downsampled_demeaned"
event_key = "onoff_feature"
nPoints = 120
word_buffer = 0.2
chunk_buffer = 1
subset_value = None

for electrode_num in range(0, 120):
    config_path = generate_file_name(
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
        extension="json",
        save_item="config",
    )
    print(config_path)

    # if not os.path.exists(config_path):
    default_config = {
        "base_dir": base_dir,
        "sub_num": sub_num,
        "trial_num": trial_num,
        "electrode_num": electrode_num,
        "resp_key": resp_key,
        "event_key": event_key,
        "nPoints": nPoints,
        "word_buffer": word_buffer,
        "chunk_buffer": chunk_buffer,
        "subset_value": subset_value,
    }
    save_config(default_config, config_path)
