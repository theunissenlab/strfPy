Conda environment used is in requirements.yml. Will also have to install:
- himalaya: https://github.com/gallantlab/himalaya
- strfPy: https://github.com/theunissenlab/strfpy
- soundsig: https://github.com/theunissenlab/soundsig

Data available:
https://braintreebank.dev/
Run their quickstart notebook to get word.csv files for each of the movies that include the est_idx and est_end_idx columns

Code:
- preprocess_data.py: functions to preprocess neural data and create data_dict that has event and feature info. uses word.csv created from above
- utils.py: util functions for preprocess_data.py and analysis.py
- create_config_files.py: creates config files that are used to run analysis for each subject and electrode
- analysis.py: reads existing config files to run analysis portion 
- model_multiple_features_notebook.ipynb: notebook format of analysis.py