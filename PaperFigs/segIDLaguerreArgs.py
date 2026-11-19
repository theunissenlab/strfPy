"""
Precomputes what summaryStats.ipynb needs to rebuild the time-varying receptive fields (RFs)
of the segmentation + identification (SEG+ID) Laguerre models saved by process_unit():

    - the Laguerre (tau, alpha) parameters of each unit, obtained by refitting the saved
      segmentation kernel with fit_kernel_LG() exactly as in process_unit(),
    - the map from nwb file name to nwb identifier (the model directories are named by identifier),
    - the frequency axis of the spectrogram used for the spect_windows feature.

summaryStats.ipynb runs in an environment without pynwb/strfpy, so this script must be run in
the strf environment:
    ~/opt/anaconda3/envs/strf/bin/python PaperFigs/segIDLaguerreArgs.py
"""
import os
import pickle
import warnings

import h5py
import numpy as np

from strfpy.calcSegmentedModel import fit_kernel_LG
from strfpy.timeFreq import timefreq_raw

warnings.filterwarnings('ignore')

nwb_dir = '/Volumes/condor/lthomas/Projects/Thesis_analysis/nwbs'
model_dir = '/Volumes/condor/lthomas/Projects/Thesis_analysis/MODELS_main_261024'
out_file = '/Users/frederictheunissen/Desktop/segID_laguerre_args.pkl'

nPoints = 150   # Same as in process_unit()

# 1. nwb file name -> nwb identifier
nwb_identifiers = {}
stim_fs = None
for f in sorted(os.listdir(nwb_dir)):
    if not f.endswith('.nwb'):
        continue
    with h5py.File(os.path.join(nwb_dir, f), 'r') as h:
        identifier = h.attrs.get('identifier')
        if identifier is None:
            identifier = h['identifier'][()]
        nwb_identifiers[f[:-4]] = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
        if stim_fs is None:
            stim_name = list(h['stimulus/presentation'].keys())[0]
            stim_fs = float(h['stimulus/presentation'][stim_name]['starting_time'].attrs['rate'])

# 2. Frequency axis - same parameters as preprocSound.preprocess_sound_nwb()
stim_params = {'fband': 120, 'nstd': 6, 'high_freq': 8000, 'low_freq': 250, 'log': 1, 'stim_rate': 1000}
tfrep = timefreq_raw(np.random.randn(int(stim_fs)), stim_fs, 'ft', stim_params)
freqs = np.asarray(tfrep['f'])

# 3. Laguerre parameters for every saved unit
laguerre_args = {}
for identifier in nwb_identifiers.values():
    unit_dirs = os.listdir(os.path.join(model_dir, identifier)) if os.path.isdir(os.path.join(model_dir, identifier)) else []
    for unit_dir in unit_dirs:
        unit = int(unit_dir.split('_')[-1])
        seg_file = os.path.join(model_dir, identifier, unit_dir, '%d_segModel.pkl' % unit)
        if not os.path.exists(seg_file):
            continue
        with open(seg_file, 'rb') as f:
            segModel = pickle.load(f)
        learned_conv_kernel = segModel['weights'].reshape(2, nPoints)
        laguerre_args[(identifier, unit)] = fit_kernel_LG(learned_conv_kernel, nPoints, nD=2)
    print('%s: %d units' % (identifier, len(unit_dirs)), flush=True)

with open(out_file, 'wb') as f:
    pickle.dump({'nwb_identifiers': nwb_identifiers, 'freqs': freqs, 'laguerre_args': laguerre_args,
                 'model_dir': model_dir, 'nPoints': nPoints}, f)
print('Saved %d units to %s' % (len(laguerre_args), out_file))
