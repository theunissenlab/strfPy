"""
Parallel version of runFile.ipynb: runs process_unit() on all the good units of an nwb file using several worker processes.

Each unit's result is saved to its own pickle file in unit_dir as soon as it is done, so if the run is interrupted
(crash, Ctrl-C) running the same command again only fits the units that are missing.  When all units are done the results
are gathered in a single DataFrame and saved to out_pkl (same format as resultsDF in runFile.ipynb).

Memory: the seg+ID fits store one (nb x nb) auto-covariance matrix per stimulus, where nb = 25 Laguerre x 2 events x nPCs.
With nPCs = 100 a unit peaks at ~17 GB, so 6 workers use ~100 GB.  The memory grows as nPCs^2: reduce the number of workers
for larger nPCs (e.g. nPCs = 200 needs ~4x more memory per unit).

Usage (in the strf environment), e.g. the 100 PC run:
    ~/opt/anaconda3/envs/strf/bin/python runFileParallel.py --nPCs 100 --segmentBuffer 30 --nWorkers 6 \
        --out "/Users/frederictheunissen/Work Data/OperantEphys/NWB_Files/HpiPur2667F_site02_240905_072851_pb_op_ks4_lat_250215_100PCs.pkl"
To keep it running after closing the terminal, start it with nohup ... > run.log 2>&1 &
"""
import os

# Number of BLAS/LAPACK threads per worker.  This must be set before numpy is imported.  With spawn (the macOS default)
# each worker re-imports this file, so the setting also applies to the workers.
THREADS_PER_WORKER = os.environ.get('THREADS_PER_WORKER', '5')
for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(var, THREADS_PER_WORKER)

import argparse
import contextlib
import io
import pickle
import tempfile
import time
import warnings

import matplotlib
matplotlib.use('Agg')   # process_unit makes figures - no display in the workers
import pandas as pd


def run_one_unit(args):
    """Fits all the models for one unit and saves the result in unit_dir/unit_<unit>.pkl"""
    nwb_file, unit, segmentBuffer, nPCs, unit_dir, verbose = args
    warnings.filterwarnings('ignore')

    # trnDirectFit writes temporary files in tempdir/<nwb identifier>, which is shared by all the units of a file:
    # give each unit its own tempdir so that workers do not overwrite each other's files.
    tempfile.tempdir = tempfile.mkdtemp(prefix='strfpy_u%d_' % unit)

    from strfpy.calcSegmentedModel import process_unit

    t0 = time.time()
    if verbose:
        result = process_unit(nwb_file, unit, segmentBuffer=segmentBuffer, nPCs=nPCs)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = process_unit(nwb_file, unit, segmentBuffer=segmentBuffer, nPCs=nPCs)

    # result is None when the unit has no valid trials: the None is saved anyway so that a resumed run does not try this unit again
    with open(os.path.join(unit_dir, 'unit_%d.pkl' % unit), 'wb') as f:
        pickle.dump(result, f)
    if result is None:
        print('unit %d skipped: no valid trials' % unit, flush=True)
    else:
        print('unit %d done in %.0f min: r2segIDModelLG %.3f r2STRF %.3f'
              % (unit, (time.time() - t0)/60, result['r2segIDModelLG'], result['r2STRF']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nwb_file', default='/Users/frederictheunissen/Work Data/OperantEphys/NWB_Files/HpiPur2667F_site02_240905_072851_pb_op_ks4_lat_250215.nwb')
    parser.add_argument('--out', required=True, help='pickle file for the results DataFrame')
    parser.add_argument('--unit_dir', default=None, help='folder for the per-unit results (default: <out without .pkl>_units)')
    parser.add_argument('--nPCs', type=int, default=50, help='number of PCs of the seg+ID features')
    parser.add_argument('--segmentBuffer', type=int, default=30, help='ms of stimulus before each event in the seg+ID windows')
    parser.add_argument('--nWorkers', type=int, default=6, help='number of units fitted in parallel')
    parser.add_argument('--verbose', action='store_true', help='show the output of process_unit')
    args = parser.parse_args()

    unit_dir = args.unit_dir if args.unit_dir is not None else os.path.splitext(args.out)[0] + '_units'
    os.makedirs(unit_dir, exist_ok=True)

    # The good units, as in runFile.ipynb
    import pynwb
    with pynwb.NWBHDF5IO(args.nwb_file, mode='r') as nwb_io:
        units = nwb_io.read().units.to_dataframe()
    good_units = list(units[units.group == 'good'].index)

    todo = [u for u in good_units if not os.path.exists(os.path.join(unit_dir, 'unit_%d.pkl' % u))]
    print('%d good units, %d left to run with %d workers x %s threads (nPCs = %d, segmentBuffer = %d)'
          % (len(good_units), len(todo), args.nWorkers, THREADS_PER_WORKER, args.nPCs, args.segmentBuffer), flush=True)

    if len(todo) > 0:
        from multiprocessing import Pool
        with Pool(args.nWorkers) as pool:
            pool.map(run_one_unit, [(args.nwb_file, u, args.segmentBuffer, args.nPCs, unit_dir, args.verbose) for u in todo], chunksize=1)

    results = []
    skipped = []
    for u in good_units:
        with open(os.path.join(unit_dir, 'unit_%d.pkl' % u), 'rb') as f:
            result = pickle.load(f)
        if result is None:
            skipped.append(u)
        else:
            results.append(result)
    pd.DataFrame(results).to_pickle(args.out)
    print('Saved %d units to %s' % (len(results), args.out), flush=True)
    if len(skipped) > 0:
        print('Skipped %d units with no valid trials: %s' % (len(skipped), skipped), flush=True)


if __name__ == '__main__':
    main()
