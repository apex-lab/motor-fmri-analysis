from scipy.interpolate import interp1d
from mne.filter import filter_data
from bids import BIDSLayout
import numpy as np
import pandas as pd
import requests
import pickle
import os
import argparse

from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.surface import load_surf_data
from nilearn.signal import clean

from util.glove import load_joint_angles
from util.bids import DataSink
from util.myosuite import (
    get_joints_and_muscles,
    joints_to_myo_obs,
    get_activations
    )

DATA_DIR = 'bids_temp'
ENV_NAME = 'myoHandPoseRandom-v0'
POLICY = 'best_policy.pickle'
POLICY_URL = '/'.join([
    'https://github.com',
    'facebookresearch', 'myosuite',
    'raw', 'main', 'myosuite',
    'agents', 'baslines_NPG',
    'myoHandPoseRandom-v0', '2022-02-27_11-03-32',
    '11_env=myoHandPoseRandom-v0,seed=3',
    'iterations', POLICY
    ])

def _downsample_to_timestamps(layout, t, activations, tr_timestamps):
    activations_filt = filter_data(
        activations.T, # time must be last dim
        1/np.diff(t).mean(), # sampling rate
        l_freq = None,
        h_freq = layout.get_tr(task = 'motor')**-1 / 2.
    )
    interp_func = interp1d(x = t, y = activations_filt.T, axis = 0)
    acts_interp = interp_func(tr_timestamps)
    return acts_interp

def process_motion(layout, sub, run, n_volumes):
    '''
    Computes activations of trained myosuite model for each
    (joint position, target position) subject was in during a motor task
    run. Then, downsamples the model activations to just the TR times, so
    saved output can be used to fit encoding model in another script.

    Parameters
    ----------
    layout : bids.BIDSLayout
    sub : str
    run : int
    n_volumes : int
        the number of volumes in correesponding BOLD run
    '''
    ## get some info from moysuite environment
    joints, muscles = get_joints_and_muscles(ENV_NAME)
    # and trained neural network model
    r = requests.get(POLICY_URL)
    with open(POLICY, 'wb') as f:
        f.write(r.content)
    pi = pickle.load(open(POLICY, 'rb'))

    ## get the timestamp for each TR in run
    TR = layout.get_tr(task = 'motor')
    tr_timestamps = np.linspace(0, n_volumes * TR, n_volumes)
    # adjust for slice timing correction
    tr_timestamps += TR / 2

    ## load corresponding behavioral data
    events = layout.get(
        subject = sub,
        task = 'motor',
        run = run,
        suffix = 'events'
    )[0].get_df()
    joint_angles, t, sfreq = load_joint_angles(layout, sub, run, True)

    ## convert to expected format for observation in myosuite environment
    obs = joints_to_myo_obs(joint_angles, t, events, ENV_NAME)

    # get activations from neural network model
    activations = get_activations(pi.model, obs.to_numpy())
    # record how many units are in each layer
    units_per_layer = [layer.shape[1] for layer in activations]
    assert(units_per_layer[0] == obs.shape[1])
    assert(units_per_layer[-1] == len(muscles))
    # then combine layers into one array
    activations = np.concatenate(activations, axis = 1)

    # downsample activations to TR times
    acts_interp = _downsample_to_timestamps(
        layout, t, activations, tr_timestamps
    )
    # and put into dataframe
    cols = []
    cols += obs.columns.to_list()
    for i, n_units in enumerate(units_per_layer[1:-1]):
        cols += ['unit%d_layer%d'%(j, i + 1) for j in range(n_units)]
    cols += [m + '_musc' for m in muscles]
    df = pd.DataFrame(
        acts_interp,
        columns = cols
    )
    assert(df.shape[1] == sum(units_per_layer))

    # finally, save for posterity
    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'myoprep'
    )
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        run = run,
        desc = 'downsamp',
        datatype = 'motion',
        suffix = 'activations',
        extension = '.tsv'
    )
    df.to_csv(fpath, sep = '\t', index = False)
    fpath = fpath.replace('downsamp', 'fullsamp')
    fpath = fpath.replace('activations', 'observations')
    obs_cropped = obs[tr_timestamps[0]:tr_timestamps[-1]]
    times = obs_cropped.index.values
    obs_cropped.insert(0, 'time', times)
    obs_cropped.to_csv(fpath, sep = '\t', index = False)


def process_bold(layout, sub, run):
    '''
    Cleans BOLD data via:
    - detrending
    - removal of compcor (and motion) confound regressors
    - z-standardization of signal
    and then saves as a .npy file for later use.

    Parameters
    ----------
    layout : bids.BIDSLayout
    sub : str
    run : int

    Returns
    ----------
    n_volumes : int
        the number of volumes in BOLD run
    '''
    ## load fMRIPrepped BOLD data on fsaverage surface
    left_f, _, right_f, sidecar_f = layout.get(
        subject = sub,
        task = 'motor',
        run = run,
        space = 'fsaverage',
        suffix = 'bold',
        scope = 'fMRIPrep'
    )
    left = load_surf_data(left_f.path)
    right = load_surf_data(right_f.path)
    bold = np.concatenate([left, right], axis = 0)
    confounds = load_confounds_strategy(left_f.path, 'compcor')[0]

    bold_clean = clean(
        signals = bold.T, # n_samples x n_sources
        confounds = confounds,
        t_r = layout.get_tr(task = 'motor'),
        standardize = 'zscore_sample'
    )

    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'myoprep'
    )
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        run = run,
        desc = 'clean',
        datatype = 'func',
        suffix = 'bold',
        extension = '.npy'
    )
    np.save(fpath, bold_clean, allow_pickle = False)
    return bold_clean.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    args = parser.parse_args()
    layout = BIDSLayout(DATA_DIR, derivatives = True)
    runs = layout.get_runs(task = 'motor', subject = args.sub)
    for run in runs:
        n_volumes = process_bold(layout, args.sub, run)
        process_motion(layout, args.sub, run, n_volumes)
