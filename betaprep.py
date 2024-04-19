from nilearn.glm.first_level import (
    FirstLevelModel, first_level_from_bids,
    make_first_level_design_matrix, run_glm
)
from nilearn.glm import compute_contrast
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.surface import load_surf_data

from scipy.stats import lognorm
import numpy as np
import pandas as pd
import argparse
import re
import os

from bids import BIDSLayout
from util.bids import DataSink

BIDS_PATH = 'bids_temp'

def get_z_map(design_matrix, labels, estimates, cond):
    contrast_values = (design_matrix.columns == cond) * 1.0
    contrast = compute_contrast(
        labels, estimates,
        contrast_values,
        stat_type = 't'
    )
    z_map = contrast.z_score()
    return z_map

def load_beta_series_run(layout, sub, run):
    '''
    Loads a single run, cleans BOLD data, and computes beta series
    for actuated muscle movements using the "least squares all" approach.
    '''
    query = dict(
        subject = sub,
        run = run,
        task = 'agency'
    )
    left_f, _, right_f, sidecar_f = layout.get(
        space = 'fsaverage',
        suffix = 'bold',
        scope = 'fMRIPrep',
        **query
    )
    left = load_surf_data(left_f.path)
    right = load_surf_data(right_f.path)
    bold = np.concatenate([left, right], axis = 0)
    events_f = layout.get(scope = 'raw', suffix = 'events', **query)[0]
    confounds = load_confounds_strategy(left_f.path, 'compcor')[0]
    events_df = events_f.get_df()

    # drop last trial which always goes over, past the last TR,
    # which makes the design matrix wacky
    events_df = events_df.iloc[:-1, :]

    # edit event times to reflect time and duration actuated movements
    events_df.onset += events_df.latency # use stim time as onset
    # and use time between stim and button press as duration of movement
    move_time = np.median(
        np.maximum(events_df.response_time - events_df.latency, 0)
        )
    events_df.duration = move_time
    agency = events_df.agency.astype(bool)
    events_df.trial_type = np.select(
        [agency, ~agency],
        ['self', 'other']
    )

    # Transform events DataFrame for LSA
    lsa_events_df = events_df.copy()
    conditions = lsa_events_df["trial_type"].unique()
    condition_counter = {c: 0 for c in conditions}
    for i_trial, trial in lsa_events_df.iterrows():
        trial_condition = trial["trial_type"]
        condition_counter[trial_condition] += 1
        trial_name = '%s__%03d'%(
            trial_condition,
            condition_counter[trial_condition]
        )
        lsa_events_df.loc[i_trial, "trial_type"] = trial_name
    lsa_events_df = lsa_events_df[['onset', 'duration', 'trial_type']]

    n_scans = bold.shape[1]
    tr = sidecar_f.get_dict()['RepetitionTime']
    slice_corr = tr / 2 # adjust TR timestamps for slice timing correction
    frame_times = tr * (np.arange(n_scans) + slice_corr)

    design_matrix = make_first_level_design_matrix(
        frame_times,
        events = lsa_events_df,
        hrf_model = 'spm',
        add_regs = confounds
    )

    labels, estimates = run_glm(
        bold.T.astype(float), # explicit float cast is necessary
        design_matrix.values.astype(float),
        n_jobs = -1,
        noise_model = 'ar1'
    )

    # get beta series
    trial_conds = lsa_events_df.trial_type
    z_maps = [
        get_z_map(design_matrix, labels, estimates,cond)
        for cond in trial_conds
    ]
    X = np.stack(z_maps)
    y = events_df.agency.to_numpy().astype(int)
    return X, y, events_df

def main(layout, sub):
    '''
    Computes beta series for all runs of one subject, and saves
    consolidated (across runs) neural and behavioral data.

    Applies the trial exclusion criteria from our previous EEG study that
    validated this task (DOI: https://doi.org/10.1523/JNEUROSCI.1116-23.2023
    or preprocessing.py@v0.0.2 at https://github.com/apex-lab/agency-analysis)
    to identify outlier trials, but doesn't remove these trials yet. Instead,
    they're just marked with a boolean 'exclude' column in the consolidated
    events .tsv/dataframe.
    '''
    runs = layout.get_runs(subject = sub, task = 'agency')
    runs = np.sort(runs)
    stim_runs = runs[runs != 1] # remove baseline run (i.e. w/ no stim)
    run_labs = []
    Xs = []
    ys = []
    events_all = []
    for run in stim_runs:
        print('processing BOLD for run %d...'%run)
        if run == 1:
            continue
        X, y, events = load_beta_series_run(layout, sub, run)
        run_labs.append(np.full_like(y, run))
        Xs.append(X)
        ys.append(y)
        events['run'] = run
        events_all.append(events)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    run = np.concatenate(run_labs)
    events = pd.concat(events_all)
    print('Done computing beta series.')

    ## flag w/ exclusion criteria from 10.1523/JNEUROSCI.1116-23.2023
    lags = events.response_time - events.latency
    stim_lags = lags[events.pressed_first == False]
    params = lognorm.fit(stim_lags)
    lower = lognorm.ppf(.025, params[0], params[1], params[2])
    upper = lognorm.ppf(.975, params[0], params[1], params[2])
    events['exclude'] = (lags > upper) | (lags < lower)
    k = events.exclude.sum()
    n = events.shape[0]
    print('Excluding %d/%d trials, or %.02f%%.'%(k, n, 100*k/n))

    ## and save everything 
    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'betaprep'
    )
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'betas',
        datatype = 'func',
        suffix = 'X',
        extension = '.npy'
    )
    np.save(fpath, X, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'betas',
        datatype = 'func',
        suffix = 'y',
        extension = '.npy'
    )
    np.save(fpath, y, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'betas',
        datatype = 'func',
        suffix = 'events',
        extension = '.tsv'
    )
    events.to_csv(fpath, sep = '\t', index = False, na_rep = 'n/a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    args = parser.parse_args()
    layout = BIDSLayout(BIDS_PATH, derivatives = True)
    if args.sub == 'all':
        subs = layout.get_subjects(scope = 'raw')
    else:
        subs = [args.sub]
    for sub in subs:
        print('Computing beta series for sub-%s...'%sub)
        main(layout, sub)
