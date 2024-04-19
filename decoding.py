import numpy as np
import pandas as pd
import argparse
import re
import os

from bids import BIDSLayout
from util.bids import DataSink

from sklearn.metrics import accuracy_score as score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from mne.decoding import LinearModel
from mne.parallel import parallel_func
from mne.decoding import get_coef

from joblib import dump

BIDS_PATH = 'bids_temp'
N_PERMUTATIONS = 5000

def load_subject_data(layout, sub):
    '''
    reads in cleaned data from Session 2 from which to decode
    '''
    # load beta series
    events_f, X_f, y_f = layout.get(subject = sub, scope = 'betaprep')
    events = events_f.get_df()
    X = np.load(X_f.path)
    y = np.load(y_f.path)
    # and remove faulty stimulation trials
    include = ~events.exclude
    X, y, events = X[include], y[include], events[include]
    run = events.run.to_numpy()
    return X, y, run, events

def load_subject_mask(layout, sub, mask_type = 'difference'):
    '''
    reads in masks derived from Session 1 data
    '''
    f = layout.get(subject = sub, suffix = 'mask', desc = mask_type)[0]
    return np.load(f)

def whole_brain_mask(X):
    return np.ones(X.shape[1]).astype(bool)

def shuffle_within_runs(v, runs, seed = None):
    rng = np.random.default_rng(seed)
    v = v.copy()
    for r in np.unique(runs):
        v_run = v[runs == r]
        rng.shuffle(v_run)
        v[runs == r] = v_run
    return v

def _permutation(y, y_hat, run, seed = None):
    if seed == 0:
        yh = y_hat
    else:
        yh = shuffle_within_runs(y_hat, run, seed)
    return score(y, yh)

def permutation_test(y, y_hat, run):
    print('Starting permutation test...')
    accuracy = score(y, hat)
    parallel, p_func, n_jobs = parallel_func(
        _permutation, n_jobs = -1,
        verbose = 1
    )
    out = parallel(
        p_func(y, y_hat, run, seed = i)
        for i in range(N_PERMUTATIONS + 1)
    )
    H0 = np.array(out)
    return H0

def shuffle_between_models(yh0, yh1, seed = None):
    rng = np.random.default_rng(seed)
    n = len(yh0)
    idxs = np.arange(n)
    idxs_mod = rng.binomial(1, .5, size = n)
    yh = np.stack([yh0, yh1], axis = 0)
    return yh[idxs_mod, idxs], yh[~idxs_mod, idxs]

def _permutation(y, y_hat0, y_hat1, seed = None):
    if seed == 0:
        yh0, yh1 = y_hat0, y_hat1
    else:
        yh0, yh1 = shuffle_between_models(y_hat0, y_hat1, seed)
    return score(y, yh0) - score(y, yh1)

def permutation_test_paired(y, y_hat0, y_hat1):
    print('Starting paired permutation test...')
    accuracy = score(y, hat)
    parallel, p_func, n_jobs = parallel_func(
        _permutation, n_jobs = -1,
        verbose = 1
    )
    out = parallel(
        p_func(y, y_hat0, y_hat1, seed = i)
        for i in range(N_PERMUTATIONS + 1)
    )
    H0 = np.array(out)
    return H0

def make_model(X, y, run, mask):
    # construct classification pipeline
    lasso_pcr = make_pipeline(
        StandardScaler(),
        PCA(
            whiten = False,
            svd_solver = 'full'
        ),
        LinearModel(LogisticRegressionCV(
            Cs = np.logspace(1, 10, 20),
            penalty = 'l1',
            solver = 'liblinear',
            cv = 5, # number of nested folds
            random_state = 0
        ))
    )
    # get cross-validate d predictions for later permutation testing
    x = X[:, mask]
    print('Fitting model with %d predictors...'%x.shape[1])
    y_hat = cross_val_predict(
        lasso_pcr,
        x, y,
        groups = run,
        cv = LeaveOneGroupOut(),
        n_jobs = -1,
        verbose = 1
    )
    # now fit final model and extract model coefs & "patterns" as Haufe et al.
    lasso_pcr.fit(x, y)
    filters = np.full(mask.shape, np.nan)
    filters[mask] = get_coef(lasso_pcr, 'filters_', inverse_transform = True)
    patterns = np.full(mask.shape, np.nan)
    patterns[mask] = get_coef(lasso_pcr, 'patterns_', inverse_transform = True)
    return lasso_pcr, y_hat, filters, patterns

def save_model(sink, sub, name, model, filters, patterns, H0):
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = name,
        datatype = 'func',
        suffix = 'filters',
        extension = '.npy'
    )
    np.save(fpath, filters, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = name,
        datatype = 'func',
        suffix = 'patterns',
        extension = '.npy'
    )
    np.save(fpath, patterns, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = name,
        datatype = 'func',
        suffix = 'scores',
        extension = '.npy'
    )
    np.save(fpath, H0, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = name,
        datatype = 'func',
        suffix = 'reg',
        extension = '.npy'
    )
    dump(model, fpath)


def main(layout, sub):

    X, y, run, events  = load_subject_data(layout, sub)
    y_hat = dict() # to hold predictions from different models
    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'decoding'
    )

    # first fit model with the theory-driven mask
    name = 'theory'
    theory = name # save for later
    mask = load_subject_mask(layout, sub, mask_type = 'difference')
    model, y_hat[name], filters, patterns = make_model(X, y, run, mask)
    # now compare predictions to chance
    null_scores = permutation_test(y, y_hat[name], run)
    save_model(sink, sub, name, model, filters, patterns, null_scores)

    name = 'visuomotor'
    mask = load_subject_mask(layout, sub, mask_type = 'control')
    model, y_hat[name], filters, patterns = make_model(X, y, run, mask)
    # now compare predictions to chance
    null_scores = permutation_test(y, y_hat[name], run)
    save_model(sink, sub, name, model, filters, patterns, null_scores)

    name = 'cortex'
    brain = name # save for later
    mask = whole_brain_mask(X)
    model, y_hat[name], filters, patterns = make_model(X, y, run, mask)
    # now compare predictions to chance
    null_scores = permutation_test(y, y_hat[name], run)
    save_model(sink, sub, name, model, filters, patterns, null_scores)

    null_delta = permutation_test_paired(y, y_hat[brain], y_hat[theory])
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'delta',
        datatype = 'func',
        suffix = 'scores',
        extension = '.npy'
    )
    np.save(fpath, null_delta, allow_pickle = False)

    df = pd.DataFrame(y_hat)
    df['y'] = y
    df['run'] = run
    df['response_time'] = events.response_time.to_numpy()
    df['latency'] = events.latency.to_numpy()
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'predictions',
        datatype = 'func',
        suffix = 'yhat',
        extension = '.tsv'
    )
    df.to_csv(fpath, sep = '\t', index = False, na_rep = 'n/a')

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
        mask = load_subject_mask(layout, sub, 'difference')
        if mask.sum() == 0:
            print('Skipping sub-%s...'%sub)
        else:
            print('Starting decoding for sub-%s...'%sub)
            main(layout, sub)
            print('Subject %s complete!'%sub)
    print('Done!!!')
