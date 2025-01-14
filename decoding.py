import numpy as np
import pandas as pd
import argparse
import re
import os

from bids import BIDSLayout
from util.bids import DataSink

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_curve
from scipy.interpolate import interp1d

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

def score(y, y_hat, run):
    auroc = []
    fpr_interp = np.linspace(0, 1., 100)
    tpr_interp = []
    for r in np.unique(run):
        fpr, tpr, _ = roc_curve(y[run == r], y_hat[run == r])
        auroc.append(auc(fpr, tpr))
        interp_func = interp1d(x = fpr, y = tpr)
        tpr = interp_func(fpr_interp)
        tpr[0] = 0. # always acheivable for fpr==0
        tpr_interp.append(tpr)
    tpr = np.mean(np.stack(tpr_interp, axis = 0), 0)
    return np.mean(auroc), fpr_interp, tpr

def _permutation(y, y_hat, run, seed = None):
    if seed == 0:
        yh = y_hat
    else:
        yh = shuffle_within_runs(y_hat, run, seed)
    return score(y, yh, run)

def permutation_test(y, y_hat, run):
    print('Starting permutation test...')
    parallel, p_func, n_jobs = parallel_func(
        _permutation, n_jobs = -1,
        verbose = 1
    )
    out = parallel(
        p_func(y, y_hat, run, seed = i)
        for i in range(N_PERMUTATIONS + 1)
    )
    auroc, fpr, tpr = zip(*out)
    auroc = np.array(auroc)
    fpr = np.stack(fpr)
    tpr = np.stack(tpr)
    return auroc, np.stack([fpr, tpr], axis = 1)

def shuffle_between_models(yh0, yh1, seed = None):
    rng = np.random.default_rng(seed)
    n = len(yh0)
    idxs = np.arange(n)
    idxs_mod = rng.binomial(1, .5, size = n)
    yh = np.stack([yh0, yh1], axis = 0)
    return yh[idxs_mod, idxs], yh[~idxs_mod, idxs]

def _permutation_paired(y, y_hat0, y_hat1, run, seed = None):
    if seed == 0:
        yh0, yh1 = y_hat0, y_hat1
    else:
        yh0, yh1 = shuffle_between_models(y_hat0, y_hat1, seed)
    return score(y, yh0, run)[0] - score(y, yh1, run)[0]

def permutation_test_paired(y, y_hat0, y_hat1, run):
    print('Starting paired permutation test...')
    parallel, p_func, n_jobs = parallel_func(
        _permutation_paired, n_jobs = -1,
        verbose = 1
    )
    out = parallel(
        p_func(y, y_hat0, y_hat1, run, seed = i)
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
        verbose = 1,
        method = 'decision_function' # i.e. log-odds
    )
    # now fit final model and extract model coefs & "patterns" as Haufe et al.
    lasso_pcr.fit(x, y)
    filters = np.full(mask.shape, np.nan)
    filters[mask] = get_coef(lasso_pcr, 'filters_', inverse_transform = True)
    patterns = np.full(mask.shape, np.nan)
    patterns[mask] = get_coef(lasso_pcr, 'patterns_', inverse_transform = True)
    return lasso_pcr, y_hat, filters, patterns

def save_model(sink, sub, name, model, filters, patterns, scores, roc):
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
        suffix = 'auc',
        extension = '.npy'
    )
    np.save(fpath, scores, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = name,
        datatype = 'func',
        suffix = 'roc',
        extension = '.npy'
    )
    np.save(fpath, roc, allow_pickle = False)
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


def main(layout, sub, mask_to_use = 'individual'):

    X, y, run, events  = load_subject_data(layout, sub)
    y_hat = dict() # to hold predictions from different models
    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'decoding'
    )

    # first fit model with the theory-driven mask
    name = 'theory' if mask_to_use == 'individual' else 'theoryGroup'
    theory = name # save for later
    if mask_to_use == 'individual':
        mask = load_subject_mask(layout, sub, mask_type = 'difference')
    elif mask_to_use == 'group':
        mask = load_subject_mask(layout, 'group', mask_type = 'difference')
    model, y_hat[name], filters, patterns = make_model(X, y, run, mask)
    # now compare predictions to chance
    null_scores, roc = permutation_test(y, y_hat[name], run)
    save_model(sink, sub, name, model, filters, patterns, null_scores, roc)

    name = 'visuomotor' if mask_to_use == 'individual' else 'visuomotorGroup'
    control = name
    if mask_to_use == 'individual':
        mask = load_subject_mask(layout, sub, mask_type = 'control')
    elif mask_to_use == 'group':
        mask = load_subject_mask(layout, 'group', mask_type = 'control')
    model, y_hat[name], filters, patterns = make_model(X, y, run, mask)
    # now compare predictions to chance
    null_scores, roc = permutation_test(y, y_hat[name], run)
    save_model(sink, sub, name, model, filters, patterns, null_scores, roc)

    name = 'cortex' if mask_to_use == 'individual' else 'cortexGroup'
    brain = name # save for later
    mask = whole_brain_mask(X)
    model, y_hat[name], filters, patterns = make_model(X, y, run, mask)
    # now compare predictions to chance
    null_scores, roc = permutation_test(y, y_hat[name], run)
    save_model(sink, sub, name, model, filters, patterns, null_scores, roc)

    sufx = '' if mask_to_use == 'individual' else 'Group'
    null_delta = permutation_test_paired(y, y_hat[brain], y_hat[theory], run)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'cortexVtheory%s'%sufx,
        datatype = 'func',
        suffix = 'auc',
        extension = '.npy'
    )
    np.save(fpath, null_delta, allow_pickle = False)

    null_delta = permutation_test_paired(y, y_hat[control], y_hat[theory], run)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'visuomotorVtheory%s'%sufx,
        datatype = 'func',
        suffix = 'auc',
        extension = '.npy'
    )
    np.save(fpath, null_delta, allow_pickle = False)

    null_delta = permutation_test_paired(y, y_hat[brain], y_hat[control], run)
    fpath = sink.get_path(
        subject = sub,
        session = '2',
        task = 'agency',
        desc = 'cortexVvisuomotor%s'%sufx,
        datatype = 'func',
        suffix = 'auc',
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
        desc = 'predictions%s'%sufx,
        datatype = 'func',
        suffix = 'logodds',
        extension = '.tsv'
    )
    df.to_csv(fpath, sep = '\t', index = False, na_rep = 'n/a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    parser.add_argument('mask', type = str)
    args = parser.parse_args()
    layout = BIDSLayout(BIDS_PATH, derivatives = True)
    if args.sub == 'all':
        subs = layout.get_subjects(scope = 'raw')
    else:
        subs = [args.sub]
    for sub in subs:
        if args.mask == 'group':
            print('Starting decoding for sub-%s...'%sub)
            main(layout, sub, 'group')
            print('Subject %s complete!'%sub)
        else:
            mask = load_subject_mask(layout, sub, 'difference')
            if mask.sum() == 0:
                print('Skipping sub-%s...'%sub)
            else:
                print('Starting decoding for sub-%s...'%sub)
                main(layout, sub, 'individual')
                print('Subject %s complete!'%sub)
    print('Done!!!')
