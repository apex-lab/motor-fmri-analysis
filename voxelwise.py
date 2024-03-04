from bids import BIDSLayout
import numpy as np
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit

from himalaya.backend import set_backend
backend = set_backend('numpy', on_error = 'warn')
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
from himalaya.scoring import r2_score as score_func
from voxelwise_tutorials.delayer import Delayer

from mne.parallel import parallel_func
from time import perf_counter
from joblib import dump, load

# general hard coded vars
DATA_DIR = 'bids_temp'
SUB = '01'
# permutation test args
N_PERMUTATIONS = 5000
BLOCK_SIZE = 10

layout = BIDSLayout(DATA_DIR, derivatives = True)

# load X's
X_fs = layout.get(subject = SUB, desc = 'downsamp', suffix = 'activations')
Xs = [x.get_df() for x in X_fs]
Xs = [x.loc[:, (x != 0).any(axis = 0)] for x in Xs] # rm cols w/ all 0s
X = np.concatenate([x.to_numpy() for x in Xs], axis = 0)

# load y's & run idx
bold_fs = layout.get(subject = SUB, desc = 'clean', suffix = 'bold')
bold = [np.load(b.path) for b in bold_fs]
run = [np.full(b.shape[0], i + 1) for i, b in enumerate(bold)]
bold = np.concatenate(bold, axis = 0)
run = np.concatenate(run)

assert(X.shape[0] == bold.shape[0])
assert(X.shape[0] == run.shape[0])

# what layer does each unit in ANN come from?
feat_types = np.array([col.split('_')[-1] for col in Xs[0].columns])
feat_types[np.isin(feat_types,  ['pos', 'vel', 'err'])] = 'input'
feat_types[np.isin(feat_types,  ['musc'])] = 'output'
feat_types[np.isin(feat_types,  ['layer1', 'layer2'])] = 'hidden'
appears_first = lambda ft: np.where(feat_types == ft)[0].min()
fts = np.unique(feat_types).tolist()
fts.sort(key = appears_first)
ft_cts = [(feat_types == ft).sum() for ft in fts] # counts
for ft, ct in zip(fts, ft_cts):
    print(ft, '-->', ct, 'units')

# make train/test split
train_idxs = run <= 8
test_idxs = ~train_idxs
X_train, y_train = X[train_idxs], bold[train_idxs]
X_test, y_test = X[test_idxs], bold[test_idxs]
run_train, run_test = run[train_idxs], run[test_idxs]

## fit kinematics-informed model
pipeline = make_pipeline(
    StandardScaler(with_mean = True, with_std = True),
    Delayer(delays = [1, 2, 3, 4]),
    RidgeCV(
        alphas = np.logspace(1, 10, 20),
        cv = PredefinedSplit(run_train),
        solver_params = dict(n_targets_batch = 1000)
    )
)
t0 = perf_counter()
if os.path.exists('model.joblib'):
    pipeline = load('model.joblib')
else:
    pipeline.fit(X_train, y_train)
    dump(pipeline, 'model.joblib')
t1 = perf_counter()
print('Training took %.03f minutes for ANN model.'%((t1 - t0)/60))

## generate permutation null distribution
yhat = pipeline.predict(X_test)
yhat = backend.to_numpy(yhat)
n_block_idxs = yhat.shape[0] // BLOCK_SIZE
block_idxs = np.arange(n_block_idxs)
block = np.concatenate([np.full(BLOCK_SIZE, i) for i in block_idxs])

def permutation_1samp(y_test, yhat, seed):
    yhat = yhat.copy()
    if seed == 0:
        yhat_perm = yhat
    else:
        rng = np.random.default_rng(seed)
        block_idxs = np.arange(n_block_idxs)
        rng.shuffle(block_idxs)
        yhat_perm = np.concatenate([yhat[block == idx] for idx in block_idxs])
    scrs = score_func(y_test, yhat_perm)
    return scrs

parallel, p_func, n_jobs = parallel_func(
    permutation_1samp, n_jobs = -1,
    verbose = 1
)
t0 = perf_counter()
out = parallel(
    p_func(y_test, yhat, i)
    for i in range(N_PERMUTATIONS + 1)
)
t1 = perf_counter()
H0 = np.stack(out, axis = 0)
np.save('H0_main.npy', H0, allow_pickle = False)
print('Finished computing null distribution in %.03f minutes'%((t1 - t0)/60))

## fit control model
control_pipeline = make_pipeline(
    StandardScaler(with_mean = True, with_std = True),
    Delayer(delays = [1, 2, 3, 4]),
    KernelRidgeCV(
        alphas = np.logspace(1, 10, 20),
        kernel = 'rbf', # use nonlinear projection of input
        cv = PredefinedSplit(run_train),
        solver_params = dict(n_targets_batch = 1000)
    )
)

t0 = perf_counter()
if os.path.exists('control.joblib'):
    control_pipeline = load('control.joblib')
else:
    control_pipeline.fit(X_train[:, feat_types == 'input'], y_train)
    dump(control_pipeline, 'control.joblib')
t1 = perf_counter()
print('Training took %.03f minutes for control model.'%((t1 - t0)/60))

# and generate permutation null for control model as well
yhat = control_pipeline.predict(X_test[:, feat_types == 'input'])
yhat = backend.to_numpy(yhat)
t0 = perf_counter()
out = parallel(
    p_func(y_test, yhat, i)
    for i in range(N_PERMUTATIONS + 1)
)
t1 = perf_counter()
H0_control = np.stack(out, axis = 0)
np.save('H0_control.npy', H0_control, allow_pickle = False)
print('Finished computing null distribution in %.03f minutes'%((t1 - t0)/60))

## permutation test comparing the two models
yhat_model = pipeline.predict(X_test)
yhat_model = backend.to_numpy(yhat_model)
yhat_control = control_pipeline.predict(X_test[:, feat_types == 'input'])
yhat_control = backend.to_numpy(yhat_control)
yhat = np.stack([yhat_model, yhat_control], axis = 0)

def permutation_paired(y_test, yhat, seed):
    yhat = yhat.copy()
    # randomly shuffle blocks of consecutive TRs b/w models
    if seed == 0: # except on first "permutation," which is just observed
        yhat_model_perm = yhat[0,...]
        yhat_control_perm = yhat[1,...]
    else:
        rng = np.random.default_rng(seed)
        swap = rng.binomial(1, .5, n_block_idxs)
        model_perm = swap[block]
        control_perm = -1*(model_perm - 1)
        yhat_model_perm = yhat[model_perm, np.arange(yhat.shape[1]), :]
        yhat_control_perm = yhat[control_perm, np.arange(yhat.shape[1]), :]
    # compute test scores on shuffled data
    scrs_model = score_func(y_test, yhat_model_perm)
    scrs_control = score_func(y_test, yhat_control_perm)
    # negative scores aren't meaningful so impose floor of zero
    scrs_model = np.maximum(scrs_model, 0)
    scrs_control = np.maximum(scrs_control, 0)
    # and return difference in R2 between models
    delta_perm = scrs_model - scrs_control
    return delta_perm

parallel, p_func, n_jobs = parallel_func(
    permutation_paired, n_jobs = -1,
    verbose = 1
)
t0 = perf_counter()
out = parallel(
    p_func(y_test, yhat, i)
    for i in range(N_PERMUTATIONS + 1)
)
t1 = perf_counter()
H0_delta = np.stack(out, axis = 0)
np.save('H0_delta.npy', H0_delta, allow_pickle = False)
print('Finished computing paired null in %.03f minutes'%((t1 - t0)/60))
