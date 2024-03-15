from bids import BIDSLayout
import numpy as np
import shap
import torch
import re
import os
import argparse
from joblib import load, Parallel, delayed
from torch.autograd import Variable
from statsmodels.stats.multitest import multipletests

from util.bids import DataSink
from util.myosuite import get_activations


ALPHA = 0.05 # for rejection mask
BIDS_DIR = 'bids_temp'
POLICY_FILE = 'best_policy.pickle'

N_BACKGROUND_SAMPLES = 100
N_TEST_SET_SAMPLES = 1000

this_dir = os.path.dirname(os.path.abspath(__file__))
bids_dir = os.path.join(this_dir, BIDS_DIR)
policy_file = os.path.join(this_dir, POLICY_FILE)

class EncodingModel(torch.nn.Module):

    def __init__(self, policy, scaler, coefs, non_empty):
        super().__init__()
        self._mu = torch.from_numpy(scaler.mean_).float()
        self._std = torch.from_numpy(scaler.scale_).float()
        self.weights = torch.from_numpy(coefs).float()
        self._model = policy.model
        self._non_empty = non_empty

    def forward(self, x):
        '''
        x : a ([n_observations,] observation_size) torch.tensor
            An observation vector in same form as returned by env.step().
        '''
        acts = get_activations(self._model, x, return_numpy = False)
        acts = torch.concat(acts, axis = 1)
        acts = acts[:, self._non_empty]
        acts_scaled = (acts - self._mu) / self._std
        return acts_scaled @ self.weights

def main(layout, sub):

    policy = load(policy_file)

    # get rejection mask for voxelwise model
    control_f, diff_f, model_f = layout.get(
        subject = sub, suffix = 'r2', scope = 'voxelwise'
    )
    H0 = np.load(model_f.path)
    ps = (H0[0,:] <= H0).mean(0)
    mask, _, _, _ = multipletests(ps, method = 'fdr_bh', alpha = ALPHA)

    # load X's and run indices
    X_fs = layout.get(
        subject = sub, task = 'motor',
        desc = 'fullsamp', suffix = 'observations'
        )
    Xs = [x.get_df() for x in X_fs]
    Xs = [df.loc[:, df.columns != 'time'] for df in Xs]
    X = np.concatenate([x.to_numpy() for x in Xs], axis = 0)
    run_idxs = [int(re.findall('run-(\w+)_', f.path)[0]) for f in X_fs]
    run = [np.full(x.shape[0], r) for x, r in zip(Xs, run_idxs)]
    run = np.concatenate(run)

    # make train/test split
    train_idxs = run <= 8
    test_idxs = ~train_idxs
    X_train, X_test = X[train_idxs], X[test_idxs]

    # extract feature type labels
    x = Xs[0]
    feat_types = np.array([col.split('_')[-1] for col in x.columns])

    # load and parse voxelwise encoding model pipeline
    mod_f = layout.get(
        subject = sub, desc = 'model', suffix = 'reg', scope = 'voxelwise'
    )[0]
    pipeline = load(mod_f.path)
    scaler = pipeline['standardscaler']
    delayer = pipeline['delayer']
    ridge = pipeline[-1]

    ## grab model coefficients (assuming was trained w/ numpy)
    primal_coef = ridge.coef_
    # split up by delay
    primal_coef_per_delay = delayer.reshape_by_delays(primal_coef, axis = 0)
    del primal_coef
    # average over delays
    average_coef = np.mean(primal_coef_per_delay, axis = 0)
    del primal_coef_per_delay

    # concatenate encoding weights with neural network
    acts = get_activations(policy.model, X)
    acts = np.concatenate(acts, axis = 1)
    non_empty = (acts != 0).any(axis = 0) # mask to remove empty activations
    model = EncodingModel(policy, scaler, average_coef[:, mask], non_empty)

    ## and explain predictions!
    def get_shapvals(background, test_samps):
        torch.set_grad_enabled(True)
        explainer = shap.DeepExplainer(
            model = model,
            data = Variable(torch.from_numpy(background).float()),
        )
        shap_values = explainer.shap_values(
            X = Variable(torch.from_numpy(test_samps).float()),
            check_additivity = False
        )
        shap_values = np.stack(shap_values, -1)
        return shap_values

    np.random.seed(0)
    background = shap.sample(X_train, N_BACKGROUND_SAMPLES)
    test_samps = shap.sample(X_test, N_TEST_SET_SAMPLES)
    out = Parallel(n_jobs = -1, verbose = 1)(
        delayed(get_shapvals)(background, test_samps[i:i+1, :])
        for i in range(len(test_samps))
    )
    shap_values = np.concatenate(out)

    # aggregate shap values over feature types and project to fsaverage space
    feat_space_svs = []
    for ft in ('pos', 'vel', 'err'):
        _sv = shap_values[:, feat_types == ft, :].sum(1) # sum over feat spaces
        # global shap val is (absolute) central tendency over examples
        _sv = np.nanmean(np.abs(_sv), 0)
        # now project back to fsaverage
        shapvals_fs = np.full_like(H0[0,:], np.nan)
        shapvals_fs[mask] = _sv
        feat_space_svs.append(shapvals_fs)
    sv = np.stack(feat_space_svs)

    ## and save!
    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'shapley'
    )
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        desc = 'PosVelErr',
        datatype = 'func',
        suffix = 'shap',
        extension = '.npy'
    )
    np.save(fpath, sv, allow_pickle = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    args = parser.parse_args()
    layout = BIDSLayout(bids_dir, derivatives = True)
    main(layout, args.sub)
