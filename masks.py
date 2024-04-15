from statsmodels.stats.multitest import multipletests
from bids import BIDSLayout
import numpy as np
import argparse
import os

from util.cluster_stats import get_adjacency_fsaverage, ClusterStatsOneTailed
from util.bids import DataSink

DATA_DIR = 'bids_temp'
ALPHA = .05
TFCE_STEP = 0.01

def main(layout, sub):

    sink = DataSink(
        os.path.join(layout.root, 'derivatives'),
        workflow = 'masks'
    )

    # load voxelwise null distributions
    control_f, delta_f, model_f = layout.get(
        subject = sub,
        suffix = 'r2',
        scope = 'voxelwise'
        )
    H0_control = np.load(control_f.path)
    H0_delta = np.load(delta_f.path)
    H0_model = np.load(model_f.path)

    # visuomotor mask is just FDR-corrected mask for control model > chance
    ps = (H0_control[0,:] <= H0_control).mean(0)
    reject, _, _, _ = multipletests(ps, method = 'fdr_bh', alpha = ALPHA)
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        desc = 'control',
        datatype = 'func',
        suffix = 'mask',
        extension = '.npy'
    )
    np.save(fpath, reject, allow_pickle = False)

    # and while we're at it, we'll generate same thing for the model model
    ps = (H0_model[0,:] <= H0_model).mean(0)
    reject, _, _, _ = multipletests(ps, method = 'fdr_bh', alpha = ALPHA)
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        desc = 'model',
        datatype = 'func',
        suffix = 'mask',
        extension = '.npy'
    )
    np.save(fpath, reject, allow_pickle = False)

    ## threshold-free cluster enhancement
    adjacency = get_adjacency_fsaverage()
    grid = np.arange(TFCE_STEP, 1, TFCE_STEP)
    thresholds = grid[grid < H0_delta.max()].tolist()
    clust = ClusterStatsOneTailed(adjacency, thresholds)
    _, tfce_ps, H0_clust = clust.perm_test(H0_delta)
    if tfce_ps.min() <= ALPHA:
        # TFCE step size discretizes TFCE stat, so we include
        # vertices that barely "missed" significance in mask
        # as some presumably missed just due to TFCE step size
        # (assuming effect was smooth) and we want to capture the
        # "full" ROI for subsequent decoding, so we can be
        # a tiny bit liberal as long as the global null was rejected
        # at the nominal significance level
        ps_sorted = np.sort(tfce_ps.copy()) # i.e. find lowest p that's > alpha
        alpha = ps_sorted[np.where(ps_sorted > ALPHA)[0][0]] # and include that
    else: # but of course we don't do that if global null isn't rejected
        alpha = ALPHA # cause we're not out here tryna create false positives
    mask = tfce_ps <= alpha
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        desc = 'difference',
        datatype = 'func',
        suffix = 'mask',
        extension = '.npy'
    )
    np.save(fpath, mask, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        desc = 'difference',
        datatype = 'func',
        suffix = 'pvals',
        extension = '.npy'
    )
    np.save(fpath, tfce_ps, allow_pickle = False)
    tfce_stat = clust.get_tfce_stats(H0_delta[0,:])
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        desc = 'difference',
        datatype = 'func',
        suffix = 'tfce',
        extension = '.npy'
    )
    np.save(fpath, tfce_stat, allow_pickle = False)
    fpath = sink.get_path(
        subject = sub,
        session = '1',
        task = 'motor',
        desc = 'difference',
        datatype = 'func',
        suffix = 'H0',
        extension = '.npy'
    )
    np.save(fpath, H0_clust, allow_pickle = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    args = parser.parse_args()
    layout = BIDSLayout(DATA_DIR, derivatives = True)
    if args.sub == 'all':
        subs = layout.get_subjects(scope = 'voxelwise')
    else:
        subs = [args.sub]
    for sub in subs:
        print('Generating rejection masks for sub-%s...'%sub)
        main(layout, sub)
