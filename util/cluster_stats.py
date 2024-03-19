from mne.stats.cluster_level import (
    _find_clusters,
    _cluster_indices_to_mask
)
from mne.parallel import parallel_func
from nilearn import datasets, surface
from sklearn import neighbors
import numpy as np

def get_adjacency_fsaverage(radius = 2.):
    '''
    derives adjacency matrix for vertices on the fsaverage surface

    Arguments
    --------
    radius : float
        Inter-vertex distance within which two vertices are considered
        adjacent. Fsaverage has an average inter-vertex distance of
        roughly 1mm, so 2mm should capture most (if not all) neighbors.

    Returns
    --------
    adjacency : scipy.sparse.spmatrix of shape (n_vertices, n_vertices)
    '''
    fsaverage = datasets.fetch_surf_fsaverage(mesh = "fsaverage")
    left_coords, _ = surface.load_surf_mesh(fsaverage["pial_left"])
    right_coords, _ = surface.load_surf_mesh(fsaverage["pial_right"])
    coords = np.concatenate([left_coords, right_coords])
    nn = neighbors.NearestNeighbors(radius = radius)
    adjacency = nn.fit(coords).radius_neighbors_graph(coords)
    return adjacency


class ClusterStatsOneTailed:
    '''
    Implements an upper/one-tailed cluster-based permutation test,
    or threshold-free cluster enhancement (TFCE) if a list of
    clustering thresholds is specified.

    This is an alternative API to some of MNE's internal
    functions, which allows us to use vertex-level test
    statistics from precomputed permutations (instead of
    MNE doing the permutations for us). This is handy,
    since we can use permutation schemes that MNE doesn't
    implement, such as shuffling blocks of TRs. And unlike
    nilearn's TFCE implementation, it works with arbitrary
    adjacency matrices, so we can perform clustering on
    the cortical surface (not just in volume space).
    '''

    def __init__(self, adjacency, threshold):
        '''
        Arguments
        ----------
        adjacency : scipy.sparse.spmatrix of shape (n_vertices, n_vertices)
            Specifies which vertices are next to one another for clustering.
        threshold : float or list[float]
            If float, `ClusterTest.perm_test` will implment a cluster
            based permutation test as in Maris & Oostenveld (2007) with
            specified threshold for cluster inclusion. If list of floats,
            will use FSL/nilearn-like implmentation of threshold-free
            cluster enhancement (Smith & Nichols, 2009).
        '''
        self.adj = adjacency.tocoo()
        self.thres = threshold
        return None

    def _get_cluster_stats(self, x, threshold, t_power):
        clusters, cluster_stats = _find_clusters(
            x, threshold,
            tail = 1,
            adjacency = self.adj,
            max_step = 1,
            include = None,
            partitions = None,
            t_power = t_power,
            show_info = True
        )
        clusters  =_cluster_indices_to_mask(clusters, self.adj.shape[0])
        return np.array(clusters), cluster_stats

    def get_tfce_stats(self, x, E = .5, H = 2.):
        '''
        Computes TFCE stats for each vertex, given
        an (n_vertices,) observation `x`.

        Arguments
        ----------
        E : float, default: 0.5
            Exponential weight for extent. The canonical value is 0.5.
        H : float, default: 2.0
            Exponential weight for height. The canonical value is 2.0.

        Notes
        -------
        The original TFCE paper and MNE use dh^H instead of h^H
        in computing the TFCE stat, but we use h^H to be
        consistent with FSL and with nilearn implementations
        (i.e. those used most often in the fMRI literature).
        This doesn't affect false positive rates, but it's
        something to be aware of -- there are actually two
        commonly used implementations of TFCE floating around!
        '''
        tfce_stats = np.zeros_like(x)
        for thres in self.thres:
            clusters, extent = self._get_cluster_stats(x, thres, 0)
            for c, e in zip(clusters, extent):
                h = thres # see notes in docstring
                tfce_stats[c] += (e**E) * (h**H)
        return tfce_stats

    def get_cluster_stats(self, x, t_power = 1):
        '''
        Arguments
        ------------
        x : np.array of shape (n_vertices,)
            Test statistic at each vertex/voxel.
        t_power : int, default: 1
            Exponent by which to raise test statistic at each vertex
            to before summing within a cluster. If 0,
            then cluster statistic is just count of vertices,
            if 1, then a sum, if 2, then squared sum, etc.
        '''
        if isinstance(self.thres, list):
            return None, self.get_tfce_stats(x)
        return self._get_cluster_stats(x, self.thres, t_power)

    def get_max_stat(self, x, t_power = 1):
        _, stats = self.get_cluster_stats(x, t_power)
        if stats.size == 0: # i.e. no clusters
            return 0.
        else:
            return np.max(stats)

    def perm_test(self, H0, t_power = 1, n_jobs = -1):
        '''
        Computes cluster statistics on precomputed permutation
        distributions of the test statistic for each voxel.
        Returned p-values are upper/one-tailed.

        Arguments
        ---------
        H0 : np.array of shape (n_permutations, n_vertices)
            The permutation distribution of the test statistic
            at each vertex/voxel. We assume that H0[0,:] is the
            observed test statistic.
        t_power : int, default: 1
            Exponent to raise test statistic at each vertex
            to before summing within a cluster. If 0,
            then cluster statistic is just a count, if
            1, then a sum, if 2, then squared sum, etc.
            This will be ignored if using TFCE.

        Returns
        ---------
        clusters : an (n_clusters, n_vertices) np.array or None
            Boolean masks indicating cluster membership.
            If using TFCE, this will be None.
        ps : an (n_clusters,) or (n_vertices,) np.array
            The p-values for each cluster for cluster-based permutation
            test or for each vertex for TFCE.
        H0_clust : np.array of shape (n_permutations,)
            The permutation null distribution of the
            maximum cluster statistic.
        '''
        parallel, p_func, n_jobs = parallel_func(
            self.get_max_stat,
            n_jobs = n_jobs,
            verbose = 1
        )
        out = parallel(
            p_func(H0[i, :], t_power)
            for i in range(H0.shape[0])
        )
        H0_clust = np.array(out)
        clusters, stats = self.get_cluster_stats(H0[0, :], t_power)
        clust_ps = np.array([(s <= H0_clust).mean() for s in stats])
        return clusters, clust_ps, H0_clust
