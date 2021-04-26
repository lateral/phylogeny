import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import unittest
import time
import numpy as np
import dendropy
from tree_utils import draw_tree, generate_character_matrix, character_matrix_to_counts, counts_to_rates, counts_to_estimates, bionj, weighbor, robinson_foulds, array_to_pdm, pdm_to_array
from paup import get_best_log_likelihood_for_topology, infer_tree as paup_infer_tree
from phyml import infer_tree as phyml_infer_tree
from ctmc import JC69
from cythonised.logalike import LogalikeOptimiser
from cythonised.hyperbolic_mds import MDS
from dendropy.datamodel.treemodel import Tree
from hyperboloid import Hyperboloid
from dendropy import PhylogeneticDistanceMatrix


MUTN_MODEL = JC69()
MIN_VARIANCE = 1e-5


def euclidean_pairwise_distances(pts):
    n = pts.shape[0]
    dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(((pts[i] - pts[j]) ** 2).sum())
            dists[i, j] = dist
            dists[j, i] = dist
    return dists


class Experiment:

    def __init__(self, tree_seed, num_leaves, min_length, max_length,
                 sequence_length, sequence_seed, dimension, rho,
                 logalike_hyperparams, mds_hyperparams):
        self.__dict__ = {**self.__dict__, **locals()}; self.__dict__.pop('self')
        assert self.rho != np.inf
        self.params = dict(locals())  # used for reporting
        self.params.pop('self')

        self.runs = {}
        self.leaf_names = ['T%04d' % i for i in range(num_leaves)]
        self.taxon_namespace = dendropy.TaxonNamespace(self.leaf_names)

        # generate tree
        self.generating_tree = draw_tree(tree_seed, self.taxon_namespace)
        np.random.seed(tree_seed)
        for e in self.generating_tree.edges():
            if e.length is not None:
                e.length = np.random.rand() * (max_length - min_length) + min_length

        # generate sequence
        cm = generate_character_matrix(self.generating_tree, self.sequence_length, self.sequence_seed)
        self.counts = character_matrix_to_counts(cm)
        self.rates = counts_to_rates(self.counts)
        self.est_dists, self.variances = counts_to_estimates(self.counts)
        self.variances[self.variances < MIN_VARIANCE] = MIN_VARIANCE
        # calculate the log likelihood of the true tree
        self.generating_tree_ll, _ = get_best_log_likelihood_for_topology(self.generating_tree, cm)

        self.hyperboloid = Hyperboloid(local_dimension=dimension, rho=rho)
        # generate initial points
        est_pdm = array_to_pdm(self.est_dists, self.taxon_namespace)
        init_tree = weighbor(est_pdm, self.sequence_length)
        init_tree.reroot_at_midpoint(update_bipartitions=True, suppress_unifurcations=True)
        # tune the edge lengths of that tree
        _, init_tree = get_best_log_likelihood_for_topology(init_tree, cm)
        # ensure there are no edges of length 0
        ZERO_EDGE_LENGTH = 0.0001
        for e in init_tree.edges():
            if e.length is not None:
                if e.length < ZERO_EDGE_LENGTH:
                    e.length = ZERO_EDGE_LENGTH

        vectors = self.hyperboloid.embed_tree(init_tree)
        self.initial_pts = np.array([vectors[init_tree.find_node_for_taxon(taxon)] for taxon in self.taxon_namespace])

        # find the optimum tree according to ML tree search
        # use PhyML to get the best tree
        # (it gives better optima than PAUP)
        self.optimum_tree, _, _ = phyml_infer_tree(cm)
        self.optimum_ll, _ = get_best_log_likelihood_for_topology(self.optimum_tree, cm)

    def _character_matrix(self):
        # Note: construction of the character matrix is re-done here as a local
        # variable since it refuses to pickle. Reconstruction is no problem,
        # since we use the same seed.
        cm = generate_character_matrix(self.generating_tree,
                                       self.sequence_length,
                                       self.sequence_seed)
        return cm

    def nj(self):
        est_pdm = array_to_pdm(self.est_dists, self.taxon_namespace)
        return est_pdm.nj_tree(), None

    def bionj(self):
        est_pdm = array_to_pdm(self.est_dists, self.taxon_namespace)
        return bionj(est_pdm), None

    def weighbor(self):
        est_pdm = array_to_pdm(self.est_dists, self.taxon_namespace)
        return weighbor(est_pdm, self.sequence_length), None

    def phyml(self):
        tree, _, _ = phyml_infer_tree(self._character_matrix())
        return tree, None

    def paup(self):
        tree, _, _ = paup_infer_tree(self._character_matrix())
        return tree, None

    def logalike(self):
        logalike = LogalikeOptimiser(rho=self.rho,
                                     lr=self.logalike_hyperparams['lr'],
                                     max_step_size=self.logalike_hyperparams['max_step_size'])
        logalike.set_points(self.initial_pts.copy())
        converged, failed, checkpoints = \
            logalike.fit(self.rates,
                         stopping_distance=self.logalike_hyperparams['stopping_distance'],
                         max_rounds=self.logalike_hyperparams['max_rounds'],
                         checkpoint_interval=1000)
        meta = {'converged': converged, 'failed': failed, 'rounds_done': logalike.rounds_done}
        meta['constrained'] = self._satisfies_constraints(logalike.get_points())
        pw_dists = self.hyperboloid.pairwise_distances(logalike.get_points())
        est_pdm = array_to_pdm(pw_dists, self.taxon_namespace)
        return weighbor(est_pdm, self.sequence_length), meta

    def mds(self):
        mds = MDS(rho=self.rho, lr=self.mds_hyperparams['lr'], max_step_size=self.mds_hyperparams['max_step_size'])
        mds.set_points(self.initial_pts.copy())
        converged, failed, checkpoints = \
            mds.fit(self.est_dists,
                    self.variances,
                    stopping_distance=self.mds_hyperparams['stopping_distance'],
                    max_rounds=self.mds_hyperparams['max_rounds'],
                    checkpoint_interval=1000)
        meta = {'converged': converged, 'failed': failed, 'rounds_done': mds.rounds_done}
        meta['constrained'] = self._satisfies_constraints(mds.get_points())
        pw_dists = self.hyperboloid.pairwise_distances(mds.get_points())
        est_pdm = array_to_pdm(pw_dists, self.taxon_namespace)
        return weighbor(est_pdm, self.sequence_length), meta

    def evaluate_method(self, method):
        result = {}
        start_time = time.time()
        tree, meta = method()
        result['duration'] = time.time() - start_time
        result['meta'] = meta
        result['tree'] = tree.as_string(schema='newick')
        result['ll'], _ = get_best_log_likelihood_for_topology(tree, self._character_matrix())
        result['rf-to-generating'] = robinson_foulds(tree, self.generating_tree)
        # measure the RF to the optimum tree according to ML tree search
        result['optimum-ll'] = self.optimum_ll
        result['rf-to-optimum'] = robinson_foulds(tree, self.optimum_tree)
        self.runs[method.__name__] = result
        return self

    def evaluate_all(self):
        self.evaluate_method(self.mds)
        self.evaluate_method(self.nj)
        self.evaluate_method(self.bionj)
        self.evaluate_method(self.weighbor)
        self.evaluate_method(self.paup)
        self.evaluate_method(self.phyml)
        self.evaluate_method(self.logalike)
        return self

    def _satisfies_constraints(self, pts):
        for pt in pts:
            if not self.hyperboloid.contains(pt):
                return False
        return True

    def summarise_results(self):
        res = {**(self.params)}
        for name, run in self.runs.items():
            for key, value in run.items():
                res['%s_%s' % (name, key)] = value
        res['generating_tree_ll'] = self.generating_tree_ll
        return res


class TestExperiment(unittest.TestCase):

    def test_experiment(self):
        logalike_hyperparams = {
            'stopping_distance': 1e-6,
            'max_step_size': 0.1,
            'max_rounds': 100,
            'lr': 0.005
        }
        mds_hyperparams = dict(logalike_hyperparams)

        experiment = Experiment(tree_seed=0,
                                num_leaves=5,
                                min_length=0.05,
                                max_length=0.4,
                                sequence_length=200,
                                sequence_seed=0,
                                dimension=5,
                                rho=0.75,
                                logalike_hyperparams=logalike_hyperparams,
                                mds_hyperparams=mds_hyperparams)
        experiment.evaluate_all()
