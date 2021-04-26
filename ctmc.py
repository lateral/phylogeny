import numpy as np
import unittest


OVERFLOW_DISTANCE = 10  # distance to use in distance estimation when there is no maximum likelihood estimate


class F81:

    def __init__(self, initial_dist):
        self.initial_dist = np.array(initial_dist)
        assert (self.initial_dist.sum() == 1)
        self.num_chars = len(initial_dist)
        self.effective_rate = 1.
        # The following rate of ALL mutations gives an EFFECTIVE (i.e. non-redundant) mutation rate of 1 / unit time
        self.rate = self.effective_rate / self.initial_dist.dot(1 - self.initial_dist)
        # calculate the Q matrix
        self.infinitesimal_generator = self._infinitesimal_generator()

    def ctmc_transitions(self, time_elapsed):
        """
        Returns the P matrix of the mutation model.
        """
        decay = np.exp(-self.rate * time_elapsed)
        mat = np.empty((self.num_chars, self.num_chars), dtype=np.float64)
        for i in range(self.num_chars):
            for j in range(self.num_chars):
                if i == j:
                    mat[i, j] = self.initial_dist[j] + (1 - self.initial_dist[j]) * decay
                else:
                    mat[i, j] = self.initial_dist[j] * (1 - decay)
        return mat

    def _infinitesimal_generator(self):
        """
        Return the Q-matrix (incorporating the rate term, self.rate).
        """
        mat = np.empty((self.num_chars, self.num_chars), dtype=np.float64)
        for i in range(self.num_chars):
            for j in range(self.num_chars):
                if i == j:
                    mat[i, j] = self.initial_dist[j] - 1
                else:
                    mat[i, j] = self.initial_dist[j]
        return self.rate * mat


class JC69(F81):

    def __init__(self):
        super().__init__([0.25] * 4)

    def counts_to_estimates(self, counts):
        """
        Given a 2d Numpy array `counts` counting the number of times each
        nucleotide was observed to change into another (so an integer count
        matrix, the sum of whose entries is the sequence length), return the
        maximum likelihood estimate and the variance.
        """
        assert counts.shape == (4, 4)
        # TODO this could also be implemented in F81
        num_sites = counts.sum()
        num_same = np.trace(counts)
        diffs_per_site = (num_sites - num_same) / num_sites
        try:
            dist = (-3 / 4) * np.log(1 - (4 / 3) * diffs_per_site)
        except FloatingPointError:
            # there is no ML estimate; standard practice is just to use a very
            # large distance.
            dist = OVERFLOW_DISTANCE
        # Calculate the variance:
        # Uses the formula from W. J. Bruno, N. D. Socci, and A. L. Halpern.
        # Weighted neighbor joining: a likelihood-based approach to
        # distance-based phylogeny reconstruction. Mol. Biol. Evol.,
        # 17(1):189–197, Jan 2000.
        variance = np.exp(8 * dist / 3) * diffs_per_site * (1 - diffs_per_site) / num_sites
        return dist, variance

    def corrected_distance(self, seq0, seq1):
        """
        Return the "evolutionary distance" between sequences seq0 and seq1, according to the JC69 model.
        """
        # TODO this should be implemented as a special case of the F81 distance correction
        assert len(seq0) == len(seq1)
        num_sites = len(seq0)
        num_diffs = 0
        for i in range(num_sites):
            if seq0[i] != seq1[i]:
                num_diffs += 1
        diffs_per_site = num_diffs / num_sites
        try:
            dist = (-3 / 4) * np.log(1 - (4 / 3) * diffs_per_site)
        except FloatingPointError:
            dist = OVERFLOW_DISTANCE  # there is no ML estimate; standard practice is just to use a very large distance.
        return dist


def draw_from(dist):
    return np.random.choice(range(len(dist)), p=dist)


class TestCTMC(unittest.TestCase):

    def check_is_row_stochastic(self, mat):
        for i in range(mat.shape[0]):
            row = mat[i,:]
            self.assertTrue((row >= 0).all())
            self.assertEqual(row.sum(), 1.)

    def _check_q_matrix(self, model):
        Q = model.infinitesimal_generator
        # row sums of Q matrix should be 0
        row_sums = Q.sum(axis=1).round(8)
        self.assertTrue((row_sums == np.zeros((4,), dtype=np.float64)).all())

    def _check_p_matrix(self, model):
        # should get the identity matrix for trans probas when duration is 0
        P0 = model.ctmc_transitions(0)
        self.assertTrue((P0 == np.eye(4)).all())
        # should always get a row stochastic matrix
        t = 1
        Pt = model.ctmc_transitions(1)
        self.check_is_row_stochastic(Pt)
        # limit should be the initial distribution
        Pt = model.ctmc_transitions(100)
        for i in range(4):
            self.assertTrue((Pt[i,:] == model.initial_dist).all())

    def test_jc69(self):
        model = JC69()
        self.assertEqual(model.rate, 4./3)
        self._check_p_matrix(model)
        self._check_q_matrix(model)

    def test_f81(self):
        model = F81([0.1, 0.3, 0.2, 0.4])
        self._check_p_matrix(model)
        self._check_q_matrix(model)

    def test_jc69_corrected_distance(self):
        model = JC69()
        N = 30000
        t = 0.02
        seq0 = [draw_from(model.initial_dist) for _ in range(N)]
        Pt = model.ctmc_transitions(t)
        seq1 = [draw_from(Pt[seq0[i],:]) for i in range(N)]
        estimate = model.corrected_distance(seq0, seq1)
        self.assertAlmostEqual(t, estimate, places=2)

    def test_counts_to_estimates(self):
        # a count matrix where rate of difference is 0.5
        counts = 2 * np.eye(4, dtype=np.int64) + np.ones((4, 4), dtype=np.int64)
        expected_dist = -3/4 * np.log(1/3)  # from JC69 formula
        model = JC69()
        dist, variance = model.counts_to_estimates(counts)
        self.assertAlmostEqual(dist, expected_dist)
        self.assertGreaterEqual(variance, 0.)
        # try again but for longer sequences
        # distance should be the same, variance less
        dist, tighter_variance = model.counts_to_estimates(100 * counts)
        self.assertAlmostEqual(dist, expected_dist)
        self.assertLess(tighter_variance, variance)
        # zero variance when there is zero change
        counts = 4 * np.eye(4, dtype=np.int64)
        dist, variance = model.counts_to_estimates(counts)
        self.assertAlmostEqual(dist, 0.)
        self.assertAlmostEqual(variance, 0.)
