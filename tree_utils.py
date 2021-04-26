import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import re
import numpy as np
import dendropy
import itertools
import unittest
import subprocess
import random
import os
import pandas as pd
from io import StringIO
from collections import Counter, defaultdict
from tempfile import TemporaryDirectory
from dendropy.simulate import treesim
from dendropy.datamodel.treemodel import Tree
from dendropy.model.discrete import Jc69, simulate_discrete_chars
from dendropy.calculate.treecompare import symmetric_difference
from dendropy.calculate.phylogeneticdistance import PhylogeneticDistanceMatrix as PDM
from ctmc import JC69

from io import StringIO
from graphviz import Source


def array_to_pdm(dists, taxon_namespace):
    """
    Parameters
    ----------
    taxon_namespace : defines the ordering of the leaves
    dists : a 2d NumPy array of interleaf distances.
    """
    labels = taxon_namespace.labels()
    assert dists.shape == (len(labels), len(labels))
    df = pd.DataFrame(dists, index=labels, columns=labels)
    _out = StringIO()
    df.to_csv(_out)
    _out.seek(0)
    return PDM.from_csv(_out, taxon_namespace=taxon_namespace)


def pdm_to_array(pdm):
    """
    Return a 2d numpy array containing the patristic_distance of each pair
    of leaves.  The order of the leaves is that given by
    leaves_and_labels(pdm).
    """
    leaves, _ = leaves_and_labels(pdm)
    dists = np.zeros((len(leaves), len(leaves)), dtype=np.float64)
    for i0, l0 in enumerate(leaves):
        for i1, l1 in enumerate(leaves):
            dists[i0, i1] = pdm.patristic_distance(l0, l1)
    return dists


def leaves_and_labels(pdm):
    """
    Return the lists of the leaves and labels, where the order is that
    defined by the TaxonNamespace.
    """
    leaves = [taxon for taxon in pdm.taxon_namespace]
    labels = [leaf.label for leaf in leaves]
    return leaves, labels


def to_phylip(pdm):
    """
    Return a string representing a Phylip compatible encoding of the
    provided distance matrix.
    """
    _, taxon_labels = leaves_and_labels(pdm)
    dists = pdm_to_array(pdm)
    _out = '%i' % len(taxon_labels)
    lines = [_out]
    for label, dists_row in zip(taxon_labels, dists):
        dists_row_str = ' '.join([str(val) for val in dists_row])
        lines.append('%s\t%s' % (label, dists_row_str))
    return '\n'.join(lines)


def unrooted_tree_topology_invariant(tree):
    """
    Two trees have the same topology after unrooting i.f.f.
    unrooted_tree_topology_invariant(tree1) == unrooted_tree_topology_invariant(tree2)
    See https://dendropy.org/primer/bipartitions.html
    """
    tree = dendropy.Tree(tree)  # makes a clone
    tree.deroot()
    tree.update_bipartitions()
    return tuple(sorted([be.split_bitmask for be in tree.bipartition_encoding]))


def robinson_foulds(tree1, tree2):
    """
    Returns the RF distance between the two trees provided, after derooting.
    Does not change the trees passed.
    Does not halve the distance as some do - so the maximal distance between
    unrooted trees with 4 leaves is 2.
    NOTE: This function included to force the arguments of symmetric_difference
    to be unrooted, see:
    https://github.com/jeetsukumaran/DendroPy/issues/109
    """
    # clone trees, retaining only the taxonnamespace and taxon instances
    tree1 = tree1.clone(1)
    tree2 = tree2.clone(1)
    # make the copies unrooted
    tree1.deroot()
    tree2.deroot()
    return symmetric_difference(tree1, tree2)


def draw_tree(seed, taxon_namespace):
    """
    Generate and return a random rooted tree using a pure birth process.
    Returns an instance of dendropy.datamodel.treemodel.Tree.
    The unrooted topology of these rooted trees are uniformly distributed.
    """
    rng = random.Random(seed)
    n_leaves = len(taxon_namespace)
    tree = treesim.birth_death_tree(birth_rate=1,
                                    death_rate=0,
                                    num_total_tips=n_leaves,
                                    rng=rng,
                                    taxon_namespace=taxon_namespace)
    return tree


def random_agglomeration(node_names, branch_length_generator=np.random.rand):
    """
    Return a random tree with branch lengths drawn from repeated calls to
    `branch_length_generator`.  Which nodes are agglomerated at each step
    is determined by np.random.shuffle.
    Returns a Newick encoding.
    The unrooted topology of these rooted trees are uniformly distributed.
    """
    nodes = list(node_names)
    while len(nodes) > 1:
        np.random.shuffle(nodes)
        left = nodes.pop()
        right = nodes.pop()
        left_length = branch_length_generator()
        right_length = branch_length_generator()
        parent = '(%s:%f,%s:%f)' % (left, left_length, right, right_length)
        nodes.append(parent)
    return nodes[0] + ';'


def to_graphviz(tree):
    """
    Given a dendropy.Tree instance `tree`, return a graphviz object depicting
    it that auto-displays in the Jupyter notebook.
    """
    _out = StringIO()
    tree.write_as_dot(_out)
    graph = Source(_out.getvalue())
    return graph


def weighbor(pdm, sequence_length):
    """
    Return a the unrooted Tree built by Weighbor, given the pairwise distances
    as a PhylogeneticDistanceMatrix.
    """
    _dist_mat_phylip = to_phylip(pdm)
    cmd = os.path.expanduser('~/phylogeny/Weighbor/weighbor')
    args = ('-L %i -b 4' % sequence_length).split(' ')  # uses JC69
    proc = subprocess.run([cmd] + args,
                          input=_dist_mat_phylip.encode(),
                          stdout=subprocess.PIPE)
    newick = proc.stdout.decode().strip()
    return Tree.get_from_string(src=newick, schema='newick',
                                taxon_namespace=pdm.taxon_namespace,
                                rooting="default-unrooted")


def bionj(pdm):
    """
    Return a the unrooted Tree built by BIONJ, given the pairwise distances
    as a PhylogeneticDistanceMatrix.
    The variances are assumed to be equal to the distances.
    """
    cmd = os.path.expanduser('~/phylogeny/bionj/bionj')
    with TemporaryDirectory() as tmp_dir:
        dists_file = os.path.join(tmp_dir, 'infile')
        tree_file = os.path.join(tmp_dir, 'outfile')
        args = [dists_file, tree_file]

        with open(dists_file, 'w') as f:
            f.write(to_phylip(pdm))
        proc = subprocess.run([cmd] + args, stdout=subprocess.PIPE)
        with open(tree_file, 'r') as f:
            newick = f.read()

    return Tree.get_from_string(src=newick, schema='newick',
                                taxon_namespace=pdm.taxon_namespace,
                                rooting="default-unrooted")


def character_matrix_to_counts(cm):
    """
    Given a dendropy CharacterMatrix `cm`, return a 4d array
        counts[i, j, x, y]
    giving the gene_x-gene_y transition counts for sequence i to sequence j.
    """
    labels = [taxon.label for taxon in cm.taxon_namespace]
    lookup = dict(zip('ACGT', range(4)))
    def encode_sequence(dendropy_seq):
        return [lookup[char] for char in str(dendropy_seq)]
    sequences = [encode_sequence(cm[label]) for label in labels]
    n_species = len(sequences)
    n_sites = len(sequences[0])
    counts = np.zeros((n_species, n_species, 4, 4), dtype=np.int64)
    for i in range(n_species):
        for j in range(n_species):
            for site in range(n_sites):
                counts[i, j, sequences[i][site], sequences[j][site]] += 1
    return counts


def rate_of_difference(counts):
    """
    Given a 2d matrix of transition counts, return the rate of difference.
    """
    N = counts.sum()
    same = np.trace(counts)
    diff = N - same
    return diff / N


def counts_to_rates(counts):
    """
    Given a 4d array
        counts[i, j, x, y]
    giving the nucleotide_x-nucleotide_y transition counts for sequence i to
    sequence j, return the 2d rates-of-difference matrix.
    """
    rates = np.zeros(counts.shape[:2], dtype=np.float64)
    for i in range(rates.shape[0]):
        for j in range(rates.shape[1]):
            rates[i,j] = rate_of_difference(counts[i,j])
    return rates


def counts_to_estimates(counts):
    """
    Given a 4d array
        counts[i, j, x, y]
    giving the nucleotide_x-nucleotide_y transition counts for sequence i to
    sequence j, return a tuple of 2d arrays giving the distance estimates and
    their sample variances according to the JC69 model.
    """
    num_leaves = counts.shape[0]
    est_dists = np.zeros((num_leaves, num_leaves), dtype=np.float64)
    variances = np.zeros((num_leaves, num_leaves), dtype=np.float64)
    mm = JC69()
    for i in range(num_leaves):
        for j in range(num_leaves):
            est_dist, variance = mm.counts_to_estimates(counts[i,j])
            est_dists[i,j] = est_dist
            variances[i,j] = variance
    return est_dists, variances


def generate_character_matrix(tree, sequence_length, seed):
    rng = random.Random(seed)
    mutn_model = Jc69(rng=rng)
    cm = simulate_discrete_chars(sequence_length,
                                 tree,
                                 mutn_model,
                                 rng=rng)
    return cm


# FIXME needed?
def generate_jc69_sequences(tree, sequence_length, seed):
    """
    Generate sequences of homologous genes using the tree model defined by the
    tree.
    `tree` is an instance of dendropy.datamodel.treemodel.Tree.
    Returns a list of sequences of ints with values in 0 .. 3, encoding A, C,
    G, T (resp.).
    The order of the sequences is that of tree.leaf_nodes().
    """
    rng = random.Random(seed)
    mutn_model = Jc69(rng=rng)
    cm = simulate_discrete_chars(sequence_length,
                                 tree,
                                 mutn_model,
                                 rng=rng)
    leaf_labels = [node.taxon.label for node in tree.leaf_nodes()]
    lookup = dict(zip('ACGT', range(4)))
    def encode_sequence(dendropy_seq):
        return [lookup[char] for char in str(dendropy_seq)]
    sequences = [encode_sequence(cm[label]) for label in leaf_labels]
    return sequences


class Test(unittest.TestCase):

    def _arrays_are_equal(self, arr1, arr2):
        return (arr1 == arr2).all()

    def test_weighbor(self):
        newick = "(((a:1, b:1):1, c:2):1, (d:2, (e:1,f:1):1):1):0;"
        tree = Tree.get_from_string(newick, schema="newick")
        reconstructed = weighbor(tree.phylogenetic_distance_matrix(), 500)
        self.assertFalse(reconstructed.rooting_state_is_undefined)
        self.assertFalse(reconstructed.is_rooted)
        self.assertEqual(symmetric_difference(reconstructed, tree), 0)

    def test_bionj(self):
        newick = "(((a:1, b:1):1, c:2):1, (d:2, (e:1,f:1):1):1):0;"
        tree = Tree.get_from_string(newick, schema="newick")
        reconstructed = bionj(tree.phylogenetic_distance_matrix())
        self.assertFalse(reconstructed.rooting_state_is_undefined)
        self.assertFalse(reconstructed.is_rooted)
        self.assertEqual(symmetric_difference(reconstructed, tree), 0)

    def test_character_matrix_to_counts(self):
        newick = "(((a:0.1, b:0.1):0.1, c:0.2):0.1, (d:0.2, (e:0.1,f:0.1):0.1):0.1):0;"
        tree = Tree.get_from_string(newick, schema="newick")
        seq_len = 1000
        cm = simulate_discrete_chars(seq_len, tree, Jc69())
        counts = character_matrix_to_counts(cm)
        self.assertEqual(counts.shape, (6, 6, 4, 4))
        self.assertEqual(counts[0, 1, :, :].sum(), seq_len)

    def test_generate_jc69_sequences(self):
        src = '[&R] ((T3:1.5,(T1:0.4,T2:0.5):0.75):0.1,T4:5.4):0.2;'
        tree = Tree.get_from_string(src=src, schema='newick')
        length = 20
        permissible_genes = range(4)
        sequences = generate_jc69_sequences(tree, length, 1)

        # check format of sequences
        self.assertEqual(len(sequences), len(tree.leaf_nodes()))
        for sequence in sequences:
            self.assertEqual(len(sequence), length)
            self.assertEqual(type(sequence), list)
            for gene in sequence:
                self.assertIn(gene, permissible_genes)

        # different seed => different sequences
        different_sequences = generate_jc69_sequences(tree, length, 2)
        self.assertNotEqual(different_sequences, sequences)
        # same seed => same sequences
        same_sequences = generate_jc69_sequences(tree, length, 1)
        self.assertEqual(same_sequences, sequences)

    def test_robinson_foulds(self):
        tns = dendropy.TaxonNamespace('ABCD')
        # with some simple four taxon trees
        tree1 = Tree.get_from_string(src='((A,B),(C,D));', schema='newick', taxon_namespace=tns)
        tree2 = Tree.get_from_string(src='((A,C),(B,D));', schema='newick', taxon_namespace=tns)
        tree3 = Tree.get_from_string(src='(A,(B,(C,D)));', schema='newick', taxon_namespace=tns)  # same as tree1
        self.assertEqual(robinson_foulds(tree1, tree2), 2)
        self.assertEqual(robinson_foulds(tree1, tree1), 0)
        self.assertEqual(robinson_foulds(tree1, tree3), 0)

    def test_robinson_foulds_root_degree_invariance(self):
        tns = dendropy.TaxonNamespace('ABCD')
        # with some simple four taxon trees
        tree1 = Tree.get_from_string(src='((A,B),C,D);', schema='newick', taxon_namespace=tns)
        tree2 = Tree.get_from_string(src='(((A,B),C),D);', schema='newick', taxon_namespace=tns)
        self.assertEqual(robinson_foulds(tree1, tree2), 0)

    def test_robinson_foulds_larger(self):
        """
        T3 = (0, ((1, (2, 3)), (7, (6, (4, 5)))))
        T4 = (0, ((2, (1, 3)), (6, (4, (5, 7)))))
        should have RF dist 6, according to https://www.cs.hmc.edu/~hadas/mitcompbio/treedistance.html
        (they halve it to get 3)
        """
        tns = dendropy.TaxonNamespace(''.join([str(i) for i in range(8)]))
        tree3 = Tree.get_from_string(src='(0, ((1, (2, 3)), (7, (6, (4, 5)))));', schema='newick', taxon_namespace=tns)
        tree4 = Tree.get_from_string(src='(0, ((2, (1, 3)), (6, (4, (5, 7)))));', schema='newick', taxon_namespace=tns)
        self.assertEqual(robinson_foulds(tree3, tree4), 6)

    def test_unrooted_tree_topology_invariant(self):
        tns = dendropy.TaxonNamespace('ABCDE')
        # with some simple four taxon trees
        tree1 = Tree.get_from_string(src='((A,B),(C,D));', schema='newick', taxon_namespace=tns)
        tree2 = Tree.get_from_string(src='((A,C),(B,D));', schema='newick', taxon_namespace=tns)
        invariant1 = unrooted_tree_topology_invariant(tree1)
        invariant2 = unrooted_tree_topology_invariant(tree2)
        self.assertEqual(invariant1, invariant1)
        self.assertNotEqual(invariant1, invariant2)
        # put in all 5 taxon trees, check that we get the expected number of
        # invariants = number of unrooted tree topologies = 15
        # all these trees can be constructed by permuting the labels of
        #  A   E   C
        #   \  |  /
        #    --+--
        #   /     \
        #  B       D
        template = '(((%s,%s),%s),(%s,%s));'
        five_taxon_trees = [dendropy.Tree.get_from_string(src=template % taxa, schema='newick', taxon_namespace=tns) for taxa in itertools.permutations('ABCDE')]
        five_taxon_invariants = set([unrooted_tree_topology_invariant(tree) for tree in five_taxon_trees])
        self.assertEqual(len(five_taxon_invariants), 15)

    def test_draw_tree_rootedness(self):
        n_leaves = 10
        tns = dendropy.TaxonNamespace(range(n_leaves))
        tree = draw_tree(seed=1, taxon_namespace=tns)
        self.assertFalse(tree.rooting_state_is_undefined)
        self.assertTrue(tree.is_rooted)

    def test_draw_tree_seeding(self):
        n_leaves = 30  # with this many leaves, improbable that two draws have same unrooted topology
        tns = dendropy.TaxonNamespace(range(n_leaves))
        tree1 = draw_tree(seed=1, taxon_namespace=tns)
        self.assertEqual(len(tree1.leaf_nodes()), n_leaves)
        # check that distinct random seeds give distinct trees
        tree2 = draw_tree(seed=2, taxon_namespace=tns)
        self.assertNotEqual(unrooted_tree_topology_invariant(tree1), unrooted_tree_topology_invariant(tree2))
        # check that the same seed returns the same tree
        tree1_again = draw_tree(seed=1, taxon_namespace=tns)
        self.assertEqual(unrooted_tree_topology_invariant(tree1), unrooted_tree_topology_invariant(tree1_again))

    def test_draw_tree_unrooted_uniformity(self):
        n_leaves = 5
        tns = dendropy.TaxonNamespace(range(n_leaves))
        n_samples = 5000
        n_topologies = 15
        samples = [unrooted_tree_topology_invariant(draw_tree(seed=i, taxon_namespace=tns)) for i in range(n_samples)]
        counts = Counter(samples)
        self.assertEqual(len(counts), n_topologies)
        self.assertGreater(min(counts.values()), (0.85 / n_topologies) * n_samples)
        self.assertLess(max(counts.values()), (1.15 / n_topologies) * n_samples)

    def test_random_agglomeration_unrooted_uniformity(self):
        n_leaves = 5
        node_names = ['T%04d' % i for i in range(n_leaves)]
        tns = dendropy.TaxonNamespace(node_names)
        n_samples = 5000
        n_topologies = 15
        trees = [dendropy.Tree.get_from_string(src=random_agglomeration(node_names), schema='newick', taxon_namespace=tns) for _ in range(n_samples)]
        invs = [unrooted_tree_topology_invariant(tree) for tree in trees]
        counts = Counter(invs)
        self.assertEqual(len(counts), n_topologies)
        self.assertGreater(min(counts.values()), (0.85 / n_topologies) * n_samples)
        self.assertLess(max(counts.values()), (1.15 / n_topologies) * n_samples)

    def test_array_to_pdm(self):
        tree = dendropy.Tree.get_from_string("(((a:1.1, b:1):1, c:2):1, (d:2, (e:1,f:1):1):1):0;", schema="newick")
        # construct a NumPy array of distances
        _pdm = tree.phylogenetic_distance_matrix()
        # now try to get it back
        pdm = array_to_pdm(
            pdm_to_array(_pdm),
            _pdm.taxon_namespace)
        # check a distance
        namespace = tree.taxon_namespace
        self.assertTrue(pdm.distance(namespace.get_taxon('a'), namespace.get_taxon('c')), 4.1)

    def test_to_phylip(self):
        tree = dendropy.Tree.get_from_string("(((a:1.1, b:1):1, c:2):1, (d:2, (e:1,f:1):1):1):0;", schema="newick")
        _out = to_phylip(tree.phylogenetic_distance_matrix())
        lines = _out.split('\n')
        # some pretty weak checks, but better than nothing!
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[0], '6')
