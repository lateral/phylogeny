import unittest
import subprocess
import os
import re
import dendropy
from dendropy.model.discrete import Jc69, simulate_discrete_chars
from tree_utils import unrooted_tree_topology_invariant
import platform

from tempfile import TemporaryDirectory

executables = {
    'Darwin': 'PhyML-3.1_macOS-MountainLion',
    'Windows': 'PhyML-3.1_win32.exe',
    'Linux': 'PhyML-3.1_linux64'
}

CMD = os.path.expanduser('~/PhyML-3.1/%s' % executables[platform.system()])


def infer_tree(cm, start_tree=None):
    """
    Given a dendropy CharacterMatrix `cm`, infer a tree using PhyML under the
    JC69 substitution model, with a single rate of substitution across all
    sites.  Return the dendropy.Tree, the log likelihood and the number of
    topology changes, as a 3-tuple.
    """
    with TemporaryDirectory() as tmp_dir:
        seqs_file = os.path.join(tmp_dir, 'dna')
        init_tree_file = os.path.join(tmp_dir, 'init_tree.txt')
        tree_file = os.path.join(tmp_dir, 'dna_phyml_tree.txt')
        args = ['-i', seqs_file,
                '-d', 'nt',  # is nucleotide data
                '-n', '1',  # there is only one dataset
                '-b', '0',  # no bootstrap replicates
                '-m', 'JC69',
                '-v', '0',  # there are no invariate sites
                '-s', 'NNI',  # use NNI
                '-o', 'tl',  # optimise both the topology and the branch lengths
                '-c', '1'  # there is only one category of substitution rate in the discrete gamma model
                ]
        if start_tree is not None:
            args += ['-u', init_tree_file]
            with open(init_tree_file, 'w') as f:
                f.write(start_tree.as_string(schema='newick', suppress_edge_lengths=True))

        with open(seqs_file, 'w') as f:
            f.write(cm.as_string(schema='phylip'))
        proc = subprocess.run([CMD] + args, stdout=subprocess.PIPE)
        _output = proc.stdout.decode()

        # get the value of the LL
        matches = re.findall(r'Log likelihood of the current tree: (\S*)\.\n', _output)
        if not len(matches) == 1:
            raise ValueError(_output)
        ll = float(matches[0].strip())

        # count the number of topology changes
        topo_changes = _output.count('[Topology           ]')

        # load the optimum tree
        with open(tree_file, 'r') as f:
            newick = f.read()
        opt_tree = dendropy.Tree.get_from_string(src=newick,
                                                 schema='newick',
                                                 taxon_namespace=cm.taxon_namespace)

    return opt_tree, ll, topo_changes


class Test(unittest.TestCase):

    def test_infer_tree(self):
        gen_tree = dendropy.Tree.get_from_string("(((a:0.1, b:0.1):0.1, c:0.2):0.1, (d:0.2, (e:0.1,f:0.1):0.1):0.1):0;", schema="newick")
        seq_len = 1000  # use nice long sequences, so that it's easy to infer the tree
        cm = simulate_discrete_chars(seq_len, gen_tree, Jc69())
        start_tree = dendropy.Tree.get_from_string("(((a:0.1, f:0.1):0.1, c:0.2):0.1, (d:0.2, (e:0.1,b:0.1):0.1):0.1):0;", schema="newick")
        reconstructed, ll, num_changes = infer_tree(cm, start_tree)
        self.assertEqual(unrooted_tree_topology_invariant(gen_tree),
                         unrooted_tree_topology_invariant(reconstructed))
        self.assertIsInstance(ll, float)
        self.assertLessEqual(ll, 0)
        self.assertIsInstance(num_changes, int)
