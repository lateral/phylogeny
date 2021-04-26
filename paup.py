import unittest
import subprocess
import os
import re
import dendropy
from dendropy.model.discrete import Jc69, simulate_discrete_chars
from tree_utils import unrooted_tree_topology_invariant, generate_character_matrix, draw_tree, robinson_foulds
from collections import defaultdict

from tempfile import TemporaryDirectory

MLS_PREFIX = """
Begin paup;
[PAUP batch file.  Run with ./paup -n filename]
set autoclose=yes warntree=no warnreset=no maxtrees=1000; [maxtrees is the maximum number of trees to store in memory]
execute %s; [load the characters]
set criterion=likelihood;
lset nst=1 basefreq=equal; [specify use JC69]
"""

MLS_WITH_START_TREE = """
execute %s; [load the starting tree - it will get id 1]
hsearch swap=nni start=1; [start at our starting tree (it has id 1)]
"""

MLS_NO_START_TREE = 'hsearch swap=nni;'

MLS_POSTFIX = """
savetrees file=%s brlens replace; [brlens=include the branch lengths]
quit;
end;
"""

CHARS_FN = 'characters.nexus'
INPUT_TREE_FN = 'input_tree.nexus'
OUTPUT_FN = 'best.tree'
BATCH_FILE_FN = 'batch.paup'

CMD = os.path.expanduser('~/paup/paup')


def build_mls_paup_input(characters_file, output_file, start_tree_file=None):
    """
    Return a PAUP command for running maximum likelihood tree search on the
    given data, optionally at the specified start tree.

    Uses NNI moves for the search.
    """
    cmd = MLS_PREFIX % characters_file
    if start_tree_file is not None:
        cmd += '\n' + MLS_WITH_START_TREE % start_tree_file
    else:
        cmd += '\n' + MLS_NO_START_TREE
    cmd += '\n' + MLS_POSTFIX % output_file
    return cmd


def infer_tree(cm, start_tree=None):

    with TemporaryDirectory() as tmp_dir:

        with open(os.path.join(tmp_dir, CHARS_FN), 'w') as f:
            f.write(cm.as_string(schema='nexus'))

        if start_tree is not None:
            with open(os.path.join(tmp_dir, INPUT_TREE_FN), 'w') as f:
                start_tree_copy = dendropy.Tree(start_tree)
                start_tree_copy.deroot()  # starting trees must be unrooted
                f.write(start_tree_copy.as_string(schema='nexus', suppress_taxa_blocks=True))

        with open(os.path.join(tmp_dir, BATCH_FILE_FN), 'w') as f:
            _paup_instructions = build_mls_paup_input(CHARS_FN,
                                                      OUTPUT_FN, 
                                                      INPUT_TREE_FN if start_tree is not None else None)
            f.write(_paup_instructions)

        proc = subprocess.run([CMD, '-n', BATCH_FILE_FN],
                              stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE, cwd=tmp_dir, check=True)

        # load the optimum tree
        with open(os.path.join(tmp_dir, OUTPUT_FN)) as f:
            best_tree = f.read()
        opt_tree = dendropy.Tree.get_from_string(src=best_tree,
                                                 schema='nexus',
                                                 taxon_namespace=cm.taxon_namespace)

        # get the value of the LL
        matches = re.findall(r'Score of best tree\(s\) found = (\S*)\n', best_tree)
        if not len(matches) == 1:
            raise ValueError(best_tree)
        nll = float(matches[0].strip())

        # count the number of topology changes
        matches = re.findall(r'Total number of rearrangements tried = (\S*)\n', best_tree)
        if not len(matches) == 1:
            raise ValueError(best_tree)
        topo_changes = int(matches[0].strip())

    return opt_tree, -nll, topo_changes



LL_TEMPLATE = """
Begin paup;
[PAUP batch file.  Run with ./paup -n filename]
set autoclose=yes warntree=no warnreset=no storebrlens;
execute %s; [load the characters]
execute %s; [load the tree]
lset nst=1 basefreq=equal; [specify use JC69]
lscores / userbrlen; [calculate the log likelihood using our given branch lengths]
quit;
end;
"""

def get_log_likelihood(tree, cm):
    """
    Given a dendropy CharacterMatrix `cm`, and a dendropy Tree `tree`, return
    the log likelihood.
    """
    with TemporaryDirectory() as tmp_dir:

        with open(os.path.join(tmp_dir, CHARS_FN), 'w') as f:
            f.write(cm.as_string(schema='nexus'))

        with open(os.path.join(tmp_dir, INPUT_TREE_FN), 'w') as f:
            tree = dendropy.Tree(tree)
            tree.deroot()  # starting trees must be unrooted
            f.write(tree.as_string(schema='nexus', suppress_taxa_blocks=True))

        with open(os.path.join(tmp_dir, BATCH_FILE_FN), 'w') as f:
            f.write(LL_TEMPLATE % (CHARS_FN, INPUT_TREE_FN))

        proc = subprocess.run([CMD, '-n', BATCH_FILE_FN],
                              stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE, cwd=tmp_dir, check=True)
        _output = proc.stdout.decode()

        # get the value of the LL
        matches = re.findall(r'^-ln L +(\S*)\n', _output, flags=re.MULTILINE)
        if not len(matches) == 1:
            raise ValueError(_output)
        nll = float(matches[0].strip())
        return -1 * nll


BEST_LL_TEMPLATE = """
Begin paup;
[PAUP batch file.  Run with ./paup -n filename]
set autoclose=yes warntree=no warnreset=no;
execute %s; [load the characters]
execute %s; [load the tree]
set criterion=likelihood;
lset nst=1 basefreq=equal; [specify use JC69]
lscores;
savetrees file=%s brlens replace; [write out the tree with the tuned branch lengths]
quit;
end;
"""

def get_best_log_likelihood_for_topology(tree, cm):
    """
    Given a dendropy CharacterMatrix `cm`, and a dendropy Tree `tree`, tune the
    branch lengths of the tree, returning the log likelihood of the tuned tree
    and the tuned tree (as a 2-tuple).
    Post: robinson_foulds(tree, returned_tree) == 0
    """
    with TemporaryDirectory() as tmp_dir:

        with open(os.path.join(tmp_dir, CHARS_FN), 'w') as f:
            f.write(cm.as_string(schema='nexus'))

        with open(os.path.join(tmp_dir, INPUT_TREE_FN), 'w') as f:
            tree = dendropy.Tree(tree)
            tree.deroot()  # starting trees must be unrooted
            f.write(tree.as_string(schema='nexus', suppress_taxa_blocks=True))

        with open(os.path.join(tmp_dir, BATCH_FILE_FN), 'w') as f:
            f.write(BEST_LL_TEMPLATE % (CHARS_FN, INPUT_TREE_FN, OUTPUT_FN))

        proc = subprocess.run([CMD, '-n', BATCH_FILE_FN],
                              stderr=subprocess.PIPE,
                              stdout=subprocess.PIPE, cwd=tmp_dir, check=True)
        _output = proc.stdout.decode()

        # get the value of the LL
        matches = re.findall(r'^-ln L +(\S*)\n', _output, flags=re.MULTILINE)
        if not len(matches) == 1:
            raise ValueError(_output)
        nll = float(matches[0].strip())

        # load the tuned tree
        with open(os.path.join(tmp_dir, OUTPUT_FN)) as f:
            tuned_tree_nexus = f.read()
        tuned_tree = dendropy.Tree.get_from_string(src=tuned_tree_nexus,
                                                   schema='nexus',
                                                   taxon_namespace=cm.taxon_namespace)
        return -nll, tuned_tree


class Test(unittest.TestCase):

    def test_infer_tree(self):
        gen_tree = dendropy.Tree.get_from_string("(((a:0.1, b:0.1):0.1, c:0.2):0.1, (d:0.2, (e:0.1,f:0.1):0.1):0.1):0;", schema="newick")
        seq_len = 1000  # use nice long sequences, so that it's easy to infer the tree
        cm = simulate_discrete_chars(seq_len, gen_tree, Jc69())
        reconstructed, ll, num_changes = infer_tree(cm, start_tree=None)
        self.assertEqual(unrooted_tree_topology_invariant(gen_tree),
                         unrooted_tree_topology_invariant(reconstructed))
        self.assertIsInstance(ll, float)
        self.assertLessEqual(ll, 0)
        self.assertIsInstance(num_changes, int)

    def test_infer_tree_with_starting_tree(self):
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

    def test_get_best_log_likelihood_for_topology(self):
        newick = "(((a:0.1, b:0.1):0.1, c:0.2):0.1, (d:0.2, (e:0.1,f:0.1):0.1):0.1):0;"
        for prefix in ['[&U] ', '[&R] ', '']:
            tree = dendropy.Tree.get_from_string(prefix + newick, schema="newick")
            seq_len = 100
            cm = simulate_discrete_chars(seq_len, tree, Jc69())
            ll, output_tree = get_best_log_likelihood_for_topology(tree, cm)
            self.assertIsInstance(ll, float)
            self.assertEqual(robinson_foulds(tree, output_tree), 0)
            self.assertLessEqual(ll, 0)

    def test_get_best_likelihood_stability(self):
        num_leaves = 4
        node_names = ['T%02d' % i for i in range(num_leaves)]
        tns = dendropy.TaxonNamespace(node_names)
        cm = generate_character_matrix(draw_tree(0, tns), 2000, 1)

        # draw some random trees and check the best likelihood of their topologies
        num_samples = 20
        results = defaultdict(lambda: [])
        for i in range(num_samples):
            tree = draw_tree(seed=i, taxon_namespace=tns)
            inv = unrooted_tree_topology_invariant(tree)
            ll, output_tree = get_best_log_likelihood_for_topology(tree, cm)
            results[inv].append(ll)
        # there are only three unrooted topologies, so there should be exactly that many distinct keys
        self.assertEqual(len(results), 3)
        # for each, there should be only one distinct value
        for _, lls in results.items():
            self.assertEqual(len(set(lls)), 1)

    def test_get_log_likelihood(self):
        num_leaves = 4
        node_names = ['T%02d' % i for i in range(num_leaves)]
        tns = dendropy.TaxonNamespace(node_names)
        tree = draw_tree(0, tns)
        cm = generate_character_matrix(tree, 2000, 1)
        ll = get_log_likelihood(tree, cm)
        self.assertLessEqual(ll, 0)
