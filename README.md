# Learning phylogenetic trees as hyperbolic point configurations

Benjamin Wilson, Lateral GmbH, 2021.

This repository contains code to replicate the experiments reported in the [paper](https://arxiv.org/abs/2104.11430).

The experiments ran on an Ubuntu 16 machine. Python 3.7.4, and Numpy
1.17.2 were installed via the [Anaconda3-2019-10](www.anaconda.com) 2019.10 distribution for scientific
Python. GCC version 5.4 was used (by Cython) for compilation.

The experiments should be replicable in any environment running Python >= 3.7
with a recent version of NumPy and Jupyter, and the additional software listed below.

### Cython 0.29.22

Cython is included any Anaconda distribution, otherwise [installable](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) using pip.

### Graphviz 2.40.1 and Python-Graphviz 0.13.2

If using Anaconda, both may be installed with:

```
conda install python-graphviz
```

### PAUP Version 4.0a165 

We used the command-line binary for Linux x86, downloaded from [here](http://phylosolutions.com/paup-test/).  This also required the installation of libpython2.7.  Our Python scripts assume that the PAUP binary is available at `$HOME/paup/paup`, so copy it there.

### PhyML 3.1

Download the binary from [here](http://www.atgc-montpellier.fr/phyml/versions.php) and copy the executables to `$HOME/PhyML-3.1`.

### DendroPy 4.4.0

Downloaded [here](https://dendropy.org/).  We used commit `86d66160b18a4054a31b016e2d1270726775a99f` (on master branch).

### Weighbor 1.2

Obtained [here](https://web.archive.org/web/20140513070758/http://www.t6.lanl.gov/billb/weighbor/download.html) or more specifically, [here](https://web.archive.org/web/20140513070758if_/http://www.t6.lanl.gov/billb/weighbor/weighbor.tar.gz).

To install (you'll need `make` for this):

```
tar xvzf weighbor-1.2.tar.gz
cd Weighbor/
make
```

The resulting executable resided at `$HOME/phylogeny/Weighbor/`.

### BIONJ

Obtained [here](http://www.atgc-montpellier.fr/bionj/download.php) and installed in `~/phylogeny/bionj`.
Compiled with `gcc BIONJ.c -o bionj`.

# Compilation

Once Cython & GCC are installed, the Cython implementation of log-a-like (and also of hyperbolic mds) can be compiled using `./install_cythonised`.
