import os
import platform
import glob
from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension


def set_gcc():
    """
    Try to find and use GCC on OSX for OpenMP support, setting the CC
    environment variable accordingly.
    """
    # For macports and homebrew
    patterns = ['/opt/local/bin/gcc-mp-[0-9].[0-9]',
                '/opt/local/bin/gcc-mp-[0-9]',
                '/usr/local/bin/gcc-[0-9].[0-9]',
                '/usr/local/bin/gcc-[0-9]',
                '/usr/bin/gcc']

    gcc_binaries = []
    for pattern in patterns:
        gcc_binaries += glob.glob(pattern)
    gcc_binaries.sort()

    if gcc_binaries:
        _, gcc = os.path.split(gcc_binaries[-1])
        os.environ["CC"] = gcc

    else:
        raise Exception('No GCC available. Install gcc from Homebrew '
                        'using brew install gcc.')

if 'darwin' in platform.platform().lower():
    set_gcc()

args = {'extra_compile_args': ['-ffast-math', '-O3'],
        'extra_link_args': []}
extensions = [
    Extension('cythonised.logalike', ['cythonised/logalike.pyx'], **args),
    Extension('cythonised.hyperbolic_mds', ['cythonised/hyperbolic_mds.pyx'], **args),
]
setup(ext_modules=cythonize(extensions, language_level='3'))
