from glob import glob
from setuptools import setup, Extension
import numpy as np
import pybind11
from setuptools.command.build_ext import build_ext
import setuptools
import os
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext

include_dirs = [
    # Path to pybind11 headers
    pybind11.get_include(),
    np.get_include()
]

include_dirs.extend(['../include/'])

module = Pybind11Extension('filterdiskann',
             sources=['index_bindings.cpp', '../src/natural_number_map.cpp', '../src/abstract_data_store.cpp', 
                      '../src/logger.cpp', '../src/partition.cpp', #'../src/pq_flash_index.cpp', 
                      '../src/in_mem_graph_store.cpp', '../src/filter_utils.cpp', '../src/ann_exception.cpp', 
                      '../src/distance.cpp', '../src/index_factory.cpp', '../src/math_utils.cpp', 
                      '../src/in_mem_data_store.cpp', '../src/memory_mapper.cpp', '../src/abstract_index.cpp',
                      '../src/scratch.cpp', '../src/pq.cpp', #'../src/linux_aligned_file_reader.cpp', 
                      '../src/utils.cpp', '../src/index.cpp', #'../src/disk_utils.cpp', 
                      '../src/windows_aligned_file_reader.cpp', '../src/natural_number_set.cpp'],
             include_dirs=include_dirs,
            #  libraries=['mysteryann'],
             language='c++')

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """

    # if has_flag(compiler, '-std=c++14'):
        # return '-std=c++14'
    # elif has_flag(compiler, '-std=c++11'):
        # return '-std=c++11'
    if has_flag(compiler, '-std=c++20'):
        return '-std=c++20'
    elif has_flag(compiler, '-std=c++17'):
        return '-std=c++17'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'unix': ['-Ofast', '-fopenmp',
                 '-DNDEBUG', '-march=native', '-mtune=native', '-ftree-vectorize','-Wall', '-DINFO',
                  '-mavx2','-mavx512f', '-mavx512cd', '-mavx512dq', '-mavx512bw', '-mavx512vl',
                  "-mfma", "-msse2", "-fno-builtin-malloc", "-fno-builtin-calloc", 
                  "-fno-builtin-realloc", "-fno-builtin-free", "-fopenmp-simd", "-funroll-loops", "-Wfatal-errors", "-DUSE_AVX2"
                 ],  # , '-w'
    }
    
    # c_opts['unix'].append('-march=native')

    link_opts = {
        'unix': [],
    }

    c_opts['unix'].append("-fopenmp")
    link_opts['unix'].extend(['-fopenmp', '-pthread'])

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        # if ct == 'unix':
            # opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        opts.append(cpp_flag(self.compiler))
            # if has_flag(self.compiler, '-fvisibility=hidden'):
                # opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(self.link_opts.get(ct, []))

        build_ext.build_extensions(self)

setup(name='filterdiskann',
    version='1.0',
    description='Python bindings for Filter-DiskANN class',
    ext_modules=[module],
    cmdclass={'build_ext': BuildExt},
    install_requires=['numpy', 'pybind11']
    )