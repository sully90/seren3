#!/usr/bin/env python
'''
This is the seren3 setup.py for an inplace build. To build the module, just type 'make'
and symlink the seren3 directory to your $PYTHON_PATH
'''

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from distutils.extension import Extension
from Cython.Build import cythonize

ext_module = Extension(
    "utils.cython.cic",
    ["src/utils/cic.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language="c++",
)

def make_config(parent_package='', top_path=None):
    config = Configuration(
        package_name="seren3",
        parent_package=parent_package,
        top_path=top_path)

    config.add_extension(
        name="analysis._interpolate3d",
        sources=["src/analysis/_interpolate3d.c"])

    config.add_extension(
        name="cosmology._power_spectrum",
        sources=["src/cosmology/_power_spectrum.c"])

    return config


def setup_seren():
    setup(
        name="seren",
        version="3.0.0",
        description="Analysis and visualization Python modules for RAMSES",
        keywords='astrophysics visualization amr ramses',
        configuration=make_config,
        ext_modules = cythonize(ext_module)
    )
    return


if __name__ == '__main__':
    setup_seren()
