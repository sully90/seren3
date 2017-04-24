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


def write_config_file():
    '''
    Writes a config file for seren3, if one doesnt exist
    '''
    import os

    cwd = os.getcwd()  # the seren3 directory location
    fname = "%s/._default_seren3_config.txt" % cwd

    if (os.path.isfile(fname) is False):
        # Write a default config file
        import ConfigParser

        config = ConfigParser.ConfigParser()

        # General
        config.add_section("general")
        config.set("general", "verbose", False)

        # Data
        config.add_section("data")
        data_dir = "%s/data/" % cwd
        if (os.path.isdir(data_dir) is False):
            os.mkdir(data_dir)
        config.set("data", "data_dir", data_dir)

        if (os.path.isdir("%s/sims/" % data_dir) is False):
            os.mkdir("%s/sims" % data_dir)
        config.set("data", "sim_dir", "%s/sims/" % data_dir)

        # Halos
        config.add_section("halo")
        config.set("halo", "default_finder", "ctrees")
        config.set("halo", "rockstar_base", "rockstar/")
        config.set("halo", "consistenttrees_base", "rockstar/hlists/")
        config.set("halo", "ahf_base", "AHF/")

        with open(fname, "w") as f:
            config.write(f)


if __name__ == '__main__':
    setup_seren()
    write_config_file()
