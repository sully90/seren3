from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

'''
Setup.py for c_cic.cpp and python wrappers with openmp enabled
'''

ext_module = Extension(
    "cic",
    ["cic.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language="c++",
)

setup(
    name = 'seren3_cython',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
