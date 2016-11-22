from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# setup(ext_modules = cythonize(
#                 "sample_points.pyx",                 # our Cython source
#                 extra_compile_args=['-fopenmp'],
#                 extra_link_args=['-fopenmp'],
#                 sources=["c_cic.cpp"],  # additional source file(s)
#                 language="c++",             # generate C++ code
#                 ))



ext_module = Extension(
    "cic",
    ["cic.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language="c++",
)

setup(
    name = 'seren2_cython',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
