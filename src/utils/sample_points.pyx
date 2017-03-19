import cython

import numpy as np
cimport numpy as np

cdef extern from "c_sample_points.cpp":
    void c_sample_sphere_surface(double * x, double * y, double * z, double r, int npoints)

@cython.boundscheck(False)
@cython.wraparound(False)
def sample_sphere_surface(np.ndarray[double, ndim=1, mode="c"] x, np.ndarray[double, ndim=1, mode="c"] y, \
                np.ndarray[double, ndim=1, mode="c"] z, double r, int npoints):
    c_sample_sphere_surface(&x[0], &y[0], &z[0], r, npoints)