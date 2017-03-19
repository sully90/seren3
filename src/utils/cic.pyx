import cython

import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "c_cic.cpp":
    void run (double * x, double * y, double * z, int NumPart, double L, int N, double * rho)

@cython.boundscheck(False)
@cython.wraparound(False)
def cic(np.ndarray[double, ndim=1, mode="c"] x, np.ndarray[double, ndim=1, mode="c"] y, \
            np.ndarray[double, ndim=1, mode="c"] z, int NumPart, double L, int N, np.ndarray[double, ndim=1, mode="c"] rho):
    run( &x[0], &y[0], &z[0], NumPart, L, N, &rho[0] )  # delta = (rho/rho.mean()) - 1.

    return None
