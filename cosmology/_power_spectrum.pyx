import numpy as np
cimport numpy as np

def fft_sample_spacing(N, boxsize):
    return _fft_sample_spacing(N, boxsize)

def fft_sample_spacing_components(N):
    return _fft_sample_spacing_components(N)

def window_function(N, p):
    return _window_function(N, p)

def cic_window_function(N):
    return _cic_window_function(N)

cdef _fft_sample_spacing(int N, float boxsize):
    """
    Return the sample spacing in Fourier space, given some symmetric 3D box in real space
    with N elements per dimension and length L.
    See https://gitorious.org/bubble/szclusters/commit/da1402ef95f4d40c28f53f88c99bf079063308c7
    """
    cdef float fac
    cdef np.ndarray kk = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray kx = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray ky = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray kz = np.zeros([N, N, N], dtype=np.float32)
    kx, ky, kz = _fft_sample_spacing_components(N)
    fac = (2. * np.pi / boxsize)
    kk = np.sqrt(kx ** 2. + ky ** 2. + kz ** 2.) * fac
    return kk

cdef _fft_sample_spacing_components(int N):
    """
    Return the sample spacing in Fourier space, given some symmetric 3D box in real space
    with N elements per dimension and length L.
    See https://gitorious.org/bubble/szclusters/commit/da1402ef95f4d40c28f53f88c99bf079063308c7
    """
    cdef int i
    cdef np.ndarray NN = np.zeros(N, dtype=np.int32)
    cdef np.ndarray kx = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray ky = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray kz = np.zeros([N, N, N], dtype=np.float32)

    NN = (N * np.fft.fftfreq(N, 1.)).astype("i")
    for i in range(N):
        kx[NN[i], :, :] = NN[i]
        ky[:, NN[i], :] = NN[i]
        kz[:, :, NN[i]] = NN[i]
    return kx, ky, kz

cdef _window_function(int N, int p):
    ''' Calculate CIC smoothing window function for PS estimation.
    '''
    cdef np.ndarray W = np.zeros([N, N, N], dtype=np.float32)
    cdef kny
    cdef np.ndarray kx = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray ky = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray kz = np.zeros([N, N, N], dtype=np.float32)
    cdef int i, j, k
    cdef float fp
    fp = float(p)
    kny = N/2  # Nyquist frequency
    kx, ky, kz = _fft_sample_spacing_components(N)
    
    #W = ( np.sinc( (np.pi*kx) / (2.*kny) ) * np.sinc( (np.pi*ky) / (2.*kny) ) * np.sinc( (np.pi*kz) / (2.*kny) ) )**p
    W = ( np.sinc( np.pi*kx/2.*kny ) * np.sinc( np.pi*ky/2.*kny ) * np.sinc( np.pi*kz/2.*kny ) ) ** fp
    return W

cdef _cic_window_function(int N):
    ''' Calculate CIC smoothing window function for PS estimation.
    '''
    cdef np.ndarray W = np.zeros([N, N, N], dtype=np.float32)
    cdef kny
    cdef np.ndarray kx = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray ky = np.zeros([N, N, N], dtype=np.float32)
    cdef np.ndarray kz = np.zeros([N, N, N], dtype=np.float32)
    cdef int i, j, k
    kny = N/2  # Nyquist frequency
    kx, ky, kz = _fft_sample_spacing_components(N)
    
    #W = ( np.sinc( (np.pi*kx) / (2.*kny) ) * np.sinc( (np.pi*ky) / (2.*kny) ) * np.sinc( (np.pi*kz) / (2.*kny) ) )**p
    W = ( 1. - (2./3.) * np.sin((np.pi*kx)/(2.*kny))**2 ) * ( 1. - (2./3.) * np.sin((np.pi*ky)/(2.*kny))**2 ) * ( 1. - (2./3.) * np.sin((np.pi*kz)/(2.*kny))**2 )
    return W

