import numpy as np
import re, math

heaviside = lambda x: 0.5 if x == 0 else 0 if x < 0 else 1  # step function


def _float_approx_equal(x, y, tol=1e-18, rel=1e-7):
    if tol is rel is None:
        raise TypeError('cannot specify both absolute and relative errors are None')
    tests = []
    if tol is not None: tests.append(tol)
    if rel is not None: tests.append(rel*abs(x))
    assert tests
    return abs(x - y) <= max(tests)


def approx_equal(x, y, *args, **kwargs):
    """approx_equal(float1, float2[, tol=1e-18, rel=1e-7]) -> True|False
    approx_equal(obj1, obj2[, *args, **kwargs]) -> True|False

    Return True if x and y are approximately equal, otherwise False.

    If x and y are floats, return True if y is within either absolute error
    tol or relative error rel of x. You can disable either the absolute or
    relative check by passing None as tol or rel (but not both).

    For any other objects, x and y are checked in that order for a method
    __approx_equal__, and the result of that is returned as a bool. Any
    optional arguments are passed to the __approx_equal__ method.

    __approx_equal__ can return NotImplemented to signal that it doesn't know
    how to perform that specific comparison, in which case the other object is
    checked instead. If neither object have the method, or both defer by
    returning NotImplemented, approx_equal falls back on the same numeric
    comparison used for floats.

    >>> almost_equal(1.2345678, 1.2345677)
    True
    >>> almost_equal(1.234, 1.235)
    False

    """
    if not (type(x) is type(y) is float):
        # Skip checking for __approx_equal__ in the common case of two floats.
        methodname = '__approx_equal__'
        # Allow the objects to specify what they consider "approximately equal",
        # giving precedence to x. If either object has the appropriate method, we
        # pass on any optional arguments untouched.
        for a,b in ((x, y), (y, x)):
            try:
                method = getattr(a, methodname)
            except AttributeError:
                continue
            else:
                result = method(b, *args, **kwargs)
                if result is NotImplemented:
                    continue
                return bool(result)
    # If we get here without returning, then neither x nor y knows how to do an
    # approximate equal comparison (or are both floats). Fall back to a numeric
    # comparison.
    return _float_approx_equal(x, y, *args, **kwargs)



def truncate(number, digits):
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper


def symlog(arr):
    '''
    Returns the symmertirc log of an array
    '''
    symlog_arr = np.zeros(len(arr))
    idx = np.where(arr >= 0.)
    symlog_arr[idx] = np.log10(arr[idx])

    idx = np.where(arr < 0.)
    symlog_arr[idx] = np.log10(np.abs(arr[idx])) * -1.
    return symlog_arr

def aout(zmin, zmax, noutput, zstep=0.001, **cosmo):
    '''
    Function which returns list of expansion factors to output at, evenly spaced in proper time
    '''
    import numpy as np
    import cosmolopy.distance as cd
    from seren3.array import SimArray

    age_func = cd.quick_age_function(1000, 0, zstep, False, **cosmo)
    z_func = cd.quick_redshift_age_function(1000, 0, zstep, **cosmo)

    age_start = age_func(zmin)
    age_end = age_func(zmax)

    age_out = np.linspace(age_start, age_end, noutput)
    z_out = z_func(age_out)
    a_out = 1./(1.+z_out)

    return a_out[::-1]


def log_error(y, y_err):
    '''
    Computes log error
    '''
    return 0.434 * (y_err/y)


def compute_ngpu_aton(ngridx, ngridy, ngridz, levelmin, do_print=True):
    '''
    Computes the number of gpus required for cudaton
    '''
    ngrid = 2**levelmin  # the RAMSES grid size

    ngpu = 1
    for i,n in zip('xyz', (ngridx, ngridy, ngridz)):
        if (ngrid % n) != 0:
            raise Exception("Incompatible grid dimensions for axis %s. %i %% %i != 0" % (i, ngrid, n))
        ngpu *= ngrid/n
    
    if do_print:
        print "%i GPUs required for domain (%i/%i/%i)" % (ngpu, ngridx, ngridy, ngridz)
    return ngpu

def flatten_nested_array(arr):
    import itertools, numpy as np
    return np.array( list(itertools.chain.from_iterable(arr)) )

def unit_vec_r(theta, phi):
    '''
    Unit vector along theta and phi
    '''
    return np.array( [np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)] )

def lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False

def first_above(value, iterable):
    '''
    Returns the value of the iterable which first exceedes value
    '''
    return next(x[0] for x in enumerate(iterable) if x[1] > value)

def is_power2(num):
    'states if a number is a power of two'

    return num != 0 and ((num & (num - 1)) == 0)

def next_greater_power_of_2(x):
    ''' Return the next highest number which is divisible by 2**n '''
    return 2 ** (x - 1).bit_length()

def mass_sph(ilevel, ndim, **cosmo):
    '''
    Returns equivalent sph mass resolution at level ilevel
    '''
    omega_b = cosmo['omega_b_0']
    omega_m = cosmo['omega_M_0']
    mass_sph=omega_b/omega_m*0.5**(ndim*ilevel)
    return mass_sph

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def sphere_from_camera(camera):
    '''
    Return a region filtering sphere from a camera object
    '''
    from pymses.utils.regions import Sphere
    pos = camera.center
    r = camera.region_size[0]/2.

    return Sphere(pos, r)

# Python version of bash which
def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def ic_symlinks(lmin, lmax, path='./'):
    # Run from within ics_ramses_vbc/
    import glob
    import os

    def cd(path):
        print 'Moving to %s' % path
        os.chdir(path)

    def ln(target):
        print 'Linking %s' % target
        os.system("ln -s %s ." % target)

    def unlink(target):
        print 'Uninking %s' % target
        os.system("unlink %s" % target)

    for i in range(lmin, lmax + 1):
        cd("level_%03d/" % i)
        files = glob.glob("../../ics_ramses/level_%03d/ic_*" % i)
        for f in files:
            # print f
            pos = f.find('ic_')
            if os.path.isfile('./%s' % f[pos:]):
                continue
            elif 'posc' in f:
                continue
            ln(f)
        cd('../')


def mem_usage(ngridmax, npartmax, type='all'):
    '''
    Calculate (approx.) mem usage for RAMSES
    TODO - RT
    '''
    if type is 'hydro':
        return (float(ngridmax) / 10. ** 6)
    elif type is 'nbody':
        return 0.7 * (float(ngridmax) / 10. ** 6.) + 0.7 * (float(npartmax) / 10. ** 7.)
    elif type is 'all':
        return 1.4 * (float(ngridmax) / 10. ** 6.) + 0.7 * (float(npartmax) / 10. ** 7.)


def is_power2(num):
    'states if a number is a power of two'

    return num != 0 and ((num & (num - 1)) == 0)


def next_greater_power_of_2(x):
    ''' Return the next highest number which is divisible by 2**n '''
    return 2 ** (x - 1).bit_length()


def divisors(number, mode='print'):
    n = 1
    while(n < number):
        if(number % n == 0):
            if mode is 'print':
                print n
            elif mode is 'yield':
                yield n
        else:
            pass
        n += 1


def prod_sum(iterable):
    ''' Product sum '''
    p = 1
    for n in iterable:
        p *= n
    return p


def int_vbc_pdf(vbc, sigma_vbc, b=np.inf):
    ''' Integrate the PDF to compute P(>vbc) '''
    from scipy.integrate import quad
    return quad(vbc_pdf, vbc, b, args=(sigma_vbc))


def vbc_pdf(v, sigma_vbc):
    return ((3.) / (2. * np.pi * (sigma_vbc ** 2.))) ** (3. / 2.) * 4. * np.pi * (v ** 2.) * np.exp(-((3. * v ** 2) / (2. * sigma_vbc ** 2)))


def deconvolve(field, N, p):
    ''' Deconvolve CIC kernel from field
    N - grid size in cells '''
    from seren3.cosmology import _power_spectrum
    W = _power_spectrum.window_function(N, p)

    import scipy.fftpack as fft
    ft = fft.fftn(field)
    ft /= np.sqrt(W)
    return fft.ifftn(ft).real


def deconvolve_cic(field, N):
    ''' Deconvolve CIC kernel from field
    N - grid size in cells '''
    from seren3.cosmology import _power_spectrum
    W = _power_spectrum.cic_window_function(N)

    import scipy.fftpack as fft
    ft = fft.fftn(field)
    ft /= np.sqrt(W)
    return fft.ifftn(ft).real


def ncols(n, cmap='rainbow'):
    import matplotlib.pylab as plt
    cmap = plt.get_cmap(cmap)
    return cmap(np.linspace(0, 1, n))


def vec_mag(vec):
    return np.sqrt(vec[:, 0] * vec[:, 0] + vec[:, 1] * vec[:, 1] + vec[:, 2] * vec[:, 2])

def cic(source, nn, field, gaussian_smooth=False, norm_pos=False, masked=False, mask_val=0.0, deconvolve=False, **kwargs):
    from cython import particle_utils as pu
    '''
    CIC smooth a discrete or continuous field

    Parameters:
            * source (seren source): object with the method flatten() to return all particles/cells
            * nn (int): number of cells per dimension
            * field (string): field to smooth (should be able to access by source.flatten()[field])
            * gaussian_smooth (boolen, False): perform gaussian smoothing on resulting field
            * kwargs (dict): args for the CIC
            For mass -> density fields, set average=False in kwargs
    '''

    if hasattr(source, 'has_key') and (field in source):
        field = source[field]
    elif hasattr(source, 'fields') and (field in source.fields):
        field = source.fields[field]
    else:
        field = source.flatten()[field]

    pos = None
    if hasattr(source, 'points'):
        pos = source.points.copy()
    elif hasattr(source, 'has_key') and ('pos' in source):
        pos = source['pos'].copy()
    elif hasattr(source, 'fields') and ('pos' in source.fields):
        pos = source.fields[field].copy()
    else:
        raise Exception("pos field not found")
    # pos = source.points * nn
    #pos = source.points.copy()

    if norm_pos:
        for i in [0, 1, 2]:
            tmp = (pos[:, i] - pos[:, i].min()) / \
                (pos[:, i].max() - pos[:, i].min())
            pos[:, i] = tmp

    pos *= nn

    if len(field.shape) > 1:  # vector
        field = vec_mag(field)

    cic_field = pu._cic(field, pos, nn, smooth=gaussian_smooth, **kwargs)
    if masked:
        if deconvolve:
            return deconvolve_cic(np.ma.masked_equal(cic_field, mask_val), nn)
        return np.ma.masked_equal(cic_field, mask_val)

    if deconvolve:
        return deconvolve_cic(cic_field)
    return cic_field


def amr_mesh_grid(nn):
    '''
    Return a numpy meshgrid for point sampling
    nn - number of cells per dimension
    '''

    x, y, z = np.mgrid[
        0:1:complex(nn), 0:1:complex(nn), 0:1:complex(nn)]

    # Reshape
    npoints = np.prod(x.shape)
    x1 = np.reshape(x, npoints)
    y1 = np.reshape(y, npoints)
    z1 = np.reshape(z, npoints)

    # Arrange for pymses
    pxyz = np.array([x1, y1, z1])
    pxyz = pxyz.transpose()

    return pxyz
