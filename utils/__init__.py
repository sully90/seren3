import numpy as np
import re

heaviside = lambda x: 0.5 if x == 0 else 0 if x < 0 else 1  # step function

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
    from seren2.cosmology import _power_spectrum
    W = _power_spectrum.window_function(N, p)

    import scipy.fftpack as fft
    ft = fft.fftn(field)
    ft /= np.sqrt(W)
    return fft.ifftn(ft).real


def deconvolve_cic(field, N):
    ''' Deconvolve CIC kernel from field
    N - grid size in cells '''
    from seren2.cosmology import _power_spectrum
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
