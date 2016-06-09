import numpy as np
try:
    import scipy
    import scipy.interpolate
except ImportError:
    pass
import math
from pynbody.analysis.hmf import PowerSpectrumCAMB, TophatFilter, variance
from pynbody.analysis import cosmology
from pynbody.array import SimArray
from helper_functions import print_msg, get_eval
import warnings
from enum import Enum

import scipy.interpolate as si
import scipy.integrate
import scipy.fftpack as fft
import cosmolopy as cp
import matplotlib.pylab as plt


def split_music_powerspec(filename):
    '''
    Write separate power spectra for use with PowerSpectrum class
    '''
    data = np.loadtxt(filename, unpack=True)
    k = data[0]
    for i, fname in enumerate(['input_powerspec_cdm.txt', 'input_powerspec_vcdm.txt',
                               'input_powerspec_baryon.txt', 'input_powerspec_vbaryon.txt', 'input_powerspec_total.txt']):
        np.savetxt(fname, np.array([k, data[i + 1]]).T)


class PowerSpectrum(PowerSpectrumCAMB):

    '''
    Class to wrap Pynbody PowerSpectrum and add support for MUSIC spectra

    Parameters:
            * context (pynbody.snapshot): the snapshot containing desired cosmological parameters
            * pstype = CAMB (string): Method used to produce power spectra (CAMB or MUSIC)
            * species = None (string): baryons, cdm or None (total)
            * kwargs (dict): arguments to be passed to pynbody.analysis.hmf.PowerSpectrumCAMB
    '''

    def __init__(self, context, filename, pstype='CAMB', species=None, **kwargs):
        super(PowerSpectrum, self).__init__(
            context, filename=filename, **kwargs)
        self._context = context
        self._type = pstype.lower()

    def at_z(self, z):
        '''
        Compute linear evolution of power spectrum for all k
        '''

        from cosmo import D_z
        origz = self._context['z']
        omegam0 = self._properties['omegaM0']
        omegal0 = self._properties['omegaL0']
        lingrowth_z = D_z(z, omegam0, omegal0)
        lingrowth_origz = D_z(origz, omegam0, omegal0)
        return self.Pk * (lingrowth_z ** 2 / lingrowth_origz ** 2)

    def dimensionless(self):
        return self.Pk * ((self.k ** 3) / (2 * np.pi ** 2))

    def swap_fourier_convention(self):
        '''
        Swap the Fourier convention from CAMB to MUSIC or vise versa
        '''
        if self._convention_changed is False:
            if self._type == 'camb':
                self.Pk /= (2 * np.pi) ** 3
            elif self._type == 'music':
                self.Pk *= (2 * np.pi) ** 3
            else:
                raise RuntimeError("Unknown PS type: %s" % self._type)
            self._convention_changed = True
        else:
            # Change back
            if self._type == 'camb':
                self.Pk *= (2 * np.pi) ** 3
            elif self._type == 'music':
                self.Pk /= (2 * np.pi) ** 3
            else:
                raise RuntimeError("Unknown PS type: %s" % self._type)
            self._convention_changed = False

# Define default cosmology
DEFAULT_COSMOLOGY = {
    'omega_M_0': 0.27,
    'omega_lambda_0': 0.73,
    'omega_b_0': 0.045,
    'omega_n_0': 0.0,
    'N_nu': 0,
    'h': 0.7,
    'n': 0.96,
    'sigma_8': 0.8,
    'baryonic_effects': True,
    'z': 0.
}


class Realisation(object):

    def __init__(self, cosmo=DEFAULT_COSMOLOGY, scale=0.2, nsamp=128, filename=None, cols=None):
        """Initialise a 3D box with matter distribution described by a given power spectrum.
        Parameters:
            scale (float): Boxsize (Mpc)
            nsamp (int): Number of cells per dimension
        """

        self.cosmo = cosmo

        # Define grid coordinates alone one dimension
        self.x = np.linspace(-scale, scale, nsamp)  # in Mpc
        self.N = self.x.size  # Grid points
        self.L = self.x[-1] - self.x[0]  # Linear size of box

        # Conversion factor for FFT of power spectrum
        # For an example, see in liteMap.py:fillWithGaussianRandomField() in
        # Flipper, by Sudeep Das. http://www.astro.princeton.edu/~act/flipper
        self.boxfactor = (self.N ** 6.) / self.L ** 3.

        # Fourier mode array
        self.set_fft_sample_spacing()  # 3D array, arranged in correct order

        # Min./max. k modes in 3D (excl. zero mode)
        self.kmin = 2. * np.pi / self.L
        self.kmax = 2. * np.pi * np.sqrt(3.) * self.N / self.L

        # Creata a 3D realisation of density/velocity perturbations in the box
        self.realise_density(filename, cols)
        # self.realise_velocity()

    def set_fft_sample_spacing(self):
        """Calculate the sample spacing in Fourier space, given some symmetric 3D 
        box in real space, with 1D grid point coordinates 'x'."""
        self.kx = np.zeros((self.N, self.N, self.N))
        self.ky = np.zeros((self.N, self.N, self.N))
        self.kz = np.zeros((self.N, self.N, self.N))
        NN = (self.N * fft.fftfreq(self.N, 1.)).astype("i")
        for i in NN:
            self.kx[i, :, :] = i
            self.ky[:, i, :] = i
            self.kz[:, :, i] = i
        fac = (2. * np.pi / self.L)
        self.k = np.sqrt(self.kx ** 2. + self.ky ** 2. + self.kz ** 2.) * fac

    def realise_density(self, filename=None, cols=None):
        """Create realisation of the power spectrum by randomly sampling
        from Gaussian distributions of variance P(k) for each k mode."""

        k, pk = (None, None)
        if filename:
            assert(cols is not None)
            tmp = np.loadtxt(filename, unpack=True)
            k = tmp[cols[0]]
            pk = tmp[cols[1]]

            def log_interp1d(xx, yy, kind='linear'):
                logx = np.log10(xx)
                logy = np.log10(yy)
                lin_interp = si.InterpolatedUnivariateSpline(logx, logy)
                log_interp = lambda zz: np.power(
                    10.0, lin_interp(np.log10(zz)))
                return log_interp
            #pk = tmp[cols[1]]
            # Interp pk values to the flattened 3D self.k array
            f = log_interp1d(k, pk)

            k = self.k.flatten()
            pk = f(k)
            pk = np.reshape(pk, np.shape(self.k))
            pk = np.nan_to_num(pk)  # Remove NaN at k=0 (and any others...)
        else:
            k = self.k.flatten()
            pk = cp.perturbation.power_spectrum(k, **self.cosmo)
            pk = np.reshape(pk, np.shape(self.k))
            pk = np.nan_to_num(pk)  # Remove NaN at k=0 (and any others...)

        # Normalise the power spectrum properly (factor of volume, and norm.
        # factor of 3D DFT)
        pk *= self.boxfactor

        # Generate Gaussian random field with given power spectrum
        #re = np.random.normal(0., 1., np.shape(self.k))
        # im = np.random.normal(0., 1., np.shape(self.k)
        import random
        random.seed(1234)
        re = np.array([random.normalvariate(0., 1.)
                       for i in range(len(self.k.flatten()))]).reshape(self.k.shape)
        im = np.array([random.normalvariate(0., 1.)
                       for i in range(len(self.k.flatten()))]).reshape(self.k.shape)
        self.delta_k = (re + 1j * im) * np.sqrt(pk)

        # Transform to real space. Here, we are discarding the imaginary part
        # of the inverse FT! But we can recover the correct (statistical)
        # result by multiplying by a factor of sqrt(2). Also, there is a factor
        # of N^3 which seems to appear by a convention in the Discrete FT.
        self.delta_x = fft.ifftn(self.delta_k).real

        # Finally, get the Fourier transform on the real field back
        self.delta_k = fft.fftn(self.delta_x)

    def realise_velocity(self):
        """Realise the (unscaled) velocity field in Fourier space. See 
           Dodelson Eq. 9.18 for an expression; we factor out the 
           time-dependent quantities here. They can be added at a later stage."""

        # If the FFT has an even number of samples, the most negative frequency
        # mode must have the same value as the most positive frequency mode.
        # However, when multiplying by 'i', allowing this mode to have a
        # non-zero real part makes it impossible to satisfy the reality
        # conditions. As such, we can set the whole mode to be zero, make sure
        # that it's pure imaginary, or use an odd number of samples. Different
        # ways of dealing with this could change the answer!
        if self.N % 2 == 0:  # Even no. samples
            # Set highest (negative) freq. to zero
            mx = np.where(self.kx == np.min(self.kx))
            my = np.where(self.ky == np.min(self.ky))
            mz = np.where(self.kz == np.min(self.kz))
            self.kx[mx] = 0.0
            self.ky[my] = 0.0
            self.kz[mz] = 0.0

        # Get squared k-vector in k-space (and factor in scaling from kx, ky,
        # kz)
        k2 = self.k ** 2.

        # Calculate components of A (the unscaled velocity)
        Ax = 1j * self.delta_k * self.kx * (2. * np.pi / self.L) / k2
        Ay = 1j * self.delta_k * self.ky * (2. * np.pi / self.L) / k2
        Az = 1j * self.delta_k * self.kz * (2. * np.pi / self.L) / k2
        Ax = np.nan_to_num(Ax)
        Ay = np.nan_to_num(Ay)
        Az = np.nan_to_num(Az)
        return fft.ifftn(Ax).real, fft.ifftn(Ay).real, fft.ifftn(Az).real

    def apply_bias(self, k_b, b):
        ''' Apply a bias to the realisations power spectrum, and recompute the 3D field.
        Parameters:
            b (array): bias to deconvolve with the delta_x field, such that:
            delta_x = ifft(delta_k/b)
        '''
        # Interpolate/extrapolate the bias to the 3D grid
        def log_interp1d(xx, yy, kind='linear'):
            logx = np.log10(xx)
            logy = np.log10(yy)
            lin_interp = si.InterpolatedUnivariateSpline(logx, logy)
            log_interp = lambda zz: np.power(
                10.0, lin_interp(np.log10(zz)))
            return log_interp
        #f = log_interp1d(k_b, b)
        f = si.InterpolatedUnivariateSpline(k_b, b)
        k = self.k.flatten()
        b = f(k).reshape(self.k.shape)
        #b = np.zeros(len(k))
        #b += 1

        # Apply the bias. self.delta_k has been recomputed from the realisation
        # in realise_density
        #pk = (self.delta_k * np.conj(self.delta_k)).flatten()
        #self.delta_k = np.sqrt(pk*b).reshape(self.k.shape)
        self.delta_k *= np.sqrt(b)

        # Inverse FFT to compute the realisation
        self.delta_x = fft.ifftn(self.delta_k).real

    ##########################################################################
    # Output quantities related to the realisation
    ##########################################################################

    def window(self, k, R):
        """Fourier transform of tophat window function, used to calculate 
           sigmaR etc. See "Cosmology", S. Weinberg, Eq.8.1.46."""
        x = k * R
        f = (3. / x ** 3.) * (np.sin(x) - x * np.cos(x))
        return f ** 2.

    def window1(self, k, R):
        """Fourier transform of tophat window function, used to calculate 
           sigmaR etc. See "Cosmology", S. Weinberg, Eq.8.1.46."""
        x = k * R
        f = (3. / x ** 3.) * (np.sin(x) - x * np.cos(x))
        return f

    def sigmaR(self, R):
        """Get variance of matter perturbations, smoothed with a tophat filter 
           of radius R h^-1 Mpc."""

        # Need binned power spectrum, with k flat, monotonic for integration.
        k, pk, stddev = self.binned_power_spectrum()

        # Only use non-NaN values
        good_idxs = -np.isnan(pk)
        pk = pk[good_idxs]
        k = k[good_idxs]

        # Discretely-sampled integrand, y
        y = k ** 2. * pk * self.window(k, R / self.cosmo['h'])
        I = scipy.integrate.simps(y, k)

        # Return sigma_R (note factor of 4pi / (2pi)^3 from integration)
        return np.sqrt(I / (2. * np.pi ** 2.))

    def sigma8(self):
        """Get variance of matter perturbations on smoothing scale of 
           8 h^-1 Mpc."""
        return self.sigmaR(8.0)

    def binned_power_spectrum(self, nbins=20):
        """Return a binned power spectrum, calculated from the realisation."""

        pk = self.delta_k * np.conj(self.delta_k)  # Power spectrum (noisy)
        pk = pk.real / self.boxfactor

        # Bin edges/centroids. Logarithmically-distributed bins in k-space.
        # FIXME: Needs to be checked for correctness
        bins = np.logspace(np.log10(self.kmin), np.log10(self.kmax), nbins)
        _bins = [0.0] + list(bins)  # Add zero to the beginning
        cent = [0.5 * (_bins[j + 1] + _bins[j]) for j in range(bins.size)]

        # Initialise arrays
        vals = np.zeros(bins.size)
        stddev = np.zeros(bins.size)

        # Identify which bin each element of 'pk' should go into
        idxs = np.digitize(self.k.flatten(), bins)

        # For each bin, get the average pk value in that bin
        for i in range(bins.size):
            ii = np.where(idxs == i, True, False)
            vals[i] = np.mean(pk.flatten()[ii])
            stddev[i] = np.std(pk.flatten()[ii])

        # First value is garbage, so throw it away
        # k ps std
        return np.array(cent[1:]), np.array(vals[1:]), np.array(stddev[1:])

    def theoretical_power_spectrum(self):
        """Calculate the theoretical power spectrum for the given cosmological 
           parameters, using Cosmolopy. Does not depend on the realisation."""
        k = np.logspace(-3.5, 1., 1e3)
        pk = cp.perturbation.power_spectrum(k, **self.cosmo)
        return k, pk

    ##########################################################################
    # Tests for consistency and accuracy
    ##########################################################################

    def test_sampling_error(self):
        """P(k) is sampled within some finite window in the interval 
           [kmin, kmax], where kmin=2pi/L and kmax=2pi*sqrt(3)*(N/2)*(1/L) 
           (for 3D FT). The lack of sampling in some regions of k-space means 
           that sigma8 can't be perfectly reconstructed (see U-L. Pen, 
           arXiv:astro-ph/9709261 for a discussion).
           This function calculates sigma8 from the realised box, and compares 
           this with the theoretical calculation for sigma8 over a large 
           k-window, and over a k-window of the same size as for the box.
        """

        # Calc. sigma8 from the realisation
        s8_real = self.sigma8()

        # Calc. theoretical sigma8 in same k-window as realisation
        _k = np.linspace(self.kmin, self.kmax, 5e3)
        _pk = cp.perturbation.power_spectrum(_k, **self.cosmo)
        _y = _k ** 2. * _pk * self.window(_k, 8.0 / self.cosmo['h'])
        _y = np.nan_to_num(_y)
        s8_th_win = np.sqrt(scipy.integrate.simps(_y, _k) / (2. * np.pi ** 2.))

        # Calc. full sigma8 (in window that is wide enough)
        _k2 = np.logspace(-5, 2, 5e4)
        _pk2 = cp.perturbation.power_spectrum(_k2, **self.cosmo)
        _y2 = _k2 ** 2. * _pk2 * self.window(_k2, 8.0 / self.cosmo['h'])
        _y2 = np.nan_to_num(_y2)
        s8_th_full = np.sqrt(
            scipy.integrate.simps(_y2, _k2) / (2. * np.pi ** 2.))

        # Calculate sigma8 in real space
        dk = np.reshape(self.delta_k, np.shape(self.k))
        dk = dk * self.window1(self.k, 8.0 / self.cosmo['h'])
        dk = np.nan_to_num(dk)
        dx = fft.ifftn(dk)
        s8_realspace = np.std(dx)

        # sigma20
        dk = np.reshape(self.delta_k, np.shape(self.k))
        dk = dk * self.window1(self.k, 20.0 / self.cosmo['h'])
        dk = np.nan_to_num(dk)
        dx = fft.ifftn(dk)
        s20_realspace = np.std(dx)

        s20_real = self.sigmaR(20.)

        # Print report
        print ""
        print "sigma8 (real.): \t", s8_real
        print "sigma8 (th.win.):\t", s8_th_win
        print "sigma8 (th.full):\t", s8_th_full
        print "sigma8 (realsp.):\t", s8_realspace
        print "ratio =", 1. / (s8_real / s8_realspace)
        print ""
        print "sigma20 (real.): \t", s20_real
        print "sigma20 (realsp.):\t", s20_realspace
        print "ratio =", 1. / (s20_real / s20_realspace)
        print "var(delta) =", np.std(self.delta_x)

    def test_parseval(self):
        """Ensure that Parseval's theorem is satisfied for delta_x and delta_k, 
           i.e. <delta_x^2> = Sum_k[P(k)]. Important consistency check for FT;
           should return unity if everything is OK.
        """
        s1 = np.sum(self.delta_x ** 2.) * \
            self.N ** 3.  # Factor of N^3 missing due to averaging
        s2 = np.sum(self.delta_k * np.conj(self.delta_k)).real
        print "Parseval test:", s1 / s2, "(should be 1.0)"

    def test(self, filename=None, cols=None):
        box = self
        re_k, re_pk, re_stddev = box.binned_power_spectrum()
        th_k, th_pk = (None, None)
        if filename:
            tmp = np.loadtxt(filename, unpack=True)
            th_k = tmp[cols[0]]
            th_pk = tmp[cols[1]]
        else:
            th_k, th_pk = box.theoretical_power_spectrum()

        # Tests
        # box.test_sampling_error()
        box.test_parseval()

        # Plot some stuff
        # plt.subplot(211)
        plt.plot(th_k, th_pk, 'b-')
        #plt.errorbar(re_k, re_pk, yerr=re_stddev, fmt=".", color='r')
        plt.plot(re_k, re_pk, 'r.')
        plt.xscale('log')
        plt.yscale('log')

        # plt.subplot(212)

        def dx(R):
            dk = np.reshape(box.delta_k, np.shape(box.k))
            dk = dk * box.window1(box.k, R / box.cosmo['h'])
            dk = np.nan_to_num(dk)
            dx = fft.ifftn(dk)
            return dx

        dx2 = dx(8.0)
        dx3 = dx(100.0)

        #plt.hist(dx2.flatten(), bins=100, alpha=0.2)
        #plt.hist(dx3.flatten(), bins=100, alpha=0.2)

        # plt.show()

############################################### Analysis #################


def power_spectrum_nd(input_array, box_dims):
    '''
    Calculate the power spectrum of input_array and return it as an n-dimensional array,
    where n is the number of dimensions in input_array
    box_side is the size of the box in comoving Mpc. If this is set to None (default),
    the internal box size is used

    Parameters:
            * input_array (numpy array): the array to calculate the
                    power spectrum of. Can be of any dimensions.
            * box_dims = None (float or array-like): the dimensions of the
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
                    [boxsize]*ndim

    Returns:
            The power spectrum in the same dimensions as the input array.
    '''

    print_msg('Calculating power spectrum...')
    ft = fft.fftshift(fft.fftn(input_array.astype('float64')))
    power_spectrum = np.abs(ft) ** 2
    print_msg('...done')

    # scale
    boxvol = np.product(map(float, box_dims))
    pixelsize = boxvol / (np.product(input_array.shape))
    power_spectrum *= pixelsize ** 2 / boxvol

    return power_spectrum


def cross_power_spectrum_nd(input_array1, input_array2, box_dims):
    ''' 
    Calculate the cross power spectrum two arrays and return it as an n-dimensional array,
    where n is the number of dimensions in input_array
    box_side is the size of the box in comoving Mpc. If this is set to None (default),
    the internal box size is used

    Parameters:
            * input_array1 (numpy array): the first array to calculate the 
                    power spectrum of. Can be of any dimensions.
            * input_array2 (numpy array): the second array. Must have same 
                    dimensions as input_array1.
            * box_dims = None (float or array-like): the dimensions of the 
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.

    Returns:
            The cross power spectrum in the same dimensions as the input arrays.

    TODO:
            Also return k values.
    '''

    assert(input_array1.shape == input_array2.shape)

    print_msg('Calculating power spectrum...')
    ft1 = fft.fftshift(fft.fftn(input_array1.astype('float64')))
    ft2 = fft.fftshift(fft.fftn(input_array2.astype('float64')))
    power_spectrum = np.real(ft1) * np.real(ft2) + np.imag(ft1) * np.imag(ft2)
    print_msg('...done')

    # scale
    #boxvol = float(box_side)**len(input_array1.shape)
    boxvol = np.product(map(float, box_dims))
    pixelsize = boxvol / (np.product(map(float, input_array1.shape)))
    power_spectrum *= pixelsize ** 2 / boxvol

    return power_spectrum


def radial_average_std(input_array, box_dims, kbins=10):
    '''
    Radially average data. Mostly for internal use.

    Parameters: 
            * input_array (numpy array): the data array
            * box_dims = None (float or array-like): the dimensions of the 
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * kbins = 10 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.

    Returns:
            A tuple with (data, bins, n_modes), where data is an array with the 
            averaged data, bins is an array with the bin centers and n_modes is the 
            number of modes in each bin

    '''

    k_comp, k = _get_k(input_array, box_dims)

    kbins = _get_kbins(kbins, box_dims, k)

    # Bin the data
    print_msg('Binning data...')
    dk = (kbins[1:] - kbins[:-1]) / 2.

    x = k.flatten()
    y = input_array.flatten()
    n, _ = np.histogram(x, bins=kbins)
    sy, _ = np.histogram(x, bins=kbins, weights=y)
    sy2, _ = np.histogram(x, bins=kbins, weights=y * y)
    mean = sy / n
    stderr = np.sqrt(sy2 / n - mean * mean) / np.sqrt(n)

    return mean, kbins[:-1] + dk, stderr


def radial_average(input_array, box_dims, kbins=10):
    '''
    Radially average data. Mostly for internal use.

    Parameters: 
            * input_array (numpy array): the data array
            * box_dims = None (float or array-like): the dimensions of the 
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * kbins = 10 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.

    Returns:
            A tuple with (data, bins, n_modes), where data is an array with the 
            averaged data, bins is an array with the bin centers and n_modes is the 
            number of modes in each bin

    '''

    k_comp, k = _get_k(input_array, box_dims)

    kbins = _get_kbins(kbins, box_dims, k)

    # Bin the data
    print_msg('Binning data...')
    dk = (kbins[1:] - kbins[:-1]) / 2.
    # Total power in each bin
    outdata = np.histogram(k.flatten(), bins=kbins,
                           weights=input_array.flatten())[0]
    # Number of modes in each bin
    n_modes = np.histogram(k.flatten(), bins=kbins)[0].astype('float')
    outdata /= n_modes

    return outdata, kbins[:-1] + dk, n_modes


def power_spectrum_1d_cic(input_array_nd, box_dims, kbins=100, return_n_modes=False, return_kernel=False):
    ''' Calculate the spherically averaged power spectrum of an array
    and return it as a one-dimensional array. Performs a deconvolution with
    CIC kernel

    Parameters:
            * input_array_nd (numpy array): the data array
            * kbins = 100 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            * box_dims = None (float or array-like): the dimensions of the
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * return_n_modes = False (bool): if true, also return the
                    number of modes in each bin

    Returns:
            A tuple with (Pk, bins), where Pk is an array with the
            power spectrum and bins is an array with the k bin centers.
    '''

    def _radial_average(input_array, k, box_dims, kbins=10):
        '''
        Radially average data. Mostly for internal use.

        Parameters: 
                * input_array (numpy array): the data array
                * box_dims = None (float or array-like): the dimensions of the 
                        box. If this is None, the current box volume is used along all
                        dimensions. If it is a float, this is taken as the box length
                        along all dimensions. If it is an array-like, the elements are
                        taken as the box length along each axis.
                * kbins = 10 (integer or array-like): The number of bins,
                        or a list containing the bin edges. If an integer is given, the bins
                        are logarithmically spaced.

        Returns:
                A tuple with (data, bins, n_modes), where data is an array with the 
                averaged data, bins is an array with the bin centers and n_modes is the 
                number of modes in each bin

        '''

        kbins = _get_kbins(kbins, box_dims, k)

        # Bin the data
        print_msg('Binning data...')
        dk = (kbins[1:] - kbins[:-1]) / 2.
        # Total power in each bin
        outdata = np.histogram(k.flatten(), bins=kbins,
                               weights=input_array.flatten())[0]
        # Number of modes in each bin
        n_modes = np.histogram(k.flatten(), bins=kbins)[0].astype('float')
        outdata /= n_modes

        return outdata, kbins[:-1] + dk, n_modes

    import _power_spectrum
    N = input_array_nd.shape[0]

    print_msg("Computing CIC kernel...")
    W = _power_spectrum.cic_window_function(N)
    print_msg("Done")

    boxsize = box_dims[0]
    boxvol = np.product(map(float, box_dims))
    pixelsize = boxvol / (np.product(input_array_nd.shape))

    print_msg("Sampling fourier modes...")
    kk = _power_spectrum.fft_sample_spacing(N, boxsize)
    print_msg("Done")

    # Compute power spectrum
    print_msg("Computing power spectrum...")
    input_array = fft.fftn(input_array_nd)
    input_array = np.abs(input_array) ** 2
    input_array /= W  # deconvolve with kernel

    # Account for grid spacing and boxsize
    input_array *= pixelsize ** 2 / boxvol
    print_msg("Done")

    # Spherically average
    ps, kbins, nmodes = _radial_average(input_array, kk, box_dims, kbins=kbins)

    # Return
    if return_n_modes:
        if return_kernel:
            return ps, kbins, nmodes, W
        return ps, kbins, nmodes
    if return_kernel:
        return ps, kbins, W
    return ps, kbins


def power_spectrum_1d_std(input_array_nd, box_dims, kbins=100, return_stderr=False):
    ''' Calculate the spherically averaged power spectrum of an array
    and return it as a one-dimensional array.

    Parameters:
            * input_array_nd (numpy array): the data array
            * kbins = 100 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            * box_dims = None (float or array-like): the dimensions of the
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * return_n_modes = False (bool): if true, also return the
                    number of modes in each bin

    Returns:
            A tuple with (Pk, bins), where Pk is an array with the
            power spectrum and bins is an array with the k bin centers.
    '''

    input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims)

    ps, bins, stderr = radial_average_std(
        input_array, kbins=kbins, box_dims=box_dims)
    if return_stderr:
        return ps, bins, stderr
    return ps, bins


def power_spectrum_1d(input_array_nd, box_dims, kbins=100, return_n_modes=False):
    ''' Calculate the spherically averaged power spectrum of an array
    and return it as a one-dimensional array.

    Parameters:
            * input_array_nd (numpy array): the data array
            * kbins = 100 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            * box_dims = None (float or array-like): the dimensions of the
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * return_n_modes = False (bool): if true, also return the
                    number of modes in each bin

    Returns:
            A tuple with (Pk, bins), where Pk is an array with the
            power spectrum and bins is an array with the k bin centers.
    '''

    input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims)

    ps, bins, n_modes = radial_average(
        input_array, kbins=kbins, box_dims=box_dims)
    if return_n_modes:
        return ps, bins, n_modes
    return ps, bins


def cross_power_spectrum_1d(input_array1_nd, input_array2_nd, box_dims, kbins=100, return_n_modes=False):
    ''' Calculate the spherically averaged cross power spectrum of two arrays 
    and return it as a one-dimensional array.

    Parameters: 
            * input_array1_nd (numpy array): the first data array
            * input_array2_nd (numpy array): the second data array
            * kbins = 100 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            * box_dims = None (float or array-like): the dimensions of the 
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * return_n_modes = False (bool): if true, also return the
                    number of modes in each bin

    Returns: 
            A tuple with (Pk, bins), where Pk is an array with the 
            cross power spectrum and bins is an array with the k bin centers.
    '''

    input_array = cross_power_spectrum_nd(
        input_array1_nd, input_array2_nd, box_dims=box_dims)

    ps, bins, n_modes = radial_average(
        input_array, kbins=kbins, box_dims=box_dims)
    if return_n_modes:
        return ps, bins, n_modes
    return ps, bins


def power_spectrum_mu(input_array, box_dims, los_axis=0, mubins=20, kbins=10, weights=None,
                      exclude_zero_modes=True):
    '''
    Calculate the power spectrum and bin it in mu=cos(theta) and k
    input_array is the array to calculate the power spectrum from

    Parameters: 
            * input_array (numpy array): the data array
            * los_axis = 0 (integer): the line-of-sight axis
            * mubins = 20 (integer): the number of mu bins
            * kbins = 10 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            * box_dims = None (float or array-like): the dimensions of the 
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * exlude_zero_modes = True (bool): if true, modes with any components
                    of k equal to zero will be excluded.

    Returns: 
            A tuple with (Pk, mubins, kbins), where Pk is an array with the 
            power spectrum of dimensions (n_mubins x n_kbins), 
            mubins is an array with the mu bin centers and
            kbins is an array with the k bin centers.

    '''

    # Calculate the power spectrum
    powerspectrum = power_spectrum_nd(input_array, box_dims=box_dims)

    return mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes)


def cross_power_spectrum_mu(input_array1, input_array2, box_dims, los_axis=0, mubins=20, kbins=10,
                            weights=None, exclude_zero_modes=True):
    '''
    Calculate the cross power spectrum and bin it in mu=cos(theta) and k
    input_array is the array to calculate the power spectrum from

    Parameters: 
            * input_array1 (numpy array): the first data array
            * input_array2 (numpy array): the second data array
            * los_axis = 0 (integer): the line-of-sight axis
            * mubins = 20 (integer): the number of mu bins
            * kbins = 10 (integer or array-like): The number of bins,
                    or a list containing the bin edges. If an integer is given, the bins
                    are logarithmically spaced.
            * box_dims = None (float or array-like): the dimensions of the 
                    box. If this is None, the current box volume is used along all
                    dimensions. If it is a float, this is taken as the box length
                    along all dimensions. If it is an array-like, the elements are
                    taken as the box length along each axis.
            * exlude_zero_modes = True (bool): if true, modes with any components
                    of k equal to zero will be excluded.

    Returns: 
            A tuple with (Pk, mubins, kbins), where Pk is an array with the 
            cross power spectrum of dimensions (n_mubins x n_kbins), 
            mubins is an array with the mu bin centers and
            kbins is an array with the k bin centers.

    TODO:
            Add support for (non-numpy) lists for the bins
    '''

    # Calculate the power spectrum
    powerspectrum = cross_power_spectrum_nd(
        input_array1, input_array2, box_dims=box_dims)

    return mu_binning(powerspectrum, los_axis, mubins, kbins, box_dims, weights, exclude_zero_modes)


def mu_binning(powerspectrum, box_dims, los_axis=0, mubins=20, kbins=10, weights=None,
               exclude_zero_modes=True):
    '''
    This function is for internal use only.
    '''

    if weights != None:
        powerspectrum *= weights

    assert(len(powerspectrum.shape) == 3)

    k_comp, k = _get_k(powerspectrum, box_dims)

    mu = _get_mu(k_comp, k, los_axis)

    # Calculate k values, and make k bins
    kbins = _get_kbins(kbins, box_dims, k)
    dk = (kbins[1:] - kbins[:-1]) / 2.
    n_kbins = len(kbins) - 1

    # Exclude k_perp = 0 modes
    if exclude_zero_modes:
        good_idx = _get_nonzero_idx(powerspectrum.shape, los_axis)
    else:
        good_idx = np.ones_like(powerspectrum)

    # Make mu bins
    if isinstance(mubins, int):
        mubins = np.linspace(-1., 1., mubins + 1)
    dmu = (mubins[1:] - mubins[:-1]) / 2.
    n_mubins = len(mubins) - 1

    # Remove the zero component from the power spectrum. mu is undefined here
    powerspectrum[tuple(np.array(powerspectrum.shape) / 2)] = 0.

    # Bin the data
    print_msg('Binning data...')
    outdata = np.zeros((n_mubins, n_kbins))
    for ki in range(n_kbins):
        print_msg('Bin %d of %d' % (ki, n_kbins))
        kmin = kbins[ki]
        kmax = kbins[ki + 1]
        kidx = get_eval()('(k >= kmin) & (k < kmax)')
        kidx *= good_idx
        for i in range(n_mubins):
            mu_min = mubins[i]
            mu_max = mubins[i + 1]
            idx = get_eval()('(mu >= mu_min) & (mu < mu_max) & kidx')
            outdata[i, ki] = np.mean(powerspectrum[idx])

            if weights != None:
                outdata[i, ki] /= weights[idx].mean()

    return outdata, mubins[:-1] + dmu, kbins[:-1] + dk


# Some methods for internal use

def _get_k(input_array, box_dims):
    '''
    Get the k values for input array with given dimensions.
    Return k components and magnitudes.
    For internal use.
    '''
    dim = len(input_array.shape)
    if dim == 1:
        x = np.arange(len(input_array))
        center = x.max() / 2.
        kx = 2. * np.pi * (x - center) / box_dims[0]
        return [kx], kx
    elif dim == 2:
        x, y = np.indices(input_array.shape, dtype='int32')
        center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])
        kx = 2. * np.pi * (x - center[0]) / box_dims[0]
        ky = 2. * np.pi * (y - center[1]) / box_dims[1]
        k = np.sqrt(kx ** 2 + ky ** 2)
        return [kx, ky], k
    elif dim == 3:
        x, y, z = np.indices(input_array.shape, dtype='int32')
        center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2,
                           (z.max() - z.min()) / 2])
        kx = 2. * np.pi * (x - center[0]) / box_dims[0]
        ky = 2. * np.pi * (y - center[1]) / box_dims[1]
        kz = 2. * np.pi * (z - center[2]) / box_dims[2]

        k = get_eval()('(kx**2 + ky**2 + kz**2 )**(1./2.)')
        return [kx, ky, kz], k


def _get_mu(k_comp, k, los_axis):
    '''
    Get the mu values for given k values and 
    a line-of-sight axis.
    For internal use
    '''

    # Line-of-sight distance from center
    if los_axis == 0:
        los_dist = k_comp[0]
    elif los_axis == 1:
        los_dist = k_comp[1]
    elif los_axis == 2:
        los_dist = k_comp[2]
    else:
        raise Exception('Your space is not %d-dimensional!' % los_axis)

    # mu=cos(theta) = k_par/k
    mu = los_dist / np.abs(k)
    mu[np.where(k < 0.001)] = np.nan

    return mu


def _get_kbins(kbins, box_dims, k):
    '''
    Make a list of bin edges if kbins is an integer,
    otherwise return it as it is.
    '''
    if isinstance(kbins, int):
        kmin = 2. * np.pi / min(box_dims)
        kbins = 10 ** np.linspace(np.log10(kmin), np.log10(k.max()), kbins + 1)
    return kbins


def _get_nonzero_idx(ps_shape, los_axis):
    '''
    Get the indices where k_perp != 0
    '''
    x, y, z = np.indices(ps_shape)
    if los_axis == 0:
        zero_idx = (y == ps_shape[1] / 2) * (z == ps_shape[2] / 2)
    elif los_axis == 1:
        zero_idx = (x == ps_shape[0] / 2) * (z == ps_shape[2] / 2)
    else:
        zero_idx = (x == ps_shape[0] / 2) * (y == ps_shape[1] / 2)
    good_idx = np.invert(zero_idx)
    return good_idx
