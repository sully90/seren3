#!/usr/bin/python
"""Generate a Gaussian random field with some power spectrum, following an 
   algorithm that Pedro wrote down."""

import numpy as np
import scipy.integrate
import scipy.fftpack as fft
import cosmolopy as cp

# Define default cosmology
DEFAULT_COSMOLOGY = {
		 'omega_M_0' : 0.27,
		 'omega_lambda_0' : 0.73,
		 'omega_b_0' : 0.045,
		 'omega_n_0' : 0.0,
		 'N_nu' : 0,
		 'h' : 0.7,
		 'n' : 0.96,
		 'sigma_8' : 0.79,
		 'baryonic_effects': True
		}

def fft_sample_spacing(x):
	"""Calculate the sample spacing in Fourier space, given some symmetric 3D 
	   box in real space, with 1D grid point coordinates 'x'."""
	N = x.size # Dimension of grid
	fac = 2.*np.pi/(x[1]-x[0]) # Overall factor (Mpc^-1)
	
	# Need slightly special FFT freq. numbering order, given by fftfreq()
	NN = ( N*fft.fftfreq(N, 1.) ).astype("i")
	
	# Get sample spacing. In 1D, k_n = 2pi*n / (N*dx), where N is length of 1D array
	# In 3D, k_n1,n2,n3 = [2pi / (N*dx)] * sqrt(n1^2 + n2^2 + n3^2)
	kk = [[[np.sqrt(i**2. + j**2. + k**2.) for i in NN] for j in NN] for k in NN]
	kk = fac * np.array(kk)
	return kk



def realise_unstructured(scale=2e3, nsamp=32):
	"""Create realisation of an unstrcutured Gaussian random field for some 
	   regular grid of side NSAMP."""
	# Generate 3D Gaussian random field in real space
	x = np.linspace(-scale, scale, nsamp) # 3D grid coords, in Mpc
	delta_xi = np.random.randn(x.size, x.size, x.size)
	return x, delta_xi


def realise(cosmo=DEFAULT_COSMOLOGY, scale=2e3, nsamp=32):
	"""Create realisation of the matter power spectrum for some regular grid of 
	   side NSAMP, with cosmology 'cosmo'."""
	
	# Generate 3D Gaussian random field in real space
	x = np.linspace(-scale, scale, nsamp) # 3D grid coords, in Mpc
	delta_xi = np.random.randn(x.size, x.size, x.size)

	# FT the field, respecting reality conditions on FFT (Ch12, Num. Rec.)
	delta_ki = fft.fftn(delta_xi) # 3D discrete FFT
	kk = fft_sample_spacing(x) # Get grid coordinates in k-space
	
	# Calculate matter power spectrum
	k = kk.flatten() # FIXME: Lots of duplicate values. Can be more efficient.
	ps = cp.perturbation.power_spectrum(k, z=0.0, **cosmo)
	norm = cp.perturbation.norm_power(**cosmo)
	ps = np.reshape(ps, np.shape(kk))
	ps = np.nan_to_num(ps) # Remove NaN at k=0 (and any others...)

	# Multiply by sqrt of power spectrum and transform back to real space
	# FIXME: 'norm' should be here, or not?
	delta_k = np.sqrt(ps) * delta_ki
	delta_x = np.real(  fft.ifftn(delta_k)  ) # Discard small imag. values
	
	return x, kk, delta_x, delta_k, ps, norm



def realisation_ps(x, delta_x, bins=40):
	"""Get binned power spectrum of realisation."""
	
	# Get power spectrum
	k = fft_sample_spacing(x).flatten()
	delta_k = fft.fftn(delta_x)
	ps = (np.abs(delta_k).flatten()**2.)

	# Get bin masks
	edges = np.linspace(0.0, np.max(k), bins)
	dk = edges[1] - edges[0]
	centroids = [0.5 * (edges[i+1] + edges[i]) for i in range(edges.size - 1)]
	m = [np.where((k >= edges[i]) & (k < edges[i+1]), True, False) for i in range(edges.size - 1)]

	# Get averaged value in each bin
	vals = np.array(  [np.sum(ps[mm]) / np.sum(mm) for mm in m]  )
	return np.array(centroids), vals


def window(k, R):
	"""Window function, which filters the FT. Used in sigma8 calculation."""
	# See Weinberg, Eq.8.1.46
	x = k*R
	f = (3. / x**3.) * ( np.sin(x) - x*np.cos(x) )
	return f**2.


def realisation_sigma8(k, pk):
	"""Calculate sigma8 for a realisation of the power spectrum."""
	R = 8. / 0.7 # 8 h^-1 Mpc # FIXME: h is hard-coded here.
	y = pk * k**2. * window(k, R) # Integrand samples
	y = np.nan_to_num(y) # Remove NaNs
	s8_sq = scipy.integrate.simps(y, k) / (2. * np.pi**2.)
	return np.sqrt(s8_sq)

def discrete_realisation_sigma8(k, pk):
	"""Calculate sigma8 for a realisation of the power spectrum."""
	R = 8. / 0.7 # 8 h^-1 Mpc # FIXME: h is hard-coded here.
	y = pk * window(k, R) # Integrand samples
	y = np.nan_to_num(y) # Remove NaNs
	return np.sqrt( np.sum(y) )

def realisation_sigmaR(k, pk, R):
	"""Calculate sigma_R for a realisation of the power spectrum."""
	R = R / 0.7 # 8 h^-1 Mpc # FIXME: h is hard-coded here.
	y = pk * k**2. * window(k, R) # Integrand samples
	y = np.nan_to_num(y) # Remove NaNs
	sR_sq = scipy.integrate.simps(y, k) / (2. * np.pi**2.)
	return np.sqrt(sR_sq)


def calc_sigma8(cosmo=DEFAULT_COSMOLOGY):
	"""Calculate sigma8 for a given cosmology."""
	k = np.linspace(0., 2., 1e4)
	R = 8. / 0.7 # 8 h^-1 Mpc # FIXME: h is hard-coded here.
	
	pk = cp.perturbation.power_spectrum(k, z=0.0, **cosmo)
	y = pk * k**2. * window(k, R) # Integrand samples
	y = np.nan_to_num(y) # Remove NaNs
	s8_sq = scipy.integrate.simps(y, k) / (2. * np.pi**2.)
	return np.sqrt(s8_sq), k, y

def calc_sigmaR(R, cosmo=DEFAULT_COSMOLOGY):
	"""Calculate sigma_R for a given cosmology."""
	k = np.linspace(0., 2., 1e4)
	R /= 0.7 # in h^-1 Mpc # FIXME: h is hard-coded here.
	
	pk = cp.perturbation.power_spectrum(k, z=0.0, **cosmo)
	y = pk * k**2. * window(k, R) # Integrand samples
	y = np.nan_to_num(y) # Remove NaNs
	sR_sq = scipy.integrate.simps(y, k) / (2. * np.pi**2.)
	return np.sqrt(sR_sq)
