import numpy as np
import scipy
import scipy.interpolate
import math
from seren3 import cosmology
from pynbody.array import SimArray
from helper_functions import print_msg, get_eval
import warnings
from enum import Enum

import scipy.interpolate as si
import scipy.integrate
import scipy.fftpack as fft
import matplotlib.pylab as plt

class FieldFilter(object):

    def __init__(self):
        raise RuntimeError, "Cannot instantiate directly, use a subclass instead"

    def M_to_R(self, M):
        """Return the mass scale (Msol h^-1) for a given length (Mpc h^-1 comoving)"""
        return (M / (self.gammaF * self.rho_bar)) ** 0.3333

    def R_to_M(self, R):
        """Return the length scale (Mpc h^-1 comoving) for a given spherical mass (Msol h^-1)"""
        return self.gammaF * self.rho_bar * R ** 3

    @staticmethod
    def Wk(kR):
        raise RuntimeError, "Not implemented"

class TophatFilter(FieldFilter):

    def __init__(self, **cosmo):
        self.gammaF = 4 * math.pi / 3
        self.cosmo = cosmo

        omega0 = cosmo["omega_M_0"]
        rho_bar = SimArray(cosmology.rho_mean_z(omega0, **cosmo), "kg m**-3")
        self.rho_bar = rho_bar.in_units("Msol Mpc^-3") # h^2 a^-3")
        self.rho_bar *= cosmo['h']**2 * cosmo['aexp']**3
        self.rho_bar.units = "Msol Mpc^-3 h^2 a^-3"

    @staticmethod
    def Wk(kR):
        return 3 * (np.sin(kR) - kR * np.cos(kR)) / (kR) ** 3

TF = Enum('k', 'c', 'b', 'phot', 'massless_neutrino', 'massive_neutrino',
          'tot', 'nonu', 'totde', 'weyl', 'vcdm', 'vbaryon', 'vbc')

_camb_dir = "/lustre/scratch/astro/ds381/camb/CAMB/MUSIC/"
_camb_fbase = "Unigrid_MUSIC"
class PowerSpectrumCamb(object):
    '''
    Class to read and compute power spectra from CAMB transfer functions

    Parameters:
            * context (pynbody.snapshot): the snapshot containing desired cosmological parameters
            * filename (string): transfer function filename (Jan 2015 release)
    '''
    def __init__(self, log_interpolation=True, **cosmo):
        
        self._log_interp = log_interpolation
        self.cosmo = cosmo.copy()
        if 'As' not in cosmo:
            As = 2.243e-9
            print 'Using As = %e' % As
            self.cosmo['As'] = As
        if 'ns' not in cosmo:
            ns = 0.961
            print 'Using ns = %f' % ns
            self.cosmo['ns'] = ns
        if 'sigma_8' not in cosmo:
            sigma_8 = 0.82
            print 'Using sigma-8 = %f' % sigma_8
            self.cosmo['sigma_8'] = sigma_8
        # self.cosmo['n'] = self.cosmo['ns']
        self.cosmo['N_nu'] = 0.
        self.cosmo['omega_n_0'] = 0.
        self.cosmo['omega_k_0'] = 0.

        self._default_filter = TophatFilter(**self.cosmo)
        self._norm = 1
        self._load("%s_transfer_z0_out.dat" % _camb_fbase)
        self._normalise()

        # Now load for the desired redshift
        self._load("%s_transfer_z%d_out.dat" % (_camb_fbase, self.cosmo['z']))

    def _normalise(self):
        import cosmolopy.perturbation as cpt
        # sigma_8_now = cpt.sigma_r(
        #     8. / self.cosmo['h'], **self.cosmo)[0]
        sigma_8_now = self.cosmo["sigma_8"]
        self.set_sigma8(sigma_8_now / self._lingrowth)

    def set_sigma8(self, sigma8):
        current_sigma8_2 = self.get_sigma8() ** 2
        self._norm *= sigma8 ** 2 / current_sigma8_2

    def get_sigma8(self):
        v = variance(8.0, self, self._default_filter, True)
        current_sigma8 = math.sqrt(v) / self._lingrowth  # sigma 8 at z=0
        return current_sigma8

    def _init_interpolation(self):
        if self._log_interp:
            self._interp = scipy.interpolate.interp1d(
                np.log(self.k), np.log(self.Pk))
        else:
            self._interp = scipy.interpolate.interp1d(np.log(self.k), self.Pk)

    def _load(self, filename):
        self._tfs = np.loadtxt("%s/%s" % (_camb_dir, filename), unpack=True)
        k = self._tfs[TF.k.index]
        self._orig_k_min = k.min()
        self._orig_k_max = k.max()

        tf = self._tfs[TF.tot.index]
        Pk = self.cosmo['As'] * (k ** self.cosmo['ns']) * (tf ** 2)

        bot_k = 1.e-5

        if k[0] > bot_k:
            # extrapolate out
            n = math.log10(Pk[1] / Pk[0]) / math.log10(k[1] / k[0])

            Pkinterp = 10 ** (math.log10(Pk[0]) - math.log10(k[0] / bot_k) * n)
            k = np.hstack((bot_k, k))
            Pk = np.hstack((Pkinterp, Pk))

        top_k = 1.e7

        if k[-1] < top_k:
            # extrapolate out
            n = math.log10(Pk[-1] / Pk[-2]) / math.log10(k[-1] / k[-2])

            Pkinterp = 10 ** (math.log10(Pk[-1]
                                         ) - math.log10(k[-1] / top_k) * n)
            k = np.hstack((k, top_k))
            Pk = np.hstack((Pk, Pkinterp))

        self._Pk = Pk.view(SimArray)
        self._Pk.units = "Mpc^3 h^-3"

        self.k = k.view(SimArray)
        self.k.units = "Mpc^-1 h a^-1"

        self._lingrowth = 1.
        if self.cosmo['z'] != 0:
            # self._lingrowth = cosmology.lingrowthfac(self.cosmo['z'], **self.cosmo)
            self._lingrowth = cosmology.D_z(**self.cosmo)

        self.min_k = self.k.min()
        self.max_k = self.k.max()

        self._init_interpolation()

    @property
    def Pk(self):
        return self._norm * self._Pk

    def __call__(self, k):
        if self._log_interp:
            return self._norm * np.exp(self._interp(np.log(k)))
        return self._norm * self._interp(np.log(k))

    def TF_Pk(self, TF_i):
        '''
        Compute power spectrum with desired transfer function
        '''
        k = self._tfs[0]
        TF = self._tfs[TF_i]
        Pk = self.cosmo['As'] * (k ** self.cosmo['ns']) * (TF ** 2)
        return k, (self._norm * Pk)

    def TF_species(self, species):
        index = 0
        for i in TF:
            if i.key == species:
                index = i.index
                break
        return self.TF_Pk(index)

#######################################################################
# Variance calculation
#######################################################################

def variance(M, powspec, f_filter=TophatFilter, arg_is_R=False):
    if hasattr(M, '__len__'):
        ax = pynbody.array.SimArray(
            [variance(Mi, powspec, f_filter, arg_is_R) for Mi in M])
        # hopefully dimensionless
        ax.units = powspec.Pk.units * powspec.k.units ** 3
        return ax

    if arg_is_R:
        R = M
    else:
        R = f_filter.M_to_R(M)

    integrand = lambda k: k ** 2 * powspec(k) * f_filter.Wk(k * R) ** 2
    integrand_ln_k = lambda k: np.exp(k) * integrand(np.exp(k))
    v = scipy.integrate.romberg(integrand_ln_k, math.log(powspec.min_k), math.log(
        1. / R) + 3, divmax=10, rtol=1.e-4) / (2 * math.pi ** 2)

    return v