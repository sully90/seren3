import numpy as np


def kmin(N, boxsize):
    return (2. * np.pi) / boxsize


def kmax(N, boxsize):
    return kmin(N, boxsize) * float(N) / 2.


def z_to_age(zmax=20., zmin=0., return_inverse=False, **cosmo):
    """ Return functions to compute age/z from each other respectivly """
    import cosmolopy.distance as cd
    return cd.quick_age_function(zmax=zmax, zmin=zmin, return_inverse=return_inverse, **cosmo)


def fgrowth(**cosmo):
    """ Return Lahav et al. (1991) fit to dln(D)/dln(a) """
    return ((cosmo['omega_M_0'] * (1. + cosmo['z']) ** 3.)
            / (cosmo['omega_M_0'] * (1. + cosmo['z']) -
               (cosmo['omega_M_0'] + cosmo['omega_lambda_0'] - 1.)
               * (1. + cosmo['z']) ** 2.
               + cosmo['omega_lambda_0'])) ** (4. / 7.)


def rho_crit_now(units='si', **cosmo):
    '''
    Return critial density of the universe today
    H0 - Hubble constant in units kms^-1 Mpc^-1
    '''
    H0 = cosmo['h'] * 100.
    G = None
    if units == 'si':
        G = 6.6743E-11
    elif units == 'cgs':
        G = 6.6743E-8  # cgs
    # H0 *= (1000 / 3.08567758E22)  # s^-1
    rho_crit_0 = (3 * (H0 * (1000. / 3.08567758E22)) ** 2) / (8 * np.pi * G)
    return rho_crit_0


def rho_mean_z(omega0, units='si', **cosmo):
    '''
    Return mean density of the universe at redshift z
    H0 - Hubble constant in units kms^-1 Mpc^-1
    '''
    return omega0 * rho_crit_now(units=units, **cosmo) * (1. + cosmo['z']) ** 3


def hzoverh0(**cosmo):
    """ returns: H(a) / H0  = [omegam/a**3 + (1-omegam)]**0.5 """
    return np.sqrt(cosmo['omega_M_0'] * np.power(cosmo['aexp'], -3) + (1. - cosmo['omega_M_0']))


def _lingrowthintegrand(a, omegam):
    """ (e.g. eq. 8 in lukic et al. 2008)   returns: da / [a*H(a)/H0]**3 """
    return np.power((a * hzoverh0(**{'aexp': a, 'omega_M_0': omegam})), -3)


def lingrowthfac(red, return_norm=False, **cosmo):
    """
    returns: linear growth factor, b(a) normalized to 1 at z=0, good for flat lambda only
    a = 1/1+z
    b(a) = Delta(a) / Delta(a=1)   [ so that b(z=0) = 1 ]
    (and b(a) [Einstein de Sitter, omegam=1] = a)

    Delta(a) = 5 omegam / 2 H(a) / H(0) * integral[0:a] [da / [a H(a) H0]**3]
    equation  from  peebles 1980 (or e.g. eq. 8 in lukic et al. 2008) """
    # need to add w ~= , nonflat, -1 functionality

    import scipy.integrate

    if (abs(cosmo['omega_M_0'] + cosmo['omega_lambda_0'] - 1.) > 1.e-4):
        raise RuntimeError(
            "Linear growth factors can only be calculated for flat cosmologies")

    # 1st calc. for z=z
    lingrowth = scipy.integrate.quad(
        _lingrowthintegrand, 0., cosmo['aexp'], (cosmo['omega_M_0']))[0]
    lingrowth *= 5. / 2. * cosmo['omega_M_0'] * \
        hzoverh0(**cosmo)

    # then calc. for z=0 (for normalization)
    a0 = 1.
    lingrowtha0 = scipy.integrate.quad(
        _lingrowthintegrand, 0., a0, (cosmo['omega_M_0']))[0]
    lingrowtha0 *= 5. / 2. * \
        cosmo['omega_M_0'] * \
        hzoverh0(**{'aexp': a0, 'omega_M_0': cosmo['omega_M_0']})

    lingrowthfactor = lingrowth / lingrowtha0
    if return_norm:
        return lingrowthfactor, lingrowtha0
    else:
        return lingrowthfactor


def rate_linear_growth(**cosmo):
    """Calculate the linear growth rate b'(a), normalized
    to 1 at z=0, for the cosmology of snapshot f.
    The output is in 'h Gyr^-1' by default. If a redshift z is specified,
    it is used in place of the redshift in output f."""

    z = cosmo['z']
    a = 1. / (1. + z)
    omegam0 = cosmo['omega_M_0']
    omegal0 = cosmo['omega_lambda_0']

    b, X = lingrowthfac(z, return_norm=True, **cosmo)
    I = _lingrowthintegrand(a, omegam0)

    term1 = -(1.5 * omegam0 * a ** -3) * b / \
        np.sqrt(1. - omegam0 + omegam0 * a ** -3)
    term2 = (2.5 * omegam0) * hzoverh0(**cosmo) ** 2 * a * I / X

    res = cosmo['h'] * (term1 + term2) * 100. #  km s^-1 Mpc^-1

    return res


def D_z(**cosmo):
    """
    Unnormalised linear growth factor
    D(a) = 5 omegam / 2 H(a) / H(0) * integral[0:a] [da / [a H(a) H0]**3]
    equation  from  peebles 1980 (or e.g. eq. 8 in lukic et al. 2008) """

    import scipy.integrate

    omegam0 = cosmo['omega_M_0']
    omegal0 = cosmo['omega_lambda_0']

    if (abs(omegam0 + omegal0 - 1.) > 1.e-4):
        raise RuntimeError(
            "Linear growth factors can only be calculated for flat cosmologies")

    a = 1. / (1. + cosmo['z'])

    # 1st calc. for z=z
    lingrowth = scipy.integrate.quad(_lingrowthintegrand, 0., a, (omegam0))[0]
    lingrowth *= 5. / 2. * omegam0 * hzoverh0(**cosmo)
    return lingrowth


def omega_z(omega0, z):
    return omega0 / (omega0 + (1. / (1. + z)) ** 3 * (1 - omega0))


def Hubble_z(z, omega_m, H0):
    aexp = 1. / (1. + z)
    return ((omega_m / aexp ** 3) + (1 - omega_m)) ** 0.5 * H0


def del_c(z, omega_m):
    a = 1. / (1 + z)
    x = -((1 - omega_m) * a ** 3) / (omega_m + (1 - omega_m) * a ** 3)
    del_c = (178 + 82 * x - 39 * (x ** 2)) / (1 + x)
    # print 'del_c(%f) = %1.2e'%(z, del_c)

    return del_c

# Hoeft et al. 2006 characteristic mass


def hoeft_Mc(z, omega_m=0.27):
    if z < 0:
        z = 0
    # Tau encodes evolution of min. virial temp.
    tau_z = 0.73 * (z + 1) ** 0.18 * np.exp(-(0.25 * z) ** 2.1)
    del_c_z = del_c(z, omega_m=omega_m)
    del_c_0 = del_c(0, omega_m=omega_m)

    Mc_equ6 = (tau_z * (1. / (1 + z))) ** (3. / 2.) * \
        (del_c_0 / del_c_z) ** 0.5
    # print 'Equ 6 = ', Mc_equ6
    Mc = Mc_equ6 * 10 ** 10
    return Mc


def okamoto_Mc(z, Vc, H0, omega_m=0.27):
    import constants as C
    omega_m = omega_z(omega_m, z)
    del_vir = del_c(z, omega_m)
    Vc_3 = Vc ** 3
    A = 1. / np.sqrt(0.5 * del_vir * omega_m)
    B = (1. + z) ** (-3. / 2.)
    return ((Vc_3 * A * B) / (C.G * H0)) / C.Msun  # Msun

# Okamoto 2008 Virial Temperature


def T_vir(M, z, omega_m, H0, mu=0.59):
    import constants as C
    delta_vir = del_c(z, omega_m)
    #delta_vir = 1000
    M *= C.Msun
    return 0.5 * ((mu * C.mp) / C.kb) * ((delta_vir * omega_m) / 2.) ** (1. / 3.) * (1 + z) * (C.G * M * H0) ** (2. / 3.)


def T_vir_new(M, z, omega_m, H0, mu=0.59):
    import constants as C
    del_c_z = del_c(z, omega_m)
    M = M.in_units('kg')
    return 0.5 * ((mu * C.mp) / C.kb) * ((del_c_z * omega_m) / 2.) ** (1. / 3.) * (1. + z) * (C.G * M * H0.in_units('1/s')) ** (2. / 3.)


def V_c(T_vir=None, mu=0.59, M=None, Rvir=None):
    import constants as C
    if T_vir:
        V_circ = np.sqrt((2 * C.kb * T_vir) / (mu * C.mp))
    elif (M is not None) and (Rvir is not None):
        V_circ = np.sqrt(C.G * M / Rvir)
    else:
        raise Exception(
            "Must supply Virial Temperature or Mass and Virial Radius!")
    return V_circ


def density_power_spectrum(field, boxsize, species, kernel=None, dimensionless=True, **cosmo):
    '''
    Compute the density power spectrum given a continuous field

    Parameters:
            * field (ndarray): the input ND field in units of kg/m^3
            * boxsize (float): boxsize in Mpc/h
            * species (string): CDM or baryons
            * dimensionless (bool, True): return the dimensionless power spectrum
            * cosmo (dict): Dictionary containing cosmological model
    '''
    from . import power_spectrum

    boxsize = [float(boxsize)] * len(field.shape)
    omega0 = cosmo['omega_b_0'] if species.lower() == 'baryons' else (
        cosmo['omega_M_0'] - cosmo['omega_b_0'])

    rho_mean = rho_mean_z(omega0, **cosmo)
    delta = (field - rho_mean) / rho_mean

    ps, kbins = (None, None)
    if kernel == 'cic':
        print 'Performing CIC deconvolution in PS estimation...'
        ps, kbins = power_spectrum.power_spectrum_1d_cic(delta, boxsize)
    else:
        ps, kbins = power_spectrum.power_spectrum_1d(delta, boxsize)

    # Dimensionless density power spectrum
    if dimensionless:
        ps = np.sqrt(ps * (kbins ** 3) / (2 * np.pi ** 2))

    return ps, kbins


def velocity_power_spectrum(field, boxsize, species, kernel=None, dimensions_km_s=True):
    '''
    Compute the velocity power spectrum given a continuous field

    Parameters:
            * field (ndarray): the input ND field in units of km/s
            * boxsize (float): boxsize in Mpc/h
            * species (string): CDM or baryons
            * dimensions_km_s (bool, True): return the power spectrum in units of km/s
            * cosmo (dict): Dictionary containing cosmological model
    '''
    from . import power_spectrum

    boxsize = [float(boxsize)] * len(field.shape)

    ps, kbins = (None, None)

    if kernel == 'cic':
        print 'Performing CIC deconvolution in PS estimation...'
        ps, kbins = power_spectrum.power_spectrum_1d_cic(field, boxsize)
    else:
        ps, kbins = power_spectrum.power_spectrum_1d(field, boxsize)

    if dimensions_km_s:
        ps = np.sqrt(ps * (kbins ** 3) / (2 * np.pi ** 2))

    return ps, kbins


def linear_velocity_ps(k, delta_k, **cosmo):
    '''
    Compute velocity power spectrum using equations in Iliev et al. 2007
    k - wavenumber
    delta_k - dimensionless density power spectrum
    H0 - Hubble constant today
    omegam0 - Total matter density today
    omegal0 - Totel dark energy density today
    z - redshift to compute velocity power spectrum
    '''
    # First, compute the growth factor
    D_z_growth = D_z(**cosmo)
    H0 = cosmo['h'] * 100.
    # Compute E(z) = H(z)/H0 term
    E_z = hzoverh0(**cosmo)
    # Compute and return linear power spectrum
    t1 = delta_k ** 2 / k ** 2
    t2 = (9 * H0 ** 2 * cosmo['omega_M_0'] **
          2 * (1. + cosmo['z']) ** 4) / (4 * E_z ** 2)
    t3 = (1 - (5 / (3 * (1. + cosmo['z']) * D_z_growth))) ** 2

    return np.sqrt(t1 * t2 * t3)

__all__ = ["density_power_spectrum",
           "velocity_power_spectrum", "linear_velocity_ps"]
