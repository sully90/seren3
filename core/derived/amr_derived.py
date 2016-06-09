import seren3
import numpy as np
from pymses.utils import constants as C

mH = C.mH.coeff  # kg
kB = C.kB.coeff  # m^2 kg s^-2 K-1
e = C.eV.coeff  # electron charge in J

@seren3.derived_quantity(requires=["xHII"], latex=r'xHI')
def amr_xHI(context, dset):
    '''
    Hydrogen neutral fraction
    '''
    return 1. - dset['xHII']

@seren3.derived_quantity(requires=["rho"], latex=r'nH [cm$^{-3}$]')
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    from pymses.utils import constants as C
    return dset["rho"] * context.info["unit_density"].express(C.H_cc)

@seren3.derived_quantity(requires=["rho"], latex=r'nHe [cm$^{-3}$]')
def amr_nHe(context, dset):
    '''
    Helium number density
    '''
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    nHe = 0.25 * amr_nH(context, dset) * (Y_frac/X_frac)
    return nHe


@seren3.derived_quantity(requires=["rho", "xHII"], latex=r'nHI [cm$^{-3}$]')
def amr_nHI(context, dset):
    '''
    Return neutral hydrogen number density
    '''
    from pymses.utils import constants as C
    return dset["rho"] * context.info["unit_density"].express(C.H_cc) * (1. - dset["xHII"])


@seren3.derived_quantity(requires=["rho", "xHII"], latex=r'nHII [cm$^{-3}$]')
def amr_nHII(context, dset):
    '''
    Return ionized hydrogen number density
    '''
    from pymses.utils import constants as C
    return dset["rho"] * context.info["unit_density"].express(C.H_cc) * dset["xHII"]


@seren3.derived_quantity(requires=["nH", "dx"], latex=r'NH [cm$^{-2}$]')
def amr_NH(context, dset):
    '''
    Return hydrogen column density
    '''
    from pymses.utils import constants as C
    dx = None
    if "dx" in dset:
        dx = dset["dx"]
    else:
        dx = dset.get_sizes()
    return amr_nH(context, dset) * dx * context.info["unit_length"].express(C.cm)


@seren3.derived_quantity(requires=["nHI", "dx"], latex=r'NHI [cm$^{-2}$]')
def amr_NHI(context, dset):
    '''
    Return neutral hydrogen column density
    '''
    from pymses.utils import constants as C
    dx = None
    if "dx" in dset:
        dx = dset["dx"]
    else:
        dx = dset.get_sizes()
    return amr_nHI(context, dset) * dx * context.info["unit_length"].express(C.cm)


@seren3.derived_quantity(requires=["nHII", "dx"], latex=r'NHII [cm$^{-2}$]')
def amr_NHII(context, dset):
    '''
    Return ionized hydrogen column density
    '''
    from pymses.utils import constants as C
    dx = None
    if "dx" in dset:
        dx = dset["dx"]
    else:
        dx = dset.get_sizes()
    return amr_nHII(context, dset) * dx * context.info["unit_length"].express(C.cm)


@seren3.derived_quantity(requires=["rho", "dx"], latex=r'Cell Mass [code_mess]')
def amr_cell_mass(context, dset):
    '''
    Return cell mass in code units
    '''
    return dset['rho'] * dset['dx'] ** 3


@seren3.derived_quantity(requires=["rho"], latex=r"$\Delta_{\mathrm{b}}$")
def amr_baryon_overdensity(context, dset):
    '''
    Returns baryon overdensity (rho/rho_mean)
    '''
    from seren3 import cosmology
    from pymses.utils import constants as C
    cosmo = context.cosmo
    omegab_0 = cosmo['omega_b_0']
    rho_mean = cosmology.rho_mean_z(omegab_0, **cosmo)

    unit_d = context.info['unit_density'].express(C.kg / C.m**3)
    rho = dset['rho'] * unit_d

    return rho / rho_mean


@seren3.derived_quantity(requires=["rho"], latex=r'$\delta_{\mathrm{b}}$')
def amr_delta(context, dset):
    from seren3 import cosmology
    from pymses.utils import constants as C
    cosmo = context.cosmo
    omegab_0 = cosmo['omega_b_0']
    rho_mean = cosmology.rho_mean_z(omegab_0, **cosmo)

    unit_d = context.info['unit_density'].express(C.kg / C.m**3)
    rho = dset['rho'] * unit_d

    return (rho - rho_mean) / rho_mean


@seren3.derived_quantity(requires=["P", "rho"], latex=r'$c_{s} [code_velocity]$')
def amr_cs(context, dset):
    return np.sqrt(1.66667 * dset['P'] / dset['rho'])


@seren3.derived_quantity(requires=["xHII", "xHeII", "xHeIII"], latex=r"$\mu$")
def amr_mu(context, dset):
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    return 1. / ( X_frac*(1. + dset["xHII"]) + 0.25 * Y_frac * (1. + dset["xHeII"] + 2. * dset["xHeIII"]) )


@seren3.derived_quantity(requires=["P", "rho"], latex=r'T/$\mu$ [K]')
def amr_T2(context, dset):
    return (dset['P'] / dset['rho']) * context.info['unit_temperature'].val  # K


@seren3.derived_quantity(requires=["T2", "mu"], latex=r'T [K]')
def amr_T(context, dset):
    mu = amr_mu(context, dset)
    return amr_T2(context, dset) / mu


@seren3.derived_quantity(requires=["T"], latex=r'kT [keV]')
def amr_kT(context, dset):
    T = amr_T(context, dset)
    return (kB * T / e) / 1e3  # keV


@seren3.derived_quantity(requires=["P", "rho", "mu"], latex=r'S [m$^{2}$ keV]')
def amr_entropy(context, dset):
    mu = amr_mu(context, dset) * context.C.mH.val
    rho = dset['rho'] * context.info['unit_density'].val  # kg/m**3
    gammam1 = 2. / 3.
    entropy = amr_kT(context, dset) / ((rho / mu)**gammam1)
    return entropy  # m^2 keV


# @seren3.derived_quantity(requires=["Fp1", "Fp2", "Fp3"], latex=r'$\Gamma$ [s$^{-1}$]')
@seren3.derived_quantity(requires=["Np1", "Np2", "Np3"], latex=r'$\Gamma$ [s$^{-1}$]')
def amr_Gamma(context, dset, iIon=0):
    """ Photoionization rate """
    from seren3.utils import vec_mag as mag

    emi = 0.
    # shape = dset['Fp1'].shape
    # ndim = dset['Fp1'].ndim
    unit_Fp = context.info['unit_photon_flux_density'].val
    for i in range(1, context.info['nGroups'] + 1):
        # Fp = None
        # if ndim == 3:
        #     Fp = mag(dset['Fp%d' % i]).reshape(shape[0] * shape[1], 3) * unit_Fp.val
        # else:
        #     Fp = mag(dset['Fp%d' % i]) * unit_Fp.val

        # emi = emi + Fp * \
        #     context.info['group%d' % i]['csn'][iIon].val

        emi = emi + dset["Np%d" % i] * unit_Fp * \
            context.info['group%d' % i]['csn'][iIon].val

    # if ndim == 3:
    #     return emi.reshape(shape[0], shape[1])
    # else:
    #     return emi
    return emi

# @seren3.derived_quantity(requires=["rho", "xHII", "xHeII", "xHeIII", "Fp1", "Fp2", "Fp3"], latex=r"$\mathcal{H}$ [erg cm$^{-3}$ s$^{-1}$]")
@seren3.derived_quantity(requires=["rho", "xHII", "xHeII", "xHeIII", "Np1", "Np2", "Np3"], latex=r"$\mathcal{H}$ [erg cm$^{-3}$ s$^{-1}$]")
def amr_PHrate(context, dset):
    """ Photoheating rate """
    from seren3.utils import vec_mag as mag
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    # mH= 1.6600000e-24
    # unit_nH = X_frac * (context.info['unit_density'].express(context.C.g_cc)) / mH

    homPHrates = [4.719772E-24, 7.311075E-24, 8.531654E-26]  # From Joki's ramses_info.f90
    nH = amr_nH(context, dset)  # cm^-3
    # nH = dset['rho'] * unit_nH

    nHI = nH * (1. - dset['xHII'])  # nHI

    nHe = 0.25 * nH * (Y_frac/X_frac)

    nHeI = nHe * ( (1. - dset['xHeII'] - dset['xHeIII']) )  # nHeI
    nHeI[nHeI < 0.] = 0.

    nHeII = nHe * dset['xHeII']

    nN = [nHI, nHeI, nHeII]

    # print nHI.shape, nHeI.shape, nHeII.shape, dset["Np1"].shape

    emi = np.zeros(nHI.shape)
    if context.info['nGroups'] == 0:
        for iIon in range(0, 3):
            emi += nH[iIon, :] * homPHrates[iIon]

    else:
        C = context.C
        Np_ndim = dset["Np1"].ndim
        info_rt = context.info_rt
        unit_Fp = info_rt['unit_photon_flux_density'].express(C.cm**-2 * C.s**-1)

        for iGroup in range(1, info_rt['nGroups'] + 1):
            for iIon in range(0, 3):
                Np = dset["Np%d" % iGroup] * unit_Fp

                emi = emi + nN[iIon] * Np \
                    * (info_rt["group%d" % iGroup]["cse"][iIon].express(C.cm**2) * info_rt["group%d" % iGroup]["egy"][0].express(C.erg) \
                    - info_rt["group%d" % iGroup]["csn"][iIon].express(C.cm**2) * (info_rt["photon_properties"]["groupL0"][iIon].express(C.erg)))

                # emi = emi + nN[iIon, :] * dset["Np%d" % iGroup] * unit_Fp \
                #     * (info_rt["group%d" % iGroup]["cse"][iIon].express(C.cm**2) * info_rt["group%d" % iGroup]["egy"][0].express(C.erg) \
                #     - info_rt["group%d" % iGroup]["csn"][iIon].express(C.cm**2) * (info_rt["photon_properties"]["groupL0"][iIon].express(C.erg)))

    if emi.min() < 0:
        raise Exception("NEGATIVE EMI")

    return emi

@seren3.derived_quantity(requires=["Fp1", "Fp2", "Fp3"], latex=r'Fp')
def amr_Fp(context, dset):
    '''
    Combined photon flux in all groups
    '''
    return dset['Fp1'] + dset['Fp2'] + dset['Fp3']

@seren3.derived_quantity(requires=["Np1", "Np2", "Np3"], latex=r'Np')
def amr_Np(context, dset):
    '''
    Combined photon flux in all groups
    '''
    return dset['Np1'] + dset['Np2'] + dset['Np3']

######################### Pynbody ######################

from pynbody import units as pynbody_units
from pynbody.snapshot import SimSnap
from pynbody.array import SimArray

@SimSnap.derived_quantity
def mu(sim):
    X_frac, Y_frac = (0.76, 0.24)
    return 1. / ( X_frac*(1. + sim["xHII"]) + 0.25 * Y_frac * (1. + sim["xHeII"] + 2. * sim["xHeIII"]) )

@SimSnap.derived_quantity
def T2(sim):
    unit_temperature = mH / kB

    p = sim.g['p'].in_units('m**-1 kg s**-2')
    rho = sim.g['rho'].in_units('kg m**-3')

    sim.g["T2"] = (p / rho) * unit_temperature
    sim.g["T2"].units = "K"
    return sim.g["T2"]

@SimSnap.derived_quantity
def T(sim):
    sim.g["T"] = sim.g["T2"] / sim.g["mu"]
    sim.g["T"].units = "K"
    return sim.g["T"]

@SimSnap.derived_quantity
def kT(sim):
    kT = (kB * sim.g["T"] / e) / 1.e3
    return SimArray(kT, "keV")

@SimSnap.derived_quantity
def entropy(sim):
    mu_mass = SimArray(sim.g["mu"] * mH, "kg")
    rho = sim.g["rho"].in_units("kg m**-3")
    gammam1 = 2./3.
    sim.g["entropy"] =  sim.g["kT"] / ((rho / mu_mass)**gammam1)
    sim.g["entropy"].units = "m**2 keV"
    return sim.g["entropy"]
