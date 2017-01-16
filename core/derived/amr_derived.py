import seren3
from seren3.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["pos"], unit=C.Msun)
def amr_r(context, dset, center=None):
    '''
    Radial position
    '''
    pos = dset["pos"]

    if center is not None:
        pos -= center
    return ((pos ** 2).sum(axis=1)) ** (1,2)

@seren3.derived_quantity(requires=["rho", "dx"], unit=C.Msun)
def amr_mass(context, dset):
    '''
    Return cell mass in solar masses
    '''
    rho = dset["rho"].in_units("Msol pc**-3")
    dx = dset["dx"].in_units("pc")
    mass = rho * (dx**3)
    mass.set_field_latex("$\\mathrm{M}$")
    return mass

@seren3.derived_quantity(requires=["rho"], unit=C.none)
def amr_baryon_overdensity(context, dset):
    '''
    Return baryon overdensity: rho/rho_mean
    '''
    units = "kg m**-3"
    rho_mean = context.quantities.rho_mean(species='b').in_units(units)
    rho = dset["rho"].in_units(units)

    overdensity = context.array(rho/rho_mean)
    overdensity.set_field_latex("$\Delta_{\mathrm{b}}$")
    return overdensity

@seren3.derived_quantity(requires=["rho"], unit=C.none)
def amr_deltab(context, dset):
    '''
    Returns dimensionless density contrast
    '''
    units = "kg m**-3"
    rho_mean = context.quantities.rho_mean(species="b").in_units(units)
    rho = dset["rho"].in_units(units)

    db = context.array((rho-rho_mean)/rho_mean)
    db.set_field_latex("$\delta_{\mathrm{b}}$")

    return db

@seren3.derived_quantity(requires=["nH", "xHII", "nHe", "xHeII", "xHeIII"], unit=C.cm**-3)
def amr_ne(context, dset):
    '''
    Returns number density of electrons in each cell
    '''
    ne = dset["nH"] * dset["xHII"]  # num. den. of electrons from ionised hydrogen
    ne += dset["nHe"] * (dset["xHeII"] + dset["xHeIII"])  # " " from ionised Helium
    ne.set_field_latex("n$_{\\mathrm{e}}")
    return context.array(ne, "m**-3").in_units("cm**-3")

@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    rho = dset["rho"].in_units("kg m**-3")
    mH = SimArray(context.C.mH)
    X_fraction = context.info.get("X_fraction", 0.76)
    H_frac = mH / X_fraction  # Hydrogen mass fraction
    nH = (rho/H_frac).in_units("m**-3")
    nH.set_field_latex("n$_{\\mathrm{H}}$")
    return nH.in_units("cm**-3")

@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
def amr_nHe(context, dset):
    '''
    Return Helium number density
    '''
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    nHe = 0.25 * amr_nH(context, dset) * (Y_frac/X_frac)
    nHe.set_field_latex("n$_{\\mathrm{He}}$")
    return nHe.in_units("m**-3").in_units("cm**-3")

@seren3.derived_quantity(requires=["xHII"], unit=C.none)
def amr_xHI(context, dset):
    '''
    Hydrogen neutral fraction
    '''
    val = 1. - dset["xHII"]
    xHI = SimArray(val)
    xHI.set_field_latex("$\\mathrm{x}_{\\mathrm{HI}}$")
    return xHI

@seren3.derived_quantity(requires=["nH", "xHI"], unit=C.H_cc)
def amr_nHI(context, dset):
    '''
    Neutral hydrogen number density
    '''
    nHI = SimArray(dset["nH"] * dset["xHI"], dset["nH"].units)
    nHI.set_field_latex("n$_{\\mathrm{HI}}$")
    return nHI

@seren3.derived_quantity(requires=["P", "rho"], unit=C.m/C.s)
def amr_cs(context, dset):
    '''
    Gas sound speed in m/s (units are convertabke)
    '''
    rho = dset["rho"]
    P = dset["P"]
    cs = np.sqrt(1.66667 * P / rho).in_units("m s**-1")
    cs.set_field_latex("$\\mathrm{c}_{s}$")
    return cs.in_units("m s**-1")

@seren3.derived_quantity(requires=["P", "rho"], unit=C.K)
def amr_T2(context, dset):
    '''
    Gas Temperature in units of K/mu
    '''
    mH = SimArray(context.C.mH)
    kB = SimArray(context.C.kB)
    T2 = (dset["P"]/dset["rho"] * (mH / kB)).in_units("K")
    T2.set_field_latex("T$_{2}$")
    return T2

@seren3.derived_quantity(requires=["xHII", "xHeII", "xHeIII"], unit=C.none)
def amr_mu(context, dset):
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    mu = 1. / ( X_frac*(1. + dset["xHII"]) + 0.25 * Y_frac * (1. + dset["xHeII"] + 2. * dset["xHeIII"]) )
    mu.set_field_latex("$\mu$")
    return mu

@seren3.derived_quantity(requires=["T2", "mu"], unit=C.K)
def amr_T(context, dset):
    T = dset["T2"].in_units("K")/dset["mu"]
    T.set_field_latex("$\\mathrm{T}$")
    return context.array(T, "K")

@seren3.derived_quantity(requires=["ne", "alpha_A"], unit=C.year)
def amr_trec(context, dset):
    '''
    Returns recombination time scale of each cell in seconds
    '''
    trec = 1./( dset["ne"].in_units("cm**-3") * dset["alpha_A"].in_units("cm**3 s**-1") )
    return trec.in_units("yr")

@seren3.derived_quantity(requires=["T"], unit=C.cm**3 * C.s**-1)
def amr_alpha_A(context, dset):
    '''
    Returns case A rec. coefficient [cm3 s-1] for HII (Hui&Gnedin'97)
    '''
    T = dset["T"].in_units("K")
    lambda_T = 315614./T
    alpha_A = context.array(1.269e-13 * lambda_T**1.503 / ( ( 1.0+(lambda_T/0.522)**0.47 )**1.923 ),\
             "cm**3 s**-1", latex="$\alpha_{\mathrm{A}}$")
    return alpha_A


@seren3.derived_quantity(requires=["T"], unit=C.cm**3 * C.s**-1)
def amr_alpha_B(context, dset):
    '''
    Returns case B rec. coefficient [cm3 s-1] for HII (Hui&Gnedin'97)
    '''
    T = dset["T"].in_units("K")
    lambda_T = 315614./T
    alpha_B = context.array(2.753e-14 * lambda_T**1.5 / ( (1.0+(lambda_T/2.74)**0.407)**2.242 ),\
             "cm**3 s**-1", latex="$\alpha_{\mathrm{B}}$")
    return alpha_B

    
############################################### RAMSES-RT ###############################################

@seren3.derived_quantity(requires=[], unit=C.m**-2 * C.s**-1)
def amr_Flux_Mag(context, dset, group=1):
    '''
    Computes magnitude of photon flux for each cell
    Note: does not specify required fields as we don't know which group
    to use until we get here
    '''
    flux = context.g["Fp%i" % group].flatten()
    x,y,z = flux.T

    return np.sqrt(x**2 + y**2 + z**2)


@seren3.derived_quantity(requires=["Np1", "Np2", "Np3"], unit=1./C.s)
def amr_Gamma(context, dset, iIon=0):
    '''
    Gas photoionization rate in [s^-1]
    '''
    emi = 0.
    for i in range(1, context.info["nGroups"] + 1):
        Np = dset["Np%i" % i]
        csn = SimArray(context.info["group%i" % i]["csn"][iIon])
        emi += Np * csn

    Gamma = emi.in_units("s**-1")
    Gamma.set_field_latex("$\Gamma$")
    return Gamma

@seren3.derived_quantity(requires=["nH", "xHII", "xHeII", "xHeIII", "Np1", "Np2", "Np3"], unit=C.erg * C.cm**-3 * C.s**-1)
def amr_PHrate(context, dset):
    """ Photoheating rate """
    from seren3.utils import vec_mag as mag
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    # mH= 1.6600000e-24
    # unit_nH = X_frac * (context.info['unit_density'].express(context.C.g_cc)) / mH

    homPHrates = [4.719772E-24, 7.311075E-24, 8.531654E-26]  # From Joki's ramses_info.f90
    nH = dset["nH"].in_units("cm**-3")
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
        info = context.info
        unit_Fp = info['unit_photon_flux_density'].express(C.cm**-2 * C.s**-1)

        for iGroup in range(1, info['nGroups'] + 1):
            for iIon in range(0, 3):
                Np = dset["Np%d" % iGroup] * unit_Fp

                emi = emi + nN[iIon] * Np \
                    * (info["group%d" % iGroup]["cse"][iIon].express(C.cm**2) * info["group%d" % iGroup]["egy"][0].express(C.erg) \
                    - info["group%d" % iGroup]["csn"][iIon].express(C.cm**2) * (info["photon_properties"]["groupL0"][iIon].express(C.erg)))

                # emi = emi + nN[iIon, :] * dset["Np%d" % iGroup] * unit_Fp \
                #     * (info["group%d" % iGroup]["cse"][iIon].express(C.cm**2) * info["group%d" % iGroup]["egy"][0].express(C.erg) \
                #     - info["group%d" % iGroup]["csn"][iIon].express(C.cm**2) * (info["photon_properties"]["groupL0"][iIon].express(C.erg)))

    if emi.min() < 0:
        raise Exception("NEGATIVE EMI")

    return context.array(emi, "erg cm**-3 s**-1", latex="$\mathcal{H}$")
