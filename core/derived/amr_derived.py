import seren3
from seren3.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["nH", "T2"])
def amr_dSFR(context, dset, **kwargs):

    '''
    Computes the (instantaneous) star formation rate density, in Msun/yr/kpc^3, from the gas
    '''
    import numpy as np
    from seren3.core.namelist import NML

    mH = context.array(context.C.mH)
    X_fraction = context.info.get("X_fraction", 0.76)
    H_frac = mH/X_fraction  # fractional mass of hydrogen

    nml = context.nml
    nH = dset["nH"].in_units("cm**-3")

    # Load star formation model params from the namelist
    n_star = context.array(nml[NML.PHYSICS_PARAMS]["n_star"], "cm**-3")  # cm^-3
    t_star = context.quantities.t_star.in_units("yr")

    # Compute the SFR density in each cell
    sfr = nH / (t_star*np.sqrt(n_star/nH))  # atoms/yr/cm**3
    sfr *= H_frac  # kg/yr/cm**3
    sfr.convert_units("Msol yr**-1 kpc**-3")

    impose_criterion = kwargs.pop("impose_criterion", False)
    # Impose density/temperature criterion
    if impose_criterion:
        idx = np.where(nH < n_star)
        sfr[idx] = 0.

        # Compute and subtract away the non-thermal polytropic temperature floor
        g_star = nml[NML.PHYSICS_PARAMS].get("g_star", 1.)
        T2_star = context.array(nml[NML.PHYSICS_PARAMS]["T2_star"], "K")
        Tpoly = T2_star * (nH/n_star)**(g_star-1.)  # Polytropic temp. floor
        Tmu = dset["T2"] - Tpoly  # Remove non-thermal polytropic temperature floor
        idx = np.where(Tmu > 2e4)
        sfr[idx] = 0.

    return sfr


@seren3.derived_quantity(requires=["pos"])
def amr_spherical_pos(context, dset):
    '''
    Return position in spherical polar coordinates
    '''
    x,y,z = dset["pos"].T

    center=[(max(x)+min(x))/2., (max(y)+min(y))/2., (max(z)+min(z))/2.]
    x-= center[0]; y -= center[1]; z -= center[2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y/x)
    phi = np.arccos(z/r)

    res = np.array([r, theta, phi]).T
    return context.array(res)
    

@seren3.derived_quantity(requires=["rho", "dx"])
def amr_mass(context, dset):
    '''
    Return cell mass in solar masses
    '''
    rho = dset["rho"]
    dx = dset["dx"]
    mass = rho * (dx**3)
    mass.set_field_latex("$\\mathrm{M}$")
    return mass

@seren3.derived_quantity(requires=["rho"])
def amr_baryon_overdensity(context, dset):
    '''
    Return baryon overdensity: rho/rho_mean
    '''
    units = "kg m**-3"
    rho_mean = context.quantities.rho_mean(species='b')
    rho = dset["rho"]

    overdensity = context.array(rho/rho_mean)
    overdensity.set_field_latex("$\Delta_{\mathrm{b}}$")
    return overdensity

@seren3.derived_quantity(requires=["rho"])
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

@seren3.derived_quantity(requires=["nHII", "nHe", "xHeII", "xHeIII"])
def amr_ne(context, dset):
    '''
    Returns number density of electrons in each cell
    '''
    n_e = dset["nHII"] + dset["nHe"]*(dset["xHeII"]+2.*dset["xHeIII"])
    return n_e

@seren3.derived_quantity(requires=["rho"])
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    rho = dset["rho"]
    mH = context.array(context.C.mH)
    X_fraction = context.info.get("X_fraction", 0.76)
    H_frac = mH/X_fraction

    nH = rho/H_frac
    nH.set_field_latex("$\mathrm{n}_{\mathrm{H}}$")
    return nH

@seren3.derived_quantity(requires=["rho"])
def amr_nHe(context, dset):
    '''
    Return Helium number density
    '''
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    nHe = 0.25 * amr_nH(context, dset) * (Y_frac/X_frac)
    nHe.set_field_latex("n$_{\\mathrm{He}}$")
    return nHe

@seren3.derived_quantity(requires=["xHII"])
def amr_xHI(context, dset):
    '''
    Hydrogen neutral fraction
    '''
    val = 1. - dset["xHII"]
    xHI = SimArray(val)
    xHI.set_field_latex("$\\mathrm{x}_{\\mathrm{HI}}$")
    return xHI

@seren3.derived_quantity(requires=["nH", "xHII"])
def amr_nHI(context, dset):
    '''
    Neutral hydrogen number density
    '''

    xHII = np.round( dset["xHII"], decimals=5 )
    nHI = SimArray(dset["nH"] * (1. - xHII), dset["nH"].units)
    # nHI = SimArray(dset["nH"] * (1. - dset["xHII"]), dset["nH"].units)
    nHI.set_field_latex("n$_{\\mathrm{HI}}$")
    return nHI

@seren3.derived_quantity(requires=["nH", "xHII"])
def amr_nHII(context, dset):
    '''
    Neutral hydrogen number density
    '''
    nHII = SimArray(dset["nH"] * dset["xHII"], dset["nH"].units)
    nHII.set_field_latex("n$_{\\mathrm{HII}}$")
    return nHII

@seren3.derived_quantity(requires=["P", "rho"])
def amr_cs(context, dset):
    '''
    Gas sound speed in m/s (units are convertabke)
    '''
    rho = dset["rho"]
    P = dset["P"]
    cs = np.sqrt(1.66667 * P / rho)
    cs.set_field_latex("$\\mathrm{c}_{s}$")
    return cs

@seren3.derived_quantity(requires=["P", "rho"])
def amr_T2(context, dset):
    '''
    Gas Temperature in units of K/mu
    '''
    mH = SimArray(context.C.mH)
    kB = SimArray(context.C.kB)
    T2 = (dset["P"]/dset["rho"] * (mH / kB))
    T2.set_field_latex("T$_{2}$")
    return T2

@seren3.derived_quantity(requires=["xHII", "xHeII", "xHeIII"])
def amr_mu(context, dset):
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    mu = 1. / ( X_frac*(1. + dset["xHII"]) + 0.25 * Y_frac * (1. + dset["xHeII"] + 2. * dset["xHeIII"]) )
    mu.set_field_latex("$\mu$")
    return mu

@seren3.derived_quantity(requires=["T2", "mu"])
def amr_T(context, dset):

    T = dset["T2"]/dset["mu"]
    T.set_field_latex("$\\mathrm{T}$")
    return context.array(T, "K")

@seren3.derived_quantity(requires=["nH"])
def amr_TJ(context, dset):
    '''
    Computes densty dependent Jeans temperature floor
    '''
    from seren3.core.snapshot import NML

    nH = dset["nH"]

    nml = context.nml
    PHYSICS_PARAMS = nml[NML.PHYSICS_PARAMS]
    n_star = SimArray(PHYSICS_PARAMS['n_star'], "cm**-3").in_units(nH.units)
    T2_star = PHYSICS_PARAMS['T2_star']
    g_star = PHYSICS_PARAMS.get('g_star', 2.0)

    return SimArray(T2_star * (nH / n_star) ** (g_star-1.0), "K")

@seren3.derived_quantity(requires=["T2", "TJ"])
def amr_T2_minus_Tpoly(context, dset):
    '''
    Returns T2 + TJ
    '''
    Tpoly = dset["T2"] - dset["TJ"]
    Tpoly.set_field_latex("$\\mathrm{Tpoly}$")
    return context.array(Tpoly, "K")

@seren3.derived_quantity(requires=["ne", "alpha_A"])
def amr_trec(context, dset):
    '''
    Returns recombination time scale of each cell in seconds
    '''
    trec = 1./( dset["ne"] * dset["alpha_A"] )
    return trec

@seren3.derived_quantity(requires=["T"])
def amr_alpha_A(context, dset):
    '''
    Returns case A rec. coefficient [cm3 s-1] for HII (Hui&Gnedin'97)
    '''
    T = dset["T"]
    lambda_T = 315614./T
    alpha_A = context.array(1.269e-13 * lambda_T**1.503 / ( ( 1.0+(lambda_T/0.522)**0.47 )**1.923 ),\
             "cm**3 s**-1", latex="$\alpha_{\mathrm{A}}$")
    return alpha_A


@seren3.derived_quantity(requires=["T"])
def amr_alpha_B(context, dset):
    '''
    Returns case B rec. coefficient [cm3 s-1] for HII (Hui&Gnedin'97)
    '''
    T = dset["T"]
    lambda_T = 315614./T
    alpha_B = context.array(2.753e-14 * lambda_T**1.5 / ( (1.0+(lambda_T/2.74)**0.407)**2.242 ),\
             "cm**3 s**-1", latex="$\alpha_{\mathrm{B}}$")
    return alpha_B

    
############################################### RAMSES-RT ###############################################


@seren3.derived_quantity(requires=["Np1", "Np2", "Np3"])
def amr_Gamma(context, dset, iIon=0):
    '''
    Gas photoionization rate in [s^-1]
    '''
    emi = 0.
    for i in range(1, context.info["nGroups"] + 1):
        Np = dset["Np%i" % i]
        csn = SimArray(context.info["group%i" % i]["csn"][iIon])
        emi += Np * csn

    Gamma = context.array(emi, "s**-1")
    Gamma.set_field_latex("$\Gamma$")
    return Gamma

@seren3.derived_quantity(requires=["nH", "xHII", "xHeII", "xHeIII", "Np1", "Np2", "Np3"])
def amr_PHrate(context, dset):
    """ Photoheating rate """
    from seren3.utils import vec_mag as mag
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    # mH= 1.6600000e-24
    # unit_nH = X_frac * (context.info['unit_density'].express(context.C.g_cc)) / mH

    homPHrates = [4.719772E-24, 7.311075E-24, 8.531654E-26]  # From Joki's ramses_info.f90
    nH = dset["nH"]
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
