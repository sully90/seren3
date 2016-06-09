import seren3
from .part_derived import *
import numpy as np

@seren3.derived_quantity(requires=["epoch"], latex=r'Age [Gyr]')
def star_age(context, dset, **kwargs):
    return part_age(context, dset, **kwargs)


@seren3.derived_quantity(requires=["age", "mass"], latex=r'sSFR [Gyr$^{-1}$]')
def star_sSFR_z(context, dset, nbins=100, **kwargs):
    from pynbody.snapshot.ramses import RamsesSnap
    from pynbody.analysis.cosmology import age as age_func
    from pynbody.analysis.cosmology import redshift as redshift_func
    from seren3.exceptions import NoParticlesException
    from pymses.utils import constants as C

    if len(dset["age"]) == 0 or len(dset["mass"]) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'star_sSFR')

    s = RamsesSnap("%s/output_%05d" % (context.path, context.ioutput), cpus=[1])
    age_simu = age_func(s)
    mass = dset["mass"]
    age = dset["age"]

    def sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = 1e-9 * nbins / (agerange[1] - agerange[0])
        # binnorm = nbins / (agerange[1] - agerange[0])
        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return sfrhist, binmps, binsize

    sfrhist, binmps, binsize = sfr(age, mass, **kwargs)
    z = redshift_func(s, age_simu-binmps)
    M_star = mass.sum()
    sSFR = (sfrhist * 1e9) / M_star  # Msun/yr -> Msun/Gyr -> /Gyr
    return {'sSFR' : sSFR, 'z' : z}  # SFR [Gyr^-1], z


@seren3.derived_quantity(requires=["age", "mass"], latex=r'sSFR [Gyr$^{-1}$]')
def star_sSFR(context, dset, nbins=100, **kwargs):
    from seren3.exceptions import NoParticlesException
    from pymses.utils import constants as C

    age = dset['age']  # Already in Gyr
    mass = dset['mass'] * context.info['unit_mass'].express(C.Msun)

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'star_sSFR')

    def sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = 1e-9 * nbins / (agerange[1] - agerange[0])
        # binnorm = nbins / (agerange[1] - agerange[0])
        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return sfrhist, binmps, binsize

    sfrhist, binmps, binsize = sfr(age, mass, **kwargs)
    M_star = mass.sum()
    sSFR = (sfrhist * 1e9) / M_star  # Msun/yr -> Msun/Gyr
    return {'sSFR' : sSFR, 'lookback-time' : binmps, 'binsize' : binsize}  # SFR [Msun Gyr^-1], Lookback Time [Gyr], binsize [Gyr]


@seren3.derived_quantity(requires=["age", "mass"], latex=r'SFR [M$_{\odot}$ Gyr$^{-1}$]')
def star_SFR(context, dset, nbins=100, **kwargs):
    from seren3.exceptions import NoParticlesException
    from pymses.utils import constants as C

    age = dset['age']  # Already in Gyr
    mass = dset['mass'] * context.info['unit_mass'].express(C.Msun)

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'star_sSFR')

    def sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = 1e-9 * nbins / (agerange[1] - agerange[0])
        # binnorm = nbins / (agerange[1] - agerange[0])
        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return sfrhist, binmps, binsize

    sfrhist, binmps, binsize = sfr(age, mass, **kwargs)
    SFR = (sfrhist * 1e9)  # Msun/yr -> Msun/Gyr
    return {'SFR' : SFR, 'lookback-time' : binmps, 'binsize' : binsize}  # SFR [Msun/Gyr^-1], Lookback Time [Gyr], binsize [Gyr]


@seren3.derived_quantity(requires=["mass", "metal", "age"], latex=r'L [J s$^{-1}$]')
def star_luminosity(context, dset, fesc=None, lambda_A=1600., interp=True, sed_envar='RAMSES_SED_DIR', **kwargs):
    '''
    Return the UV luminosity of stars from SED tables
    lambda_A - Wavelength in Angstroms
    '''
    import os
    from seren3.utils import io
    from pymses.utils import constants as C
    from seren3.exceptions import NoParticlesException

    if "mass" not in dset or len(dset["mass"]) == 0:
        raise NoParticlesException("No stars found", "star_luminosity")

    sedir = os.environ.get(sed_envar)

    if context.quantities.has_attr('sed') is False:
        context.quantities.add_attr('sed', io.read_seds(sedir))
    agebins, zbins, Ls, SEDs = context.quantities.get_attr('sed')
    nLs = len(Ls)
    # U = 1500  # Median UV SDSS band in Angstroms
    age = dset['age'] * 1e9  # Gyr -> yr
    Z_sun = 0.02
    Z = dset['metal'] / Z_sun  # Solar metallicity

    unit_m = context.info['unit_mass'].express(C.Msun)
    star_mass = dset['mass'] * unit_m

    if fesc is None:
        nml = context.nml
        fesc = float(nml[nml.NML.RT_PARAMS]['rt_esc_frac'].replace('d', 'e'))

    if interp:
        # Interpolate Luminosities from the grid
        from seren3.analysis.interpolate import interpolate3d
        U = np.ones(len(age)) * lambda_A
        star_lums = interpolate3d(
            U, age, Z, Ls, agebins, zbins, SEDs) * star_mass
        # return star_lums * lambda_A * fesc  # L / \AA -> L
        return star_lums * fesc  # L / \AA -> L

    else:
        # Locate nearest wavelength in table
        ii = 0
        while (ii < nLs) and (Ls[ii] < lambda_A):
            ii += 1
        star_lums = []

        for i in range(len(star_mass)):
            idx_z = (np.abs(Z[i] - zbins)).argmin()
            idx_age = (np.abs(age[i] - agebins)).argmin()
            star_lums.append(SEDs[ii, idx_age, idx_z] * star_mass[i])

        star_lums = np.array(star_lums)
        return star_lums * lambda_A * fesc  # L / \AA -> L


@seren3.derived_quantity(requires=["mass", "age"], latex='L [W]')
def star_luminosity_no_metal(context, dset, fesc=1., lambda_A=1600., interp=True, **kwargs):
    '''
    Return the UV luminosity of stars using bc03 SED tables
    lambda_A - Wavelength in Angstroms
    '''
    from seren3.utils import io
    from pymses.utils import constants as C

    if context.quantities.has_attr('sed') is False:
        context.quantities.add_attr('sed', io.read_seds())
    agebins, zbins, Ls, SEDs = context.quantities.get_attr('sed')
    nLs = len(Ls)
    # U = 1500  # Median UV SDSS band in Angstroms
    age = dset['age'] * 1e9  # Gyr -> yr

    unit_m = context.info['unit_mass'].express(C.Msun)
    star_mass = dset['mass'] * unit_m
    metals = np.zeros_like(star_mass)
    metals += 1e-3

    if interp:
        # Interpolate Luminosities from the grid
        from seren3.analysis.interpolate import interpolate3d
        U = np.zeros(len(age))
        for i in range(len(age)):
            U[i] = lambda_A
        star_lums = interpolate3d(
            U, age, metals, Ls, agebins, zbins, SEDs) * star_mass
        return star_lums * lambda_A * fesc  # L / \AA -> L

    else:
        # Locate nearest wavelength in table
        ii = 0
        while (ii < nLs) and (Ls[ii] < lambda_A):
            ii += 1
        star_lums = []

        for i in range(len(star_mass)):
            idx_z = (np.abs(metals[i] - zbins)).argmin()
            idx_age = (np.abs(age[i] - agebins)).argmin()
            star_lums.append(SEDs[ii, idx_age, idx_z] * star_mass[i])

        star_lums = np.array(star_lums)
        return star_lums * lambda_A * fesc  # L / \AA -> L


# REWRITE NION TO TAKE GROUP AS ARG, BUT IF NONE DO ALL GROUPS
@seren3.derived_quantity(requires=["metal", "age"], latex=r"$\dot{N_{\mathrm{ion}}}$ [# M$_{\odot}^{-1}$]")
def star_Nion_per_Msun(context, dset, dt=0., direction=None, group=1):
    from seren3.exceptions import NoParticlesException
    from seren3.utils import io
    from scipy.interpolate import interp2d


    print 'Computing Nion for photon group ', group
    Z_sun = 0.02
    nGroups = context.info_rt['nGroups']
    nIons = context.info_rt['nIons']
    nPhotons_idx = 0  # index of photon number in SED table

    agebins, zbins, SEDs = io.read_seds_from_lists(context.path, nGroups, nIons)
    igroup = group - 1
    fn = interp2d(zbins, agebins, SEDs[:,:,igroup,nPhotons_idx])

    age = dset['age']
    Z = dset['metal'] / Z_sun

    if dt > 0 and direction is not None:
        if 'b' == direction:
            idx = np.where(age-dt >= 0)
            age = age[idx] - dt
            Z = Z[idx]
        elif 'f' == direction:
            age += dt

    if len(age) == 0:
        raise NoParticlesException("No particles with (age - dt) > 0", "star_Nion_per_Msun")

    nStars = len(age)
    nPhotons = np.zeros(nStars)
    for i in xrange(nStars):
        nPhotons[i] = fn(Z[i], age[i])

    # Multiply by escape fraction and return nPhotons
    nml = context.nml
    rt_esc_frac = float(nml[nml.NML.RT_PARAMS]['rt_esc_frac'].replace('d', 'e'))
    return rt_esc_frac * nPhotons

@seren3.derived_quantity(requires=["mass", "Nion_per_Msun"], latex=r"$\dot{N_{\mathrm{ion}}}$ [#]")
def star_Nion(context, dset, env_name="RAMSES_SED_DIR", dt=0, direction=None, group=None):
    '''
    Return number of photons emitted per stellar population
    '''
    nPhotons = star_Nion_per_Msun(context, dset, dt=dt, direction=direction, group=group)

    idx = None
    if direction == 'b':
        idx = np.where((dset['age'] - dt) > 0)
    else:
        idx = np.where(dset['age'] > 0)

    mass = dset['mass'][idx] * context.info['unit_mass'].express(context.C.Msun)
    return nPhotons * mass

# bands_available = ['u', 'b', 'v', 'r', 'i', 'j', 'h', 'k', 'U', 'B', 'V', 'R', 'I',
#                    'J', 'H', 'K']

# for band in bands_available:
#     X = lambda context, dset, b=str(
#         band): luminosity.calc_mags(context, dset, band=b)
#     X.__name__ = "star_" + band + "_mag"
#     X.__doc__ = band + " magnitude from analysis.luminosity.calc_mags"""
#     seren3.add_derived_quantity(X, ["mass", "age", "metal"])
