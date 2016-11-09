import seren3
from .part_derived import *
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["epoch"], unit=C.Gyr)
def star_age(context, dset, **kwargs):
    return part_age(context, dset, **kwargs)


@seren3.derived_quantity(requires=["age", "metal"], unit=C.s**-1 * C.Msun**-1)
def star_Nion_d(context, dset, dt=0., group=1):
    '''
    Computes the number of ionisiing photons produced by a stellar population per solar mass per second
    '''
    from seren3.array import SimArray
    from seren3.utils.sed import io
    from seren3.exceptions import NoParticlesException
    from seren3 import config
    from scipy.interpolate import interp2d

    verbose = config.get("general", "verbose")

    Z_sun = 0.02  # metallicity of the sun
    nGroups = context.info_rt["nGroups"]
    
    if(verbose): print 'Computing Nion_d for photon group %i/%i' % (group, nGroups)
    nIons = context.info_rt["nIons"]
    nPhotons_idx = 0  # index of photon number in SED

    # Load the SED table
    agebins, zbins, SEDs = io.read_seds_from_lists(context.path, nGroups, nIons)
    igroup = group - 1
    fn = interp2d(zbins, agebins, SEDs[:,:,igroup,nPhotons_idx])

    age = dset["age"].in_units("yr")
    Z = dset["metal"] /  Z_sun  # in units of solar metalicity
    # Which star particles should we keep
    if dt != 0.:
        age -= dt
        keep = np.where( age >= 0. )
        age = age[keep]
        Z = Z[keep]

    if len(age) == 0:
        raise NoParticlesException("No particles with (age - dt) > 0", "star_Nion_d")

    # interpolate photon production rate from SED
    nStars = len(age)
    nPhotons = np.zeros(nStars)
    for i in xrange(nStars):
        nPhotons[i] = fn(Z[i], age[i])

    # Multiply by (SSP) escape fraction and return
    nml = context.nml
    rt_esc_frac = float(nml[nml.NML.RT_PARAMS]['rt_esc_frac'].replace('d', 'e'))
    Nion_d = SimArray(rt_esc_frac * nPhotons, "s**-1 Msol**-1")
    Nion_d.set_latex("$\\dot{N_{\\mathrm{ion}}}$")
    return Nion_d

@seren3.derived_quantity(requires=["age", "mass"], unit=1./C.Gyr)
def star_sSFR(context, dset, nbins=100, **kwargs):
    from seren3.exceptions import NoParticlesException

    age = dset["age"].in_units("Gyr")  # Gyr
    mass = dset["mass"].in_units("Msol")  # Msol

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'star_sSFR')

    def sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = SimArray(1e-9 * nbins / (agerange[1] - agerange[0]), "yr**-1")
        # binnorm = nbins / (agerange[1] - agerange[0])
        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        # return SimArray(sfrhist, "Msol Gyr**-1"), SimArray(binmps, "Gyr"), SimArray(binsize, "Gyr")
        return SimArray(sfrhist, "Msol yr**-1"), SimArray(binmps, "Gyr"), SimArray(binsize, "Gyr")

    sfrhist, binmps, binsize = sfr(age, mass, **kwargs)
    M_star = mass.sum()
    # sSFR = (sfrhist * 1e9) / M_star  # Msun/yr -> Msun/Gyr
    sSFR = sfrhist.in_units("Msol Gyr**-1") / M_star  # Msun/yr -> Msun/Gyr -> Gyr^-1

    sSFR.set_latex("$\\mathrm{sSFR}$")
    binmps.set_latex("$\\mathrm{Lookback-Time}$")
    binsize.set_latex("$\Delta$")

    return {'sSFR' : sSFR, 'lookback-time' : binmps, 'binsize' : binsize}  # sSFR [Gyr^-1], Lookback Time [Gyr], binsize [Gyr]

@seren3.derived_quantity(requires=["age", "mass"], unit=C.Msun/C.Gyr)
def star_SFR(context, dset, nbins=100, **kwargs):
    from seren3.exceptions import NoParticlesException

    age = dset["age"].in_units("Gyr")  # Gyr
    mass = dset["mass"].in_units("Msol")  # Msol

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'star_sSFR')

    def sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = SimArray(1e-9 * nbins / (agerange[1] - agerange[0]), "yr**-1")
        # binnorm = nbins / (agerange[1] - agerange[0])
        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return SimArray(sfrhist, "Msol yr**-1"), SimArray(binmps, "Gyr"), SimArray(binsize, "Gyr")

    sfrhist, binmps, binsize = sfr(age, mass, **kwargs)
    SFR = sfrhist.in_units("Msol Gyr**-1")

    SFR.set_latex("$\\mathrm{SFR}$")
    binmps.set_latex("$\\mathrm{Lookback-Time}$")
    binsize.set_latex("$\Delta$")
    return {'SFR' : SFR, 'lookback-time' : binmps, 'binsize' : binsize}  # SFR [Msol Gyr^-1], Lookback Time [Gyr], binsize [Gyr]

