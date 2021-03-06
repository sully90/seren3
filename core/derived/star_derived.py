import seren3
from .part_derived import *
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["epoch"])
def star_age(context, dset, **kwargs):
    return part_age(context, dset, **kwargs)


@seren3.derived_quantity(requires=["age", "metal"])
def star_Nion_d(context, dset, dt=0., group=1):
    '''
    Computes the number of ionisiing photons produced by a stellar population per solar mass per second
    '''
    from seren3.array import SimArray
    from seren3.utils.sed import io
    from seren3.exceptions import NoParticlesException
    from seren3 import config
    # from seren3.analysis import interpolate
    from scipy.interpolate import interp2d

    verbose = config.get("general", "verbose")

    Z_sun = 0.02  # metallicity of the sun
    nGroups = context.info["nGroups"]
    
    if(verbose): print 'Computing Nion_d for photon group %i/%i' % (group, nGroups)
    nIons = context.info["nIons"]
    nPhotons_idx = 0  # index of photon number in SED

    # Load the SED table
    agebins, zbins, SEDs = io.read_seds_from_lists(context.path, nGroups, nIons)
    igroup = group - 1
    fn = interp2d(zbins/Z_sun, agebins, SEDs[:,:,igroup,nPhotons_idx])

    age = dset["age"].in_units("Gyr")
    Z = dset["metal"] /  Z_sun  # in units of solar metalicity
    # Which star particles should we keep
    if dt != 0.:
        age -= dt.in_units("Gyr")
        # keep = np.where( np.logical_and(age >= 0., age.in_units("Myr") <= 10.) )
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
    # nPhotons = interpolate.interpolate2d(age, Z, agebins, zbins, SEDs[:,:,igroup,nPhotons_idx])

    # Multiply by (SSP) escape fraction and return
    nml = context.nml
    NML_KEYS = nml.NML

    rt_esc_frac = float(nml[NML_KEYS.RT_PARAMS]['rt_esc_frac'].replace('d', 'e'))
    Nion_d = SimArray(rt_esc_frac * nPhotons, "s**-1 Msol**-1")
    Nion_d.set_field_latex("$\\dot{N_{\\mathrm{ion}}}$")
    return Nion_d

@seren3.derived_quantity(requires=["mass", "metal", "age"])
def star_luminosity(context, dset, lambda_A=1600., **kwargs):
    '''
    Return the UV luminosity of stars using bc03 SED tables
    lambda_A - Wavelength in Angstroms
    '''
    from seren3.utils.sed import io
    from pymses.utils import constants as C

    lambda_A = SimArray(lambda_A, "angst")

    agebins, zbins, Ls, SEDs = (None, None, None, None)
    if "sed" in kwargs:
        agebins, zbins, Ls, SEDs = kwargs.pop("sed")
    else:
        agebins, zbins, Ls, SEDs = io.read_seds()
    nLs = len(Ls)
    # U = 1500  # Median UV SDSS band in Angstroms
    age = dset['age'].in_units("yr")

    star_mass = dset['mass'].in_units("Msol")
    Z_sun = 0.02  # metallicity of the sun
    star_metal = dset["metal"] / Z_sun

    L_sol = SimArray(3.828e26, "J s**-1")

    # Interpolate Luminosities from the grid
    from seren3.analysis.interpolate import interpolate3d
    U = SimArray(np.zeros(len(age)), "angst")
    for i in range(len(age)):
        U[i] = lambda_A

    star_lums_per_Msol = SimArray(interpolate3d(U, age, star_metal, Ls, agebins, zbins/Z_sun, SEDs), "3.828e26 J Msol**-1 s**-1 angst**-1")
    # star_lums_per_Msol = interpolate3d(U, star_metal, age, Ls, zbins, agebins, SEDs)
    star_lums = star_lums_per_Msol * star_mass  # Lsol per Angstrom
    return star_lums * lambda_A  # L / \AA -> L
