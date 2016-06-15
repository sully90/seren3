import seren3
from .part_derived import *
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["epoch"], unit=C.Gyr)
def star_age(context, dset, **kwargs):
    return part_age(context, dset, **kwargs)

@seren3.derived_quantity(requires=["age", "mass"], unit=1./C.Gyr)
def star_sSFR(context, dset, nbins=100, **kwargs):
    from seren3.exceptions import NoParticlesException

    age = dset["age"].in_units("Gyr")  # Gyr
    mass = dset["mass"].in_units("Msol")  # Msol

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'star_sSFR')

    def sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = SimArray(1e-9 * nbins / (agerange[1] - agerange[0]), "Gyr**-1")
        # binnorm = nbins / (agerange[1] - agerange[0])
        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return SimArray(sfrhist, "Msol Gyr**-1"), SimArray(binmps, "Gyr"), SimArray(binsize, "Gyr")

    sfrhist, binmps, binsize = sfr(age, mass, **kwargs)
    sfrhist.set_field_name("sSFR")
    binmps.set_field_name("Lookback-time")
    binsize.set_field_name("binsize")
    M_star = mass.sum()
    sSFR = (sfrhist * 1e9) / M_star  # Msun/yr -> Msun/Gyr
    return {'sSFR' : sSFR, 'lookback-time' : binmps, 'binsize' : binsize}  # sSFR [Gyr^-1], Lookback Time [Gyr], binsize [Gyr]