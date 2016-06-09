import seren3
from .part_derived import *
import numpy as np

@seren3.derived_quantity(requires=["epoch"], latex=r'Age [Gyr]')
def gmc_age(context, dset, **kwargs):
    return part_age(context, dset, **kwargs)


@seren3.derived_quantity(requires=["age", "mass"], latex=r'sSFR [Gyr$^{-1}$]')
def gmc_sSNeR(context, dset, nbins=100, **kwargs):
    from seren3.exceptions import NoParticlesException
    from pymses.utils import constants as C
    '''
    Compute the number of SNe events per unit time
    '''

    age = dset['age']  # Already in Gyr
    mass = dset['mass'] * context.info['unit_mass'].express(C.Msun)

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing sSNeR", 'gmc_sSNeR')

    def sner(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = 1e-9 * nbins / (agerange[1] - agerange[0])
        # binnorm = nbins / (agerange[1] - agerange[0])
        weights = mass * binnorm

        snerhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, **kwargs)

        binmps = np.zeros(len(snerhist))
        binsize = np.zeros(len(snerhist))
        for i in np.arange(len(snerhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return snerhist, binmps, binsize

    snerhist, binmps, binsize = sner(age, mass, **kwargs)
    M_star = mass.sum()
    sSNeR = (snerhist * 1e9) / M_star  # Msun/yr -> Msun/Gyr -> /Gyr
    return {'sSNeR' : sSNeR, 'lookback-time' : binmps, 'binsize' : binsize}  # SNeR [Gyr^-1], Lookback Time [Gyr], binsize [Gyr]
