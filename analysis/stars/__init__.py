def sfr(context, ret_sSFR=False, nbins=100, **kwargs):
    '''
    Compute the (specific) star formation rate within this context.
    '''
    import numpy as np
    from seren3.array import SimArray
    from seren3.exceptions import NoParticlesException

    dset = context.s[["age", "mass"]].flatten()
    age = dset["age"].in_units("Gyr")
    mass = dset["mass"].in_units("Msol")

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'star_sSFR')

    def compute_sfr(age, mass, nbins=nbins, **kwargs):
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

    sfrhist, lookback_time, binsize = compute_sfr(age, mass, **kwargs)
    SFR = sfrhist.in_units("Msol Gyr**-1")

    if ret_sSFR:
        SFR /= mass.sum()  # specific star formation rate

    SFR.set_field_latex("$\\mathrm{SFR}$")
    lookback_time.set_field_latex("$\\mathrm{Lookback-Time}$")
    binsize.set_field_latex("$\Delta$")
    return SFR, lookback_time, binsize  # SFR [Msol Gyr^-1] (sSFR [Gyr^-1]), Lookback Time [Gyr], binsize [Gyr]