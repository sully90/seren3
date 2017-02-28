def test_sfr():
    '''
    Test SFR calculation is working correctly
    '''
    import seren3
    import numpy as np
    from scipy import integrate

    path = "/research/prace/david/aton/256/"
    sim = seren3.init(path)
    iout = sim.numbered_outputs[-1]  # last snapshot

    snap = sim[iout]
    snap_sfr, lbtime, bsize = sfr(snap)

    dset = snap.s["mass"].flatten()
    mstar_tot = dset["mass"].in_units("Msol").sum()
    integrated_mstar = integrate.trapz(snap_sfr, lbtime)

    assert(np.allclose(mstar_tot, integrated_mstar, rtol=1e-2)), "Error: Integrated stellar mass not close to actual."
    print "Passed"

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
        raise NoParticlesException("No particles found while computing SFR", 'analysis/stars/sfr')

    def compute_sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = SimArray(1e-9 * nbins / (agerange[1] - agerange[0]), "yr**-1")

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