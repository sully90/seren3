'''
Module for interpolating/calculating magnitues / luminosities.
Adapted from https://pynbody.github.io/pynbody/_modules/pynbody/analysis/luminosity.html to work with seren
'''

import numpy as np
import os
from seren3.analysis.interpolate import interpolate2d
from seren3 import config

_data_dir = config.get("data", "data_dir")


def bouwens_2015_schecter_fit(z):
    M_star = -20.97
    phi_star = 0.44 * 10 ** (-0.28 * (z - 6.)) * 10 ** -3.  # Mpc-3
    alpha = -1.87 - 0.1 * (z - 6.)
    return M_star, phi_star, alpha


def schecter(M, M_star, phi_star, alpha):
    return phi_star * (np.log(10) / 2.5) * 10 ** (-0.4 * (M - M_star) * (alpha + 1)) * np.exp(-10 ** (-0.4 * (M - M_star)))


def calc_luminosities_halos(snapshot, fesc=1., lambda_A=1500, pickle_path=None):
    '''
    Calculate luminosities of all halos
    Filter halos if desired
    '''
    import os

    if (pickle_path is None):
        pickle_path = "%s/pickle/ConsistentTrees/" % snapshot.path

    fname = "%s/luminosity_lambdaA_%s_database_%05i.p" % (pickle_path, int(lambda_A), snapshot.ioutput)
    print fname
    if (os.path.isfile(fname)):
        import pickle
        data = pickle.load( open(fname, "rb") )
        unit = data[data.keys()[0]]["L"].units
        lums = snapshot.array(np.zeros(len(data)), unit)
        # lums = np.zeros(len(data))

        keys = data.keys()

        for i in range(len(data)):
            key = keys[i]
            res = data[key]
            L = res["L"]
            age = res["age"].in_units("Myr")
            idx = np.where(age <= 10.)
            lums[i] = L[idx].sum()

        # L_sol = snapshot.array(3.828e26, "J s**-1")

        # lums /= L_sol        
        return lums[lums > 0].in_units("3.828e26 J s**-1")

    from seren3.utils.sed import io
    halos = snapshot.halos()
    SED = io.read_seds()
    kwargs = {"fesc" : fesc, "lambda_A" : lambda_A, "sed" : SED}
    lums = []
    for h in halos:
        if (len(h.s) > 0):
            dset = h.s["luminosity"].flatten(**kwargs)
            lums.append(np.sum(dset['luminosity']))

    return np.array(lums)


def luminosity_function(lums, z, boxsize, h, nbins=15, **kwargs):
    '''
    Plot the Luminosity function in absolute magnitudes
    lums - Luminosities in solar luminosities
    boxsize - Size of the box in Mpc/h
    Optional args:
        plot (bool) - Plot results, should also supply lambda_A
        bouwens (bool) - Plot Bouwens 2015 observations
        legend (bool) - Draw the legend
        show (bool) - Show the plot
        block (bool) - Block the interpreter while displaying plot
    '''
    gnedin_fac = 10 ** (5. - z)  # 10^(5-z) from Gnedin 2014

    #distance_cm = 10.0 * 3.08568025e18
    # zeropoint = -2.5 * np.log10(4.0 * np.pi *
    # distance_cm * distance_cm * 3631.0 * 1.0e-23)

    mags = -2.5 * np.log10(lums) + 5.8  # - zeropoint
    mags = np.delete(mags, np.where(mags == np.inf))
    mags = mags[~np.isnan(mags)]

    print mags.min()

    print len(mags)

    mhist, mbin_edges = np.histogram(mags, bins=nbins)
    mbinmps = np.zeros(len(mhist))
    mbinsize = np.zeros(len(mhist))
    for i in np.arange(len(mhist)):
        mbinmps[i] = np.mean([mbin_edges[i], mbin_edges[i + 1]])
        mbinsize[i] = mbin_edges[i + 1] - mbin_edges[i]

    y = gnedin_fac * mhist / (((boxsize / h) ** 3) * mbinsize)

    if kwargs.pop('plot', False):
        import matplotlib.pylab as plt
        color = kwargs.pop('color', 'k')

        ax = plt.subplot(111)

        ax.plot(mbinmps, y, color='b', label='RT2 z = %f' % z)

        if kwargs.pop('bouwens', True):
            # Plot Bouwens 2015 data
            M_star, phi_star, alpha = bouwens_2015_schecter_fit(z)
            from seren3.utils.sed import io
            data = io.read_bouwens_2015()
            avail_z = np.array(data.keys())
            neareat_z_idx = (np.abs(avail_z - z)).argmin()
            bouwens_z = avail_z[neareat_z_idx]
            data = data[bouwens_z]

            xdata = np.linspace(min(data['M1600']), max(mbinmps), 100)

            # Plot the points and Schecter function in the same units as above
            ax.plot(xdata, gnedin_fac * schecter(xdata, M_star, phi_star, alpha) / h**3.,
                    label='z = %1.2f Schecter Fit' % bouwens_z, color=color, linestyle='--')
            ax.errorbar(data['M1600'], gnedin_fac * np.array(data['phi']) / h**3., yerr=gnedin_fac * np.array(
                data['err']), label='z = %1.2f Bouwens 2015' % bouwens_z, color=color, fmt='o')

        ax.set_yscale('log')

        if kwargs.get('legend', False):
            plt.legend()

            # Reposition the legend below the plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

            # Put a legend below current axis
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fancybox=True, shadow=True, ncol=2, prop={'size': 10.})

        plt.xlabel(r'M$_{%d}$' % kwargs['lambda_A'])
        plt.ylabel(
            r'10$^{5-z}$ $\phi$(M$_{%d}$) / (Mpc$^{-3}$ mag$^{-1}$)' % kwargs['lambda_A'])

        if kwargs.get('show', False):
            plt.show(block=kwargs.get('block', True))

        return mbinmps, mhist, mbinsize


def calc_mags(snapshot, dset, band='v'):
    """Calculating visible magnitudes

    Using Padova Simple stellar populations (SSPs) from Girardi
    http://stev.oapd.inaf.it/cgi-bin/cmd
    Marigo+ (2008), Girardi+ (2010)
    """

    lumfile = os.path.join(_data_dir, "cmdlum.npz")
    if os.path.exists(lumfile):
        lums = np.load(lumfile)
    else:
        raise IOError("cmdlum.npz (magnitude table) not found")

    # Iterate over datasets
    # for dset in snapshot[('star', ['metal', 'age'])]:
    #dset = snapshot[('star', ['metal', 'age', 'mass'])].flatten()
    age_star = dset['age'] * 1e9  # yr
    metals = dset['metal']
    massform = dset['mass'] * snapshot.info['unit_mass'].express(C.Msun)

    # get values off grid to minmax
    age_star[np.where(age_star < np.min(lums['ages']))] = np.min(
        lums['ages'])
    age_star[np.where(age_star > np.max(lums['ages']))] = np.max(
        lums['ages'])
    metals[np.where(metals < np.min(lums['mets']))] = np.min(lums['mets'])
    metals[np.where(metals > np.max(lums['mets']))] = np.max(lums['mets'])

    age_grid = np.log10(lums['ages'])
    met_grid = lums['mets']
    mag_grid = lums[band]

    output_mags = interpolate2d(
        metals, np.log10(age_star), met_grid, age_grid, mag_grid)

    vals = output_mags - 2.5 * \
        np.log10(massform)
    return vals


def halo_mag(snapshot, halo, band='v'):
    """ Calculate halo magnitude

    Calls calc_mags for every star in passed
    in simulation, converts those magnitudes back to luminosities, adds
    those luminosities, then converts that luminosity back to magnitudes,
    which are returned.
    """
    dset = snapshot.filter(('star', '%s_mag' % band), halo).flatten()
    if ('%s_mag' % band) in dset and len(dset['%s_mag' % band]) > 0:
        return -2.5 * np.log10(np.sum(10.0 ** (-0.4 * dset['%s_mag' % band])))
    else:
        return np.nan


def halo_lum(snapshot, halo, band='v'):
    """Calculating halo luminosity

    Calls calc_mags for all stars in the halo, converts
    to luminosities and sums them
    """
    dset = snapshot.filter(('star', '%s_mag' % band), halo).flatten()
    if ('%s_mag' % band) in dset and len(dset['%s_mag' % band]) > 0:
        return np.sum(10.0 ** ((5.8 - dset['%s_mag' % band]) / 2.5))
    else:
        return np.nan