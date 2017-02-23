'''
Routines for plotting star properties
'''
import numpy as np
import matplotlib.pylab as plt

def schmidtlaw(subsnap, filename=None, center=True, pretime='50 Myr', diskheight='3 kpc', rmax='20 kpc', compare=True, \
            radial=True, clear=True, legend=True, bins=10, **kwargs):
    '''
    Plots the schmidt law setting units correctly (i.e age).
    Follows pynbodys own routine.
    '''
    import pynbody
    from pynbody.analysis import profile

    s = subsnap.pynbody_snapshot()  # sets age property
    s.physical_units()

    if not radial:
        raise NotImplementedError("Sorry, only radial Schmidt law currently supported")

    if center:
        pynbody.analysis.angmom.faceon(s.s)  # faceon to stars

    if isinstance(pretime, str):
        from seren3.array import units
        pretime = units.Unit(pretime)

    # select stuff
    diskgas = s.gas[pynbody.filt.Disc(rmax, diskheight)]
    diskstars = s.star[pynbody.filt.Disc(rmax, diskheight)]
    tform = diskstars.s["age"] - diskstars.properties["time"]

    youngstars = np.where(diskstars["age"].in_units("Myr") <= pretime.in_units("Myr"))[0]

    # calculate surface densities
    if radial:
        ps = profile.Profile(diskstars[youngstars], nbins=bins)
        pg = profile.Profile(diskgas, nbins=bins)
    else:
        # make bins 2 kpc
        nbins = rmax * 2 / binsize
        pg, x, y = np.histogram2d(diskgas['x'], diskgas['y'], bins=nbins,
                                  weights=diskgas['mass'],
                                  range=[(-rmax, rmax), (-rmax, rmax)])
        ps, x, y = np.histogram2d(diskstars[youngstars]['x'],
                                  diskstars[youngstars]['y'],
                                  weights=diskstars['mass'],
                                  bins=nbins, range=[(-rmax, rmax), (-rmax, rmax)])

    if clear:
        plt.clf()

    print ps["density"]
    plt.loglog(pg['density'].in_units('Msol pc^-2'),
               ps['density'].in_units('Msol kpc^-2') / pretime / 1e6, "+",
               **kwargs)

    if compare:
        # Prevent 0 densitiy
        min_den = max(pg['density'].in_units('Msol pc^-2').min(), 1e-6)
        xsigma = np.logspace(min_den,
                             np.log10(
                                 pg['density'].in_units('Msol pc^-2')).max(),
                             100)
        ysigma = 2.5e-4 * xsigma ** 1.5        # Kennicutt (1998)
        xbigiel = np.logspace(1, 2, 10)
        ybigiel = 10. ** (-2.1) * xbigiel ** 1.0   # Bigiel et al (2007)
        plt.loglog(xsigma, ysigma, label='Kennicutt (1998)')
        plt.loglog(
            xbigiel, ybigiel, linestyle="dashed", label='Bigiel et al (2007)')

    plt.xlabel('$\Sigma_{gas}$ [M$_\odot$ pc$^{-2}$]')
    plt.ylabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    if legend:
        plt.legend(loc=2)
    if (filename):
        plt.savefig(filename)