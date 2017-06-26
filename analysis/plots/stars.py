'''
Routines for plotting star properties
'''
import numpy as np
import matplotlib.pylab as plt


def plot_sfr(context, ax=None, label=None, **kwargs):
    '''
    Plots the star formation rate
    '''
    from seren3.analysis import stars
    from seren3.array import units

    if ax is None:
        ax = plt.gca()

    sfr_unit = None
    if "sfr_unit" in kwargs:
        sfr_unit = kwargs.pop("sfr_unit")

    sfr, lbtime, bsize = stars.sfr(context, **kwargs)
    if (sfr_unit is not None):
        sfr.convert_units(sfr_unit)

    ax.step(lbtime, sfr, linewidth=2., label=label)
    ax.set_yscale("log")

    unit = sfr.units
    dims = unit.dimensional_project([units.Unit("kg"), units.Unit("s")])

    field_name = None
    if dims[0].numerator == 1:
        # SFR [Msol / Gyr]
        field_name = "SFR"
    elif dims[0].numerator == 0:
        # sSFR [Gyr^-1]
        field_name = "sSFR"
    else:
        raise Exception("Cannot understand SFR dims: %s" % dims)
    ax.set_ylabel(r"log$_{10}$ %s [$%s$]" % (field_name, unit.latex()))
    ax.set_xlabel(r"Lookback-time [$%s$]" % lbtime.units.latex())

    #plt.show()


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

def plot_dSFR_mvir(sims, iouts, labels, cols, pickle_paths=None, **kwargs):
    '''
    Plot star formation rate density against halo virial mass
    '''
    import pickle
    import matplotlib.pylab as plt
    from seren3.analysis.plots import fit_scatter

    dSFR_units = kwargs.pop("units", "Msol yr**-1 kpc**-3")
    nbins = kwargs.pop("nbins", 25)
    legend_size = kwargs.pop("lsize", 18)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    def get_fname_halos(ppath, iout):
        return "%s/dSFR_halo_av_%05i.p" % (ppath, iout)

    def get_fname(ppath):
        return "%s/dSFR_time_averaged.p" % (ppath)

    ax=axs[0]

    if (pickle_paths is None):
        pickle_paths = ["%s/pickle/" % sim.path for sim in sims]

    for sim, iout, ppath, label, c in zip(sims, iouts, pickle_paths, labels, cols):
        fname = get_fname_halos(ppath, iout)
        data = pickle.load(open(fname, "rb"))

        snap = sim[iout]
        halos = snap.halos()

        part_mass = snap.quantities.particle_mass()

        mvir_dict = {}
        for h in halos:
            npdm = h["Mvir"] / part_mass
            mvir_dict[int(h["id"])] = {"Mvir" : h["Mvir"], "np_dm" : npdm}

        nrecrods = len(data)
        dSFR_mw = np.zeros(nrecrods)
        mvir = np.zeros(nrecrods)
        np_dm = np.zeros(nrecrods)

        for i in range(nrecrods):
            res = data[i].result
            dSFR_mw[i] = res["mw"].in_units(dSFR_units)
            mvir[i] = mvir_dict[int(data[i].idx)]["Mvir"]
            np_dm[i] = mvir_dict[int(data[i].idx)]["np_dm"]

        idx = np.where(np.logical_and(np_dm >= 50, ~np.isnan(dSFR_mw)))
        mvir = mvir[idx]; dSFR_mw = dSFR_mw[idx]

        x = np.log10(mvir)
        y = np.log10(dSFR_mw)

        # Plot
        # plt.scatter(10**x, y, color=c, **kwargs)

        bc, mean, std, sterr = fit_scatter(x, y, ret_sterr=True, nbins=nbins)
        ax.errorbar(10**bc, mean, yerr=sterr, color=c, linewidth=2., linestyle="-", label=label)

    ax.legend(loc="lower right", prop={'size':legend_size})
    # plt.yscale("log")
    ax.set_xscale("log")

    ax.set_xlabel(r"M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
    ax.set_ylabel(r"log$_{10}$ $\langle \rho_{\mathrm{SFR}} \rangle_{M}$ [M$_{\odot}$ yr$^{-1}$ kpc$^{-3}$]")

    # plt.xlim(1e7, 2e10)
    # plt.ylim(5e-9, 5e-1)
    # ax.set_ylim(-8, -2.5)
    ax.set_ylim(-8, y.max())
    # ax.set_title("z = %1.2f" % snap.z)

    text_pos = (2e7, 1)
    text = "z = %1.2f" % snap.z
    ax.text(text_pos[0], text_pos[1], text, color="k", size="x-large")

    # Time evol.
    ax = axs[1]

    min_v = np.inf
    max_v = -np.inf
    for sim, ppath, label, c in zip(sims, pickle_paths, labels, cols):
        fname = get_fname(ppath)
        data = pickle.load(open(fname, "rb"))

        nrecords = len(data)
        z = np.zeros(nrecords)
        vw = np.zeros(nrecords)
        mw = np.zeros(nrecords)

        for i in range(nrecords):
            res = data[i].result

            if (res["z"] < 6):
                break

            z[i] = res["z"]
            vw[i] = res["vw"].in_units(dSFR_units)
            mw[i] = res["mw"].in_units(dSFR_units)

        x = z
        y = np.log10(mw)
        # ax.plot(z, log_vw, linewidth=2., color=c, label=label, linestyle="-")
        # ax.plot(z, log_mw, linewidth=2., color=c, label=label, linestyle="-")

        bc, mean, std, sterr = fit_scatter(x, y, ret_sterr=True, nbins=nbins+2)
        ax.errorbar(bc, mean, yerr=std, color=c, linewidth=2., linestyle="-", label=label)

    ax.legend(loc="lower left", prop={'size':legend_size})

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"log$_{10}$ $\langle \rho_{\mathrm{SFR}} \rangle_{M}$ [M$_{\odot}$ yr$^{-1}$ kpc$^{-3}$]")
    ax.set_ylim(-7., 0.)

    fig.tight_layout(w_pad=1.)

    plt.show()
