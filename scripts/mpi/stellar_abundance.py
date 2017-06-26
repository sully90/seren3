import numpy as np
import pickle

_MASS_UNIT = "Msol h**-1"

def unpack(path, iout):
    fname = '%s/abundance_%05i.p' % (path, iout)
    dest = pickle.load( open(fname, 'rb') )
    tot_mass = []
    star_mass = []
    for rank in dest:
        for item in dest[rank]:
            res = item.result
            tot_mass.append(res["tot_mass"])
            star_mass.append(res["star_mass"])

    return np.array(star_mass), np.array(tot_mass)

def plot(simulations, ioutputs, labels, colours, nbins=10, plot_baryon_fraction=False, compare=True, dm_particle_cutoff=100, star_particle_cutoff=1, pickle_paths=None):
    import pickle
    from pynbody.plot.stars import moster, behroozi
    import matplotlib.pylab as plt
    from seren3.analysis.plots import fit_scatter

    import matplotlib
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.labelsize'] = 18
    matplotlib.rcParams['ytick.labelsize'] = 18
    matplotlib.rcParams['axes.labelsize'] = 22

    if (pickle_paths is None):
        pickle_paths = ["%s/pickle/ConsistentTrees/" % sim.path for sim in simulations]

    fig, ax = plt.subplots(figsize=(8, 8))
    # fig, ax = plt.subplots()

    for sim, ioutput, label, c, ppath in zip(simulations, ioutputs, labels, colours, pickle_paths):
        snap = sim[ioutput]
        cosmo = snap.cosmo
        cosmic_mean = cosmo["omega_b_0"]/cosmo["omega_M_0"]

        data = None
        fname = "%s/abundance_%05i.p" % (ppath, snap.ioutput)
        with open(fname, 'rb') as f:
            data = pickle.load(f)

        nrecords = len(data)
        mass = np.zeros(nrecords); stellar_mass = np.zeros(nrecords)
        np_dm = np.zeros(nrecords); np_star = np.zeros(nrecords)

        for i in range(nrecords):
            res = data[i].result
            mass[i] = res["tot_mass"]
            stellar_mass[i] = res["star_mass"]
            np_dm[i] = res["np_dm"]
            np_star[i] = res["np_star"]
            del res

        # print np_star.min(), np_star.max()
        idx = np.where(np.logical_and( np_dm >= dm_particle_cutoff, np_star >= star_particle_cutoff ))
        mass = mass[idx]; stellar_mass = stellar_mass[idx]

        stellar_mass = (stellar_mass / mass) / cosmic_mean
        stellar_mass *= 100  # %

        bc, mean, std = fit_scatter(mass, stellar_mass, nbins=10)
        bc, mean, log_std, log_sterr = fit_scatter(np.log10(mass), np.log10(stellar_mass), ret_sterr=True, nbins=nbins)

        # std = (log_std/0.434) * 10**mean
        # sterr = (log_sterr/0.434) * 10**mean

        if compare:
            x = snap.array(np.logspace(np.log10(2.5e7), np.log10(1.5e10), 100), _MASS_UNIT)
            ystarmasses, errors = moster(x.in_units("Msol"), snap.z)
            ystarmasses = snap.array(ystarmasses, "Msol").in_units(_MASS_UNIT)

            ystarmasses = (ystarmasses / x) / cosmic_mean
            ystarmasses *= 100

            # print errors.min(), errors.max()
            moster_c = "#BBBBBB"
            ax.fill_between(x, np.log10(np.array(ystarmasses)/np.array(errors)),
                             y2=np.log10(np.array(ystarmasses)*np.array(errors)),
                             facecolor=moster_c,color=moster_c,label='Moster et al (2013)',alpha=0.5)
            ax.plot(x, np.log10(ystarmasses), color=moster_c)

            # behroozi_c = "#F08080"
            behroozi_c = "lightskyblue"
            ystarmasses, errors = behroozi(x.in_units("Msol"), snap.z)
            ystarmasses = snap.array(ystarmasses, "Msol").in_units(_MASS_UNIT)

            ystarmasses = (ystarmasses / x) / cosmic_mean
            ystarmasses *= 100

            ax.fill_between(x, np.log10(np.array(ystarmasses)/np.array(errors)),
                             y2=np.log10(np.array(ystarmasses)*np.array(errors)),
                             facecolor=behroozi_c,color=behroozi_c,label='Behroozi et al (2013)',alpha=0.5)
            ax.plot(x, np.log10(ystarmasses), color=behroozi_c)
            compare = False

        if plot_baryon_fraction:
            baryon_fraction = x*cosmic_mean
            y = ((0.1*baryon_fraction)/x)/cosmic_mean
            y *= 100.
            # ax.loglog(mass, baryon_fraction, linestyle='dotted', label=r'$\Omega_{\mathrm{b}} / \Omega_{\mathrm{M}}$')
            ax.plot(x, y, linestyle='dotted', linewidth=3., label=r'0.1 $\Omega_{\mathrm{b}} / \Omega_{\mathrm{M}}$')
            plot_baryon_fraction = False

        # ax.scatter(mass, stellar_mass, color=c, alpha=0.4, s=15)
        # ax.errorbar(10**bc, 10**mean, yerr=std, color=c, linewidth=3., linestyle="-", zorder=10, label=label)
        # ax.plot(10**bc, 10**mean, color=c, linewidth=3., linestyle="-", zorder=10, label=label)
        # ax.plot(10**bc, 10**(mean+std), color=c, linewidth=3., linestyle="--", zorder=10)
        # ax.plot(10**bc, 10**(mean-std), color=c, linewidth=2., linestyle="--", zorder=10)
        # ax.plot(10**bc, mean, color=c, linewidth=3., linestyle="-", zorder=10, label=label)
        # ax.plot(10**bc, mean + log_std, color=c, linewidth=1.5, linestyle="--", zorder=10)
        # ax.plot(10**bc, mean - log_std, color=c, linewidth=1.5, linestyle="--", zorder=10)
        # ax.errorbar(10**bc, mean, yerr=log_std, color=c, linewidth=3., linestyle="none", zorder=10, label=label)
        e = ax.errorbar(10**bc, mean, yerr=log_std, color=c, label=label,\
             fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-', linewidth=2.)

    text_pos = (4.5e7, np.log10(1))
    text = 'z = %1.2f'% snap.z

    ax.set_xlabel(r'M$_{\mathrm{h}}$ [M$_{\odot}$/h]')
    # ax.set_ylabel(r'M$_{*}$ [M$_{\odot}$/h]')
    ax.set_ylabel(r'log$_{10}$(M$_{*}$/M$_{\mathrm{h}}$)/($\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$) [%]')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    # # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
    #           fancybox=True, shadow=False, ncol=5, prop={'size':10.5})
    ax.legend(prop={'size':15.}, loc="lower right")

    ax.set_xscale("log")#; ax.set_yscale("log")
    ax.text(text_pos[0], text_pos[1], text, color="k", size=18)

    ax.set_xlim(x.min(), x.max())

    plt.tight_layout()

    plt.savefig("/home/ds381/RTX_HD_stellar_abundance.pdf", format="pdf")
    plt.show()


def stars(dset):
    return np.where( np.logical_and(dset["epoch"] != 0, dset["id"] > 0.) )
    

def main(path, iout, finder='ctrees', pickle_path=None):
    import seren3
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading data")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)  # disable multiprocessing

    halos = snap.halos(finder=finder)
    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    mpi.msg("Starting worker loop")
    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        if len(h.s) > 0:

            mpi.msg("Working on halo %i \t %i" % (i, h.hid))

            part_dset = h.p[["mass", "epoch", "id", "age"]].flatten()
            gas_dset = h.g["mass"].flatten()
            gas_mass = gas_dset["mass"].in_units(_MASS_UNIT)

            star_idx = stars(part_dset)
            part_mass = part_dset["mass"].in_units(_MASS_UNIT)

            idx_young_stars = np.where(part_dset["age"][star_idx].in_units("Myr") <= 10.)

            tot_mass = part_mass.sum() + gas_mass.sum()
            star_mass = part_mass[star_idx][idx_young_stars].sum()

            if (star_mass > 0):

                np_star = len(part_dset["mass"][star_idx][idx_young_stars])
                np_dm = len(part_dset["mass"]) - np_star

                mpi.msg("%e \t %e \t %e" % (tot_mass, star_mass, star_mass/tot_mass))

                sto.idx = h["id"]
                sto.result = {"tot_mass" : tot_mass, "star_mass" : star_mass, "np_dm" : np_dm, "np_star" : np_star}

    if mpi.host:
        import os
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/abundance_%05i.p" % (pickle_path, iout)
        with open(fname, 'wb') as f:
            pickle.dump( mpi.unpack(dest), f )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    finder = sys.argv[3]
    pickle_path = None
    if len(sys.argv) > 4:
        pickle_path = sys.argv[4]

    main(path, iout, finder, pickle_path)