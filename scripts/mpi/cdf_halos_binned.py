import numpy as np
import seren3

# _FIELD = "nHI"
_FIELD = "T2_minus_Tpoly"

def test2(sim_labels, the_mass_bins=[7., 8., 9., 10.], **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.array import SimArray
    from pymses.utils import constants as C
    import matplotlib.gridspec as gridspec

    import seren3

    sim1 = seren3.load("RT2_nohm"); sim2 = seren3.load("RAMSES"); snap1 = sim1[106]; snap2 = sim2[93]

    info = snap1.info

    cols = None
    if "cols" not in kwargs:
        from seren3.utils import plot_utils
        cols = plot_utils.ncols(len(the_mass_bins), cmap=kwargs.pop("cmap", "Set1"))[::-1]
        # cols = plot_utils.ncols(2, cmap=kwargs.pop("cmap", "Set1"))[::-1]
    else:
        cols = kwargs.pop("cols")

    z = (1. / info["aexp"]) - 1.

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(4,9,wspace=0.,hspace=0.)

    ax1 = fig.add_subplot(gs[1:-1,:4])
    ax2 = fig.add_subplot(gs[:1,:4], sharex=ax1)

    axs = np.array([ax1, ax2])

    nbins=7

    ax = ax1

    binned_cdf1 = plot(snap1.path, snap1.ioutput, "%s/pickle/" % snap1.path, "nHI", ax=None, nbins=nbins)
    binned_cdf2 = plot(snap2.path, snap2.ioutput, "%s/pickle/" % snap2.path, "nHI", ax=None, nbins=nbins)

    text_pos = (7.2, 0.1)
    text = 'z = %1.2f'% z
    # ax.text(text_pos[0], text_pos[1], text, color="k", size=18)

    # for bcdf, ls in zip([binned_cdf1, binned_cdf2], ["-", "--"]):
    #     label = "All"
    #     x,y,std,stderr,n = bcdf["all"]
    #     e = ax.errorbar(x, y, yerr=std, color="k", label=label,\
    #         fmt="o", markerfacecolor="w", mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    for bcdf, ls in zip([binned_cdf1, binned_cdf2], ["-", "--"]):
        for ibin, c in zip(range(len(the_mass_bins)), cols):
        # for ibin, c in zip([0, len(the_mass_bins)-1], cols):
            x,y,std,stderr,n = bcdf[ibin]

            if ibin == len(the_mass_bins) - 1:
                upper = r"$\infty$"
            else:
                upper = "%i" % the_mass_bins[ibin+1]
            lower = "%i" % the_mass_bins[ibin]

            label = "[%s, %s)" % (lower, upper)

            if (ls == "--"):
                label = None

            upper_err = []
            lower_err = []
            for i in range(len(y)):
                ue = 0.
                le = 0.
                if (y[i] + std[i] > 1):
                    ue = 1. - y[i]
                else:
                    ue = std[i]

                if (y[i] - std[i] < 0):
                    le = y[i]
                else:
                    le = std[i]

                upper_err.append(ue)
                lower_err.append(le)

            yerr = [lower_err, upper_err]

            e = ax.errorbar(x, y, yerr=yerr, color=c, label=label,\
                fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    for sl, ls in zip(sim_labels, ["-", "--"]):
        ax.plot([2, 4], [-100, -100], linewidth=2., linestyle=ls, color="k", label=sl)

    # ax.legend(prop={"size":18}, loc="lower right")
    ax.set_ylim(-0.05, 1.2)

    ax = ax2

    linestyles = ["-", "--", "-.", ":"]

    for ibin, c, ls in zip(range(len(the_mass_bins)), cols, linestyles):
    # for ibin, c, ls in zip([0, len(the_mass_bins)-1], cols, linestyles):
        x1,y1,std1,stderr1,n1 = binned_cdf1[ibin]
        x2,y2,std2,stderr2,n2 = binned_cdf2[ibin]

        # ydiv = y1/y2
        ydiv = y1 - y2

        # error_prop = np.abs(ydiv) * np.sqrt( (std1/y1)**2 + (std2/y2)**2 )
        error_prop = np.sqrt( (std1)**2 + (std2)**2 )
        print ibin, error_prop

        e = ax.errorbar(x1, ydiv, yerr=error_prop, color=c,\
            fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    axs[0].set_xlabel(r"log$_{10}$ n$_{\mathrm{HI}}$ [m$^{-3}$]")
    # axs[0].set_xlabel(r"log$_{10}$ T [K]")
    axs[0].set_ylabel(r"CDF")
    axs[1].set_ylabel(r"$\Delta$ CDF")

    plt.setp(ax2.get_xticklabels(), visible=False)

    # ax = axs[0]
    # leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
    #   fancybox=True, shadow=False, ncol=3, prop={"size":18})
    # leg.set_title(r"log$_{10}$(Mvir) [M$_{\odot}$/h]", prop = {'size':'x-large'})

    n_star = SimArray(info["n_star"].express(C.H_cc), "cm**-3").in_units("m**-3")

    for axi in axs.flatten():
        axi.vlines(np.log10(n_star), -1, 2, color='k', linestyle='-.')
        # axi.vlines(np.log10(2e4), -5, 5, color='k', linestyle='-.')

    axs[1].set_ylim(-0.2, 1.3)
    # axs[1].set_ylim(0.2, -1.3)

    ##################################################################################

    ax1 = fig.add_subplot(gs[1:-1,5:])
    ax2 = fig.add_subplot(gs[:1,5:], sharex=ax1)

    axs = np.array([ax1, ax2])

    ax = ax1

    binned_cdf1 = plot(snap1.path, snap1.ioutput, "%s/pickle/" % snap1.path, "T2_minus_Tpoly", ax=None, nbins=nbins)
    binned_cdf2 = plot(snap2.path, snap2.ioutput, "%s/pickle/" % snap2.path, "T2_minus_Tpoly", ax=None, nbins=nbins)

    text_pos = (7.2, 0.1)
    text = 'z = %1.2f'% z
    # ax.text(text_pos[0], text_pos[1], text, color="k", size=18)

    # for bcdf, ls in zip([binned_cdf1, binned_cdf2], ["-", "--"]):
    #     label = "All"
    #     x,y,std,stderr,n = bcdf["all"]
    #     e = ax.errorbar(x, y, yerr=std, color="k", label=label,\
    #         fmt="o", markerfacecolor="w", mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    for bcdf, ls in zip([binned_cdf1, binned_cdf2], ["-", "--"]):
        for ibin, c in zip(range(len(the_mass_bins)), cols):
        # for ibin, c in zip([0, len(the_mass_bins)-1], cols):
            x,y,std,stderr,n = bcdf[ibin]

            if ibin == len(the_mass_bins) - 1:
                upper = r"$\infty$"
            else:
                upper = "%i" % the_mass_bins[ibin+1]
            lower = "%i" % the_mass_bins[ibin]

            label = "[%s, %s)" % (lower, upper)

            if (ls == "--"):
                label = None

            upper_err = []
            lower_err = []
            for i in range(len(y)):
                ue = 0.
                le = 0.
                if (y[i] + std[i] > 1):
                    ue = 1. - y[i]
                else:
                    ue = std[i]

                if (y[i] - std[i] < 0):
                    le = y[i]
                else:
                    le = std[i]

                upper_err.append(ue)
                lower_err.append(le)

            yerr = [lower_err, upper_err]

            e = ax.errorbar(x, y, yerr=yerr, color=c, label=label,\
                fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    for sl, ls in zip(sim_labels, ["-", "--"]):
        ax.plot([2, 4], [-100, -100], linewidth=2., linestyle=ls, color="k", label=sl)

    # ax.legend(prop={"size":18}, loc="lower right")
    ax.set_ylim(-0.05, 1.2)

    ax = ax2

    linestyles = ["-", "--", "-.", ":"]

    for ibin, c, ls in zip(range(len(the_mass_bins)), cols, linestyles):
    # for ibin, c, ls in zip([0, len(the_mass_bins)-1], cols, linestyles):
        x1,y1,std1,stderr1,n1 = binned_cdf1[ibin]
        x2,y2,std2,stderr2,n2 = binned_cdf2[ibin]

        # ydiv = y1/y2
        ydiv = y1 - y2

        # error_prop = np.abs(ydiv) * np.sqrt( (std1/y1)**2 + (std2/y2)**2 )
        error_prop = np.sqrt( (std1)**2 + (std2)**2 )
        print ibin, error_prop

        e = ax.errorbar(x1, ydiv, yerr=error_prop, color=c,\
            fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    # axs[0].set_xlabel(r"log$_{10}$ n$_{\mathrm{HI}}$ [m$^{-3}$]")
    axs[0].set_xlabel(r"log$_{10}$ T - T$_{\mathrm{J}}$ [K/$\mu$]")
    # axs[0].set_ylabel(r"CDF")
    # axs[1].set_ylabel(r"$\Delta$ CDF")

    plt.setp(ax2.get_xticklabels(), visible=False)

    ax = axs[0]
    leg = ax.legend(loc='upper center', bbox_to_anchor=(-0.15, -0.2),
      fancybox=True, shadow=False, ncol=3, prop={"size":18})
    leg.set_title(r"log$_{10}$(Mvir) [M$_{\odot}$/h]  :  z = %1.2f" % snap1.z, prop = {'size':'x-large'})

    n_star = SimArray(info["n_star"].express(C.H_cc), "cm**-3").in_units("m**-3")

    for axi in axs.flatten():
        # axi.vlines(np.log10(n_star), -1, 2, color='k', linestyle='-.')
        axi.vlines(np.log10(2e4), -5, 5, color='k', linestyle='-.')

    # axs[1].set_ylim(-0.2, 1.3)
    axs[1].set_ylim(0.2, -1.3)

    # plt.yscale("log")
    # plt.tight_layout()
    plt.savefig("/home/ds381/rt2_hd_cdf_nHI_T_z6_15.pdf", format="pdf")
    plt.show()

def test(binned_cdf1, binned_cdf2, info, sim_labels, field, the_mass_bins=[7., 8., 9., 10.], **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.array import SimArray
    from pymses.utils import constants as C
    import matplotlib.gridspec as gridspec

    assert len(binned_cdf1) == len(binned_cdf2)

    cols = None
    if "cols" not in kwargs:
        from seren3.utils import plot_utils
        cols = plot_utils.ncols(len(the_mass_bins), cmap=kwargs.pop("cmap", "Set1"))[::-1]
        # cols = ["r", "b", "darkorange", "k"]
    else:
        cols = kwargs.pop("cols")

    z = (1. / info["aexp"]) - 1.

    fig = plt.figure(figsize=(7,10))
    gs = gridspec.GridSpec(4,4,wspace=0.,hspace=0.)

    ax1 = fig.add_subplot(gs[1:-1,:])
    ax2 = fig.add_subplot(gs[:1,:], sharex=ax1)

    axs = np.array([ax1, ax2])

    ax = ax1

    text_pos = (7.2, 0.1)
    text = 'z = %1.2f'% z
    ax.text(text_pos[0], text_pos[1], text, color="k", size=18)

    for bcdf, ls in zip([binned_cdf1, binned_cdf2], ["-", "--"]):
        for ibin, c in zip(range(len(the_mass_bins)), cols):
            x,y,std,stderr,n = bcdf[ibin]

            if ibin == len(the_mass_bins) - 1:
                upper = r"$\infty$"
            else:
                upper = "%i" % the_mass_bins[ibin+1]
            lower = "%i" % the_mass_bins[ibin]

            label = "[%s, %s)" % (lower, upper)

            if (ls == "--"):
                label = None

            e = ax.errorbar(x, y, yerr=std, color=c, label=label,\
                fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    for sl, ls in zip(sim_labels, ["-", "--"]):
        ax.plot([2, 4], [-100, -100], linewidth=2., linestyle=ls, color="k", label=sl)

    # ax.legend(prop={"size":18}, loc="lower right")
    ax.set_ylim(-0.05, 1.2)

    ax = ax2

    linestyles = ["-", "--", "-.", ":"]

    for ibin, c, ls in zip(range(len(the_mass_bins)), cols, linestyles):
    # for ibin in range(len(the_mass_bins)):
        x1,y1,std1,stderr1,n1 = binned_cdf1[ibin]
        x2,y2,std2,stderr2,n2 = binned_cdf2[ibin]

        # ydiv = y1/y2
        ydiv = y1 - y2

        # error_prop = np.abs(ydiv) * np.sqrt( (std1/y1)**2 + (std2/y2)**2 )
        error_prop = np.sqrt( (std1)**2 + (std2)**2 )
        print ibin, error_prop

        e = ax.errorbar(x1, ydiv, yerr=error_prop, color=c,\
            fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=2.)

    axs[0].set_xlabel(r"log$_{10}$ n$_{\mathrm{HI}}$ [m$^{-3}$]")
    # axs[0].set_xlabel(r"log$_{10}$ T [K]")
    axs[0].set_ylabel(r"CDF")
    axs[1].set_ylabel(r"$\Delta$ CDF")

    plt.setp(ax2.get_xticklabels(), visible=False)

    ax = axs[0]
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
      fancybox=True, shadow=False, ncol=3, prop={"size":18})
    leg.set_title(r"log$_{10}$(Mvir) [M$_{\odot}$/h]", prop = {'size':'x-large'})

    n_star = SimArray(info["n_star"].express(C.H_cc), "cm**-3").in_units("m**-3")

    for axi in axs.flatten():
        axi.vlines(np.log10(n_star), -1, 2, color='k', linestyle='-.')
        # axi.vlines(np.log10(2e4), -5, 5, color='k', linestyle='-.')

    axs[1].set_ylim(-0.2, 1.3)
    # axs[1].set_ylim(0.2, -1.3)

    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig("/home/ds381/rt2_hd_cdf_%s_z6_15.pdf" % field, format="pdf")
    plt.show()


def plot(path, iout, pickle_path, field, the_mass_bins=[7., 8., 9., 10.], ax=None, **kwargs):
    import pickle
    from seren3.analysis.plots import fit_scatter
    from seren3.utils import flatten_nested_array
    import matplotlib.pylab as plt

    snap = seren3.load_snapshot(path, iout)
    fname = "%s/cdf_%s_halos_%05i.p" % (pickle_path, field, iout)
    data = pickle.load(open(fname, 'rb'))

    cdf_halos = []; bin_centre_halos = []; Mvir = []

    for i in range(len(data)):
        res = data[i].result
        cdf_halos.append(res["C"])
        bin_centre_halos.append(res["bc"])
        Mvir.append(res["Mvir"])

    cdf_halos = np.array(cdf_halos); bin_centre_halos = np.array(bin_centre_halos); Mvir = np.array(Mvir)

    idx = np.where(np.log10(Mvir) >= 6.5)
    cdf_halos = cdf_halos[idx]; bin_centre_halos = bin_centre_halos[idx]; Mvir = Mvir[idx]

    # Bin idx
    mass_bins = np.digitize(np.log10(Mvir), the_mass_bins, right=True)
    binned_cdf = {}

    nbins = kwargs.pop("nbins", 5)
    for i in range(len(the_mass_bins)+1):
        if (i == len(the_mass_bins)):
            x, y = ( flatten_nested_array(bin_centre_halos), flatten_nested_array(cdf_halos) )
            idx = np.where(~np.isnan(y))
            binned_cdf["all"] = fit_scatter(x[idx], y[idx], ret_sterr=True, ret_n=True, nbins=nbins)
        else:
            idx = np.where( mass_bins == i )
            x, y = ( flatten_nested_array(bin_centre_halos[idx]), flatten_nested_array(cdf_halos[idx]) )
            idx = np.where(~np.isnan(y))
            binned_cdf[i] = fit_scatter(x[idx], y[idx], ret_sterr=True, ret_n=True, nbins=nbins)

    binned_cdf['the_mass_bins'] = the_mass_bins

    if ax is None:
        return binned_cdf
        # ax = plt.gca()

    cols = None
    if "cols" not in kwargs:
        from seren3.utils import plot_utils
        # cols = plot_utils.ncols(len(the_mass_bins)+1, cmap=kwargs.pop("cmap", "jet"))[::-1]
        cols = ["r", "b", "darkorange", "k"]
    else:
        cols = kwargs.pop("cols")
    ls = kwargs.pop("linestyle", "-")

    for i,c in zip(range(len(the_mass_bins)), cols):
        x,y,std,stderr,n = binned_cdf[i]

        if i == len(the_mass_bins) - 1:
            upper = r"$\infty$"
        else:
            upper = "%1.1f" % the_mass_bins[i+1]
        lower = "%1.1f" % the_mass_bins[i]

        label = "[%s, %s)" % (lower, upper)
        print label, n

        ax.errorbar(x, y, yerr=std, label=label, linewidth=3., \
                color=c, linestyle=ls, capsize=5)

    ax.set_xlabel(r"log$_{10}$ n$_{\mathrm{HI}}$ [m$^{-3}$]")
    ax.set_ylabel("CDF")

    if kwargs.pop("legend", True):
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.15,
                         box.width, box.height * 0.9])

        # Put a legend to the right of the current axis
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=False, ncol=6, prop={"size":18})
        leg.set_title(r"log$_{10}$(Mvir) [M$_{\odot}$/h]", prop = {'size':'x-large'})

    if kwargs.pop("label_nstar", False):
        fs = 15
        n_star = snap.array(snap.info_rt["n_star"].express(snap.C.H_cc), "cm**-3").in_units("m**-3")

        ax.vlines(np.log10(n_star), -1, 2, color='k', linestyle='-.')
        # ax1.text(np.log10(snapshot.info_rt['n_star'].express(snapshot.C.H_cc)) + xr*0.01, ymax/2. + .01, r"n$_{*}$", color='r', fontsize=fs)
        # ax1.text(np.log10(n_star) + xr*0.01, 0.076, r"n$_{*}$", color='r', fontsize=fs)

    ax.set_ylim(-0.05, 1.05)
    # ax.set_yscale("log")

    return binned_cdf

_MEM_OPT=False
def main(path, iout, pickle_path):
    '''
    Compute nH CDF of all halos and bin by Mvir
    '''
    from seren3.analysis.parallel import mpi
    from seren3.analysis.plots import histograms

    snap = seren3.load_snapshot(path, iout)

    nH_range = None
    # nH_range = np.array([  1.59355764e+00,   7.93249184e+09])
    nH_range = np.array([  1.0e-01,   1.0e+10])

    if nH_range is None:
        if mpi.host:
            print "Bracketing %s" % _FIELD
            min_nH = np.inf; max_nH = -np.inf
            dset = None
            if _MEM_OPT:
                count=1
                ncpu = snap.info["ncpu"]
                for dset in snap.g[_FIELD]:
                    print "%i/%i cpu" % (count, ncpu)
                    count += 1

                    nH = dset[_FIELD]
                    if nH.min() < min_nH:
                        min_nH = nH.min()
                    if nH.max() > max_nH:
                        max_nH = nH.max()
            else:
                print "Loading full dataset"
                dset = snap.g[_FIELD].flatten()
                print "Loaded full dataset"

                min_nH = dset[_FIELD].min()
                max_nH = dset[_FIELD].max()

            nH_range = np.array([min_nH, max_nH])
            del dset

        nH_range = mpi.comm.bcast(nH_range, root=mpi.HOST_RANK)
        mpi.msg(nH_range)

    snap.set_nproc(1)
    halos = snap.halos()
    nhalos = len(halos)

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        mpi.msg("%i / %i" % (i, nhalos))
        h = halos[i]
        sto.idx = h["id"]

        if len(h.g) > 0:
            P, C, bc, dx = histograms.pdf_cdf(h.g, _FIELD, cumulative=True, plot=False, x_range=nH_range)
            if (P.max() > 0.):
                sto.result = {"C" : C, "bc" : bc, "Mvir" : h["Mvir"]}

    if mpi.host:
        import os, pickle

        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/cdf_%s_halos_%05i.p" % (pickle_path, _FIELD, iout)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    iout = int(sys.argv[2])
    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, iout, pickle_path)
    