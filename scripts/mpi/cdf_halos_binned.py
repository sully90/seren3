import numpy as np
import seren3

_FIELD = "nHI"

def plot(path, iout, pickle_path, the_mass_bins=[7., 8., 9., 10.], ax=None, **kwargs):
    import pickle
    from seren3.analysis.plots import fit_scatter
    from seren3.utils import flatten_nested_array
    import matplotlib.pylab as plt

    snap = seren3.load_snapshot(path, iout)
    fname = "%s/cdf_%s_halos_%05i.p" % (pickle_path, _FIELD, iout)
    data = pickle.load(open(fname, 'rb'))

    cdf_halos = []; bin_centre_halos = []; Mvir = []

    for i in range(len(data)):
        res = data[i].result
        cdf_halos.append(res["C"])
        bin_centre_halos.append(res["bc"])
        Mvir.append(res["Mvir"])

    cdf_halos = np.array(cdf_halos); bin_centre_halos = np.array(bin_centre_halos); Mvir = np.array(Mvir)

    # Bin idx
    mass_bins = np.digitize(np.log10(Mvir), the_mass_bins, right=True)
    binned_cdf = {}

    nbins = kwargs.pop("nbins", 5)
    for i in range(len(the_mass_bins)):
        idx = np.where( mass_bins == i )
        x, y = ( flatten_nested_array(bin_centre_halos[idx]), flatten_nested_array(cdf_halos[idx]) )
        idx = np.where(~np.isnan(y))
        binned_cdf[i] = fit_scatter(x[idx], y[idx], ret_sterr=True, ret_n=True, nbins=nbins)

    binned_cdf['the_mass_bins'] = the_mass_bins

    if ax is None:
        ax = plt.gca()

    cols = None
    if "cols" not in kwargs:
        from seren3.utils import plot_utils
        cols = plot_utils.ncols(len(the_mass_bins)+1, cmap=kwargs.pop("cmap", "jet"))[::-1]
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

        ax.errorbar(x, y, yerr=stderr, label=label, linewidth=3., \
                color=c, linestyle=ls, capsize=5)


    ax.set_xlabel(r"log$_{10}$ n$_{\mathrm{HI}}$ [m$^{-3}$]")
    ax.set_ylabel("CDF")

    if kwargs.pop("legend", True):
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                         box.width, box.height * 0.8])

        # Put a legend to the right of the current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=6, title=r"log$_{10}$(Mvir) [M$_{\odot}$/h]")

    if kwargs.pop("label_nstar", False):
        fs = 15
        n_star = snap.array(snap.info_rt["n_star"].express(snap.C.H_cc), "cm**-3").in_units("m**-3")
        ax.vlines(np.log10(n_star), -1, 2, color='r', linestyle='-.')
        # ax1.text(np.log10(snapshot.info_rt['n_star'].express(snapshot.C.H_cc)) + xr*0.01, ymax/2. + .01, r"n$_{*}$", color='r', fontsize=fs)
        # ax1.text(np.log10(n_star) + xr*0.01, 0.076, r"n$_{*}$", color='r', fontsize=fs)

    ax.set_ylim(0., 1.)

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