def test_plot():
    import matplotlib.pylab as plt
    paths = ['/lustre/scratch/astro/ds381/simulations/bpass/ramses_sed_bpass/rbubble_200/', '/lustre/scratch/astro/ds381/simulations/bpass/ramses-rt/rbubble_200/']
    iouts = [100, 100]
    labels = ['BPASS', 'BC03']
    plot(paths, iouts, labels, alpha=0.1, cols=['b', 'r'])

    plt.show()

def _plot(fesc, tot_mass, label, alpha, ax, c, nbins):
    import numpy as np
    import random
    # from seren3.analysis.plots import fit_scatter
    from seren3.analysis import plots

    # Deal with fesc>1. following Kimm & Cen 2014
    bad = np.where( fesc > 1. )
    nbad = float(len(bad[0]))
    print "%f %% of points above 1" % ((nbad/float(len(fesc)))*100.)
    for ix in bad:
        fesc[ix] = random.uniform(0.9, 1.0)

    # keep = np.where( np.logical_and(fesc >= 0., fesc <=1.) )
    # fesc = fesc[keep]
    # tot_mass = tot_mass[keep]

    log_tot_mass = np.log10(tot_mass)
    log_fesc = np.log10(100. * fesc)

    remove = np.where( np.logical_or( np.isinf(log_fesc), np.isnan(log_fesc) ) )

    log_fesc = np.delete( log_fesc, remove )
    log_tot_mass = np.delete( log_tot_mass, remove )

    bin_centres, mean, std_dev, std_err = plots.fit_scatter(log_tot_mass, log_fesc, nbins=nbins, ret_sterr=True)
    # print bin_centres, mean

    # Plot
    ax.scatter(log_tot_mass, log_fesc, marker='+', color=c, alpha=alpha)
    ax.errorbar(bin_centres, mean, yerr=std_err, color=c, label=label, linewidth=2.)

def plot_neighbouring_snaps(paths, ioutputs, labels, nbins=5, num_neighbours=2, alpha=0.25, ax=None, cols=None):
    import seren3
    import matplotlib.pylab as plt
    import numpy as np
    import pickle

    assert(num_neighbours%2 == 0)

    if ax is None:
        fig, ax = plt.subplots()

    if cols is None:
        from seren3.utils import plot_utils
        cols = plot_utils.ncols(len(paths))

    for path, iout, label, c in zip(paths, ioutputs, labels, cols):
        fesc = []
        tot_mass = []
        for i in range(iout - (num_neighbours/2), iout + (num_neighbours/2) + 1):
            # print i
            fname = "%s/pickle/fesc_%05i.p" % (path, i)
            with open(fname, "rb") as f:
                dset = pickle.load( f )
                
                for i in range(len(dset)):
                    res = dset[i].result
                    fesc.append(res["fesc"])
                    tot_mass.append(res["tot_mass"])

        fesc = np.array(fesc)
        tot_mass = np.array(tot_mass)
        _plot(fesc, tot_mass, label, alpha, ax, c, nbins)


    ax.set_xlabel(r"log$_{10}$(M$_{\mathrm{vir}}$ [M$_{\odot}$])")
    ax.set_ylabel(r"log$_{10}$(f$_{\mathrm{esc}}$ [%])")
    plt.legend()
    return ax    

def plot(paths, ioutputs, labels, nbins=5, alpha=0.25, ax=None, cols=None):
    import seren3
    import matplotlib.pylab as plt
    import numpy as np
    import pickle

    if ax is None:
        fig, ax = plt.subplots()

    if cols is None:
        from seren3.utils import plot_utils
        cols = plot_utils.ncols(len(paths))

    for path, iout, label, c in zip(paths, ioutputs, labels, cols):
        fname = "%s/pickle/fesc_%05i.p" % (path, iout)
        with open(fname, "rb") as f:
            dset = pickle.load( f )

            fesc = np.zeros(len(dset)); tot_mass = np.zeros(len(dset))
            for i in range(len(dset)):
                res = dset[i].result
                fesc[i] = res["fesc"]
                tot_mass[i] = res["tot_mass"]

            _plot(fesc, tot_mass, label, alpha, ax, c, nbins)


    ax.set_xlabel(r"log$_{10}$(M$_{\mathrm{vir}}$ [M$_{\odot}$])")
    ax.set_ylabel(r"log$_{10}$(f$_{\mathrm{esc}}$ [%])")
    plt.legend()
    return ax


def load(path, iout, finder):
    import pickle
    import numpy as np

    fname = "%s/pickle/%s/fesc_%05i.p" % (path, finder, iout)
    fesc = None; tot_mass = None
    with open(fname, 'rb') as f:
        dset = pickle.load( f )
        fesc = np.zeros(len(dset)); tot_mass = np.zeros(len(dset))
        for i in range(len(dset)):
            res = dset[i].result
            fesc[i] = res["fesc"]
            tot_mass[i] = res["tot_mass"]

    return tot_mass, fesc


def main(path, iout, finder):
    import seren3
    from seren3.analysis import fesc
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException
    import pickle, os

    mpi.msg("Loading snapshot")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)
    halos = None

    mpi.msg("Using halo finder: %s" % finder)
    mpi_spheres = None
    if mpi.host:
        halos = snap.halos(finder=finder)
        mpi_spheres = halos.mpi_spheres()

    dest = {}
    for h, sto in mpi.piter(mpi_spheres, storage=dest):
        sph = h["reg"]
        subsnap = snap[sph]
        if len(subsnap.s) > 0:
            tot_mass = subsnap.p["mass"].f.sum() + subsnap.g["mass"].f.sum()

            sto.idx = h["id"]
            h_fesc = 0.

            try:
                h_fesc = fesc(subsnap, nside=2**4)
            except NoParticlesException as e:
                mpi.msg(e.message)
            except IndexError:
                mpi.msg("Index error - skipping")
            sto.result = {"fesc" : h_fesc, "tot_mass" : tot_mass}

    if mpi.host:
        pickle_path = "%s/pickle/%s/" % (path, halos.finder.lower())
        # pickle_path = "%s/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        pickle.dump( mpi.unpack(dest), open( "%s/fesc_%05i.p" % (pickle_path, iout), "wb" ) )

if __name__ == "__main__":
    import sys
    from seren3 import config

    path = sys.argv[1]
    iout = int(sys.argv[2])

    finder = config.get("halo", "default_finder")
    if len(sys.argv) == 4:
        finder = sys.argv[3]

    # main(path, iout, finder)

    try:
        main(path, iout, finder)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        print 'Caught exception: ', e
        mpi.terminate(500, e=e)