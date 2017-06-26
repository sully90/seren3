def plot(sims, ioutputs, labels, colours, pickle_paths=None):
    import pickle
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.analysis import plots

    if (pickle_paths is None):
        pickle_paths = ["%s/pickle/" % sim.path for sim in sims]

    fig, ax = plt.subplots()

    for sim, ioutput, ppath, label, c in zip(sims, ioutputs, pickle_paths, labels, colours):
        fname = "%s/ConsistentTrees/halo_clumping_factor_%05i.p" % (ppath, ioutput)
        data = pickle.load( open(fname, "rb") )

        mvir = np.zeros(len(data))
        C = np.zeros(len(data))

        for i in range(len(data)):
            res = data[i].result

            mvir[i] = res["hprops"]["mvir"]
            C[i] = res["C"]

        bc, mean, std = plots.fit_scatter(np.log10(mvir), np.log10(C))

        ax.scatter(mvir, np.log10(C), color=c, s=15, marker="+", alpha=0.25)
        e = ax.errorbar(10**bc, mean, yerr=std, color=c, label=label,\
             fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-')

    ax.set_xlabel(r"M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
    ax.set_ylabel(r"log$_{10}$ $C$")
    ax.set_xscale("log")
    ax.legend()


def main(path, ioutput, pickle_path):
    import seren3
    import pickle, os
    from seren3.analysis.parallel import mpi

    snap = seren3.load_snapshot(path, ioutput)
    halos = snap.halos()

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        if len(h.g) > 1:
            mpi.msg("Working on halo %i \t %i" % (i, h.hid))

            C = h.clumping_factor()
            sto.idx = h["id"]
            sto.result = {"hprops" : h.properties, "C" : C}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/halo_clumping_factor_%05i.p" % (pickle_path, ioutput)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )
        mpi.msg("Done")

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    ioutput = int(sys.argv[2])

    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, ioutput, pickle_path)