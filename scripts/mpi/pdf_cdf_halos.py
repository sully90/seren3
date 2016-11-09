import numpy as np

def main(path, iout, pickle_path):
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.analysis.plots import histograms
    import pickle, os

    mpi.msg("Loading snapshot %05i" % iout)
    snapshot = seren3.load_snapshot(path, iout)
    halos = snapshot.halos(finder="ctrees")
    nhalos = len(halos)

    # Bracket min/max nH

    if mpi.host:
        mpi.msg("Bracketing nH...")
        min_nH = np.inf; max_nH = -np.inf
        for nH in snapshot.g["nH"]:
            if nH.min() < min_nH:
                min_nH = nH.min()
            if nH.max() > max_nH:
                max_nH = nH.max()
        data = {"min" : np.log10(min_nH), "max" : np.log10(max_nH)}
        mpi.msg("Done")
    else:
        data = None
    mpi.msg("bcast nH")
    data = mpi.comm.bcast(data, root=0)
    mpi.msg("Done. Processing halos...")

    min_nH = data["min"]; max_nH = data["max"]
    x_range = (min_nH, max_nH)
    mpi.msg("x_range (log): (%f, %f)" % (min_nH, max_nH))

    snapshot.set_nproc(1)
    dest = {}
    for i, sto in mpi.piter(range(nhalos), storage=dest):
        h = halos[i]
        if len(h.g["nH"].f) > 0:
            P, C, bincenters, dx = histograms.pdf_cdf(h.g, "nH", density=True, \
                    cumulative=True, x_range=x_range)
            sto.idx = h.hid
            sto.result = {"P" : P, "C" : C, "bincenters" : bincenters, "dx" : dx}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % snapshot.path

        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)

        fname = "%s/pdf_cdf_halo_%05i.p" % (pickle_path, snapshot.ioutput)
        mpi.msg("Saving data to pickle file: %s" % fname)
        with open(fname, "wb") as f:
            pickle.dump(mpi.unpack(dest), f)

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])

    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, iout, pickle_path)