import numpy as np

def mass_weighted_average(family, xHII_thresh=0.99):
    dset = family[ ["trec", "mass", "xHII"] ]
    idx = np.where( dset["xHII"] > xHII_thresh )
    return ( dset["trec"][idx] * dset["mass"][idx] ).sum() / dset["mass"][idx].sum()

def main(path, iout, pickle_path):
    import seren3
    from seren3.analysis.parallel import mpi
    import pickle, os

    mpi.msg("Loading snapshot: %05i" % iout)
    snapshot = seren3.load_snapshot(path, iout)
    snapshot.set_nproc(1)

    halos = snapshot.halos(finder="ctrees")
    nhalos = len(halos)

    dest = {}
    for i, sto in mpi.piter(range(nhalos), storage=dest):
        mpi.msg("%i /  %i" % (i+1, nhalos))
        h = halos[i]
        sto.idx = h.hid

        sto.result = {"mw" : mass_weighted_average(h.g), "Mvir" : h["Mvir"]}
    mpi.msg("Done. Waiting.")

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % snapshot.path

        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)

        fname = "%s/trec_mw_halo_av_%05i.p" % (pickle_path, snapshot.ioutput)
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