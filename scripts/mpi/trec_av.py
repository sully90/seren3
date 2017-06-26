# Filter out neutral cells!
# TODO -> Total number of recombinations per atom (integrated quantity) and mean number of recombinations
import numpy as np

mem_opt = False
field = "trec"
xHII_thresh = 0.99

def mass_weighted_average(family):
    dset = family[ ["trec", "mass", "xHII"] ]
    idx = np.where( dset["xHII"] > xHII_thresh )
    return ( dset["trec"][idx] * dset["mass"][idx] ).sum() / dset["mass"][idx].sum()

def main(path, pickle_path):
    import seren3
    from seren3.analysis.parallel import mpi
    import pickle, os

    mpi.msg("Loading simulation")
    simulation = seren3.init(path)

    if mpi.host:
        mpi.msg("Averaging field: %s" % field)

    iout_start = max(simulation.numbered_outputs[0], 20)
    iouts = range(iout_start, max(simulation.numbered_outputs)+1)
    dest = {}
    for iout, sto in mpi.piter(iouts, storage=dest):
        mpi.msg("%05i" % iout)
        snapshot = simulation[iout]
        snapshot.set_nproc(1)

        sto.result = {"mw" : mass_weighted_average(snapshot.g), \
                "z" : snapshot.z}

    if mpi.host:
        if pickle_path == None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)

        pickle.dump( mpi.unpack(dest), open("%s/%s_mw_time_averaged.p" % (pickle_path, field), "wb") )


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    pickle_path=None

    if len(sys.argv) > 2:
        pickle_path = sys.argv[2]

    main(path, pickle_path)