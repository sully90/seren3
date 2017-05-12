import numpy as np

def _volume_weighted_average(field, halo, npoints=100000):
    points = halo.sphere.random_points(npoints)
    dset = halo.g[field].sample_points(points, use_multiprocessing=False)

    return dset[field].mean()

def _mass_weighted_average(field, halo, mass_units="Msol h**-1"):
    dset = halo.g[[field, "mass"]].flatten()

    cell_mass = dset["mass"].in_units(mass_units)

    return np.sum(dset[field]*cell_mass)/cell_mass.sum()

def main(path, iout, field, pickle_path=None):
    import seren3
    import pickle, os
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading data")
    snap = seren3.load_snapshot(path, iout)
    # snap.set_nproc(1)  # disbale multiprocessing/threading
    snap.set_nproc(8)

    halos = snap.halos()
    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        mpi.msg("Working on halo %i \t %i" % (i, h.hid))

        vw = _volume_weighted_average(field, h)
        mw = _mass_weighted_average(field, h)
        # vw = _volume_weighted_average_cube(snap, field, h)
        mpi.msg("%i \t %1.5f" % (h.hid, vw))

        sto.idx = h["id"]
        sto.result = {"vw" : vw, "mw" : mw}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/%s_halo_av_%05i.p" % (pickle_path, field, iout)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )
        mpi.msg("Done")


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    field = sys.argv[3]
    pickle_path = None
    if len(sys.argv) > 4:
        pickle_path = sys.argv[4]

    main(path, iout, field, pickle_path)
    
