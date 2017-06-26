def main(path, iout, pickle_path):
    import pickle, os
    import seren3
    from seren3.analysis import baryon_fraction
    from seren3.analysis.parallel import mpi

    back_to_z = 20.
    back_to_aexp = 1. / (1. + back_to_z)

    mpi.msg("Loading data")
    snap = seren3.load_snapshot(path, iout)

    snap.set_nproc(1)  # disbale multiprocessing/threading

    halos = snap.halos(finder="ctrees")
    nhalos = len(halos)

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest, print_stats=True):
        # mpi.msg("%i / %i" % (i, nhalos))
        h = halos[i]

        tint_fbaryon_hist, I1, I2, lbtime, fbaryon_dict, tidal_force_tdyn_dict, age_dict = baryon_fraction.time_integrated_fbaryon(h, back_to_aexp)

        sto.idx = h["id"]
        sto.result = {"tint_fbaryon" : tint_fbaryon_hist, "I1" : I1, "I2" : I2, "lbtime" : lbtime, "fbaryon_dict" : fbaryon_dict, \
            "tidal_force_tdyn_dict" : tidal_force_tdyn_dict, "age_dict" : age_dict}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/tint_fbaryon_tdyn_%05i_fixed.p" % (pickle_path, iout)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])

    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]
    try:
        main(path, iout, pickle_path)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        mpi.terminate(500, e=e)