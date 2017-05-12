def main(path, ioutput, pickle_path):
    import seren3
    from seren3.analysis import stars
    from seren3.analysis.parallel import mpi

    snap = seren3.load_snapshot(path, ioutput)

    age_max = snap.array(0.9279320933559091, "Gyr")
    age_min = snap.array(0., "Gyr")
    agerange = [age_min, age_max]

    snap.set_nproc(1)
    halos = snap.halos()
    nhalos = len(halos)

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest, print_stats=True):
        h = halos[i]

        if (len(h.s) > 0):
            star_mass = h.s["mass"].flatten()["mass"].in_units("Msol")

            SFR, lookback_time, binsize = stars.sfr(h, agerange=agerange)
            sSFR = SFR / star_mass.sum()

            rho_sfr = stars.gas_SFR_density(h)

            sto.idx = h["id"]
            sto.result = {"SFR" : SFR, "sSFR" : sSFR, "lookback-time" : lookback_time,\
                    "binsize" : binsize, "rho_sfr" : rho_sfr}

    if mpi.host:
        import os, pickle

        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/sfr_halos_%05i.p" % (pickle_path, iout)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    ioutput = int(sys.argv[2])
    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, ioutput, pickle_path)