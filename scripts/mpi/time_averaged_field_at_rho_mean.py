def main(path, field, pickle_path):
    import seren3
    from seren3.analysis import mean_field_rho_mean
    from seren3.analysis.parallel import mpi
    import pickle, os

    mpi.msg("Loading simulation")
    simulation = seren3.init(path)

    if mpi.host:
        mpi.msg("Averaging field: %s" % field)

    iout_start = max(simulation.numbered_outputs[0], 1)
    iouts = range(iout_start, max(simulation.numbered_outputs)+1)

    mpi.msg("Starting with snapshot %05i" % iout_start)
    dest = {}
    for iout, sto in mpi.piter(iouts, storage=dest, print_stats=True):
        mpi.msg("%05i" % iout)
        snapshot = simulation.snapshot(iout, verbose=False)
        snapshot.set_nproc(1)

        mw, rho_mean = mean_field_rho_mean(snapshot, field, ret_rho_mean=True)
        sto.idx = iout
        sto.result = {"rho_mean" : rho_mean, "mw" : mw, "z" : snapshot.z, "aexp" : snapshot.cosmo["aexp"]}

    if mpi.host:
        if pickle_path == None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)

        pickle.dump( mpi.unpack(dest), open("%s/%s_time_averaged_at_rho_mean.p" % (pickle_path, field), "wb") )


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    field = sys.argv[2]
    pickle_path = None

    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, field, pickle_path)