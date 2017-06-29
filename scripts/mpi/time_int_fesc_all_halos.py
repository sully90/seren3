def main(path, pickle_path):
    import random
    import numpy as np
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException
    from seren3.analysis.escape_fraction import time_integrated_fesc
    from seren3.scripts.mpi import write_fesc_hid_dict

    mpi.msg("Loading simulation...")
    sim = seren3.init(path)

    iout_start = max(sim.numbered_outputs[0], 61)
    back_to_aexp = sim[60].info["aexp"]
    # iouts = range(iout_start, max(sim.numbered_outputs)+1)
    print "IOUT RANGE HARD CODED"
    iouts = range(iout_start, 110)

    for iout in iouts[::-1]:
        snap = sim[iout]
        mpi.msg("Working on snapshot %05i" % snap.ioutput)
        snap.set_nproc(1)
        halos = snap.halos(finder="ctrees")

        halo_ids = None
        if mpi.host:
            db = write_fesc_hid_dict.load_db(path, iout)
            halo_ids = db.keys()
            random.shuffle(halo_ids)

        dest = {}
        for i, sto in mpi.piter(halo_ids, storage=dest, print_stats=True):
            h = halos.with_id(i)
            res = time_integrated_fesc(h, back_to_aexp, return_data=True)
            if (res is not None):
                mpi.msg("%05i \t %i \t %i" % (snap.ioutput, h.hid, i))
                tint_fesc_hist, I1, I2, lbtime = res

                fesc = I1/I2
                sto.idx = h.hid
                sto.result = {'tint_fesc_hist' : tint_fesc_hist, 'fesc' : fesc, 'I1' : I1, \
                        'I2' : I2, 'lbtime' : lbtime, 'Mvir' : h["Mvir"]}
        if mpi.host:
            import pickle, os
            # pickle_path = "%s/pickle/%s/" % (snap.path, halos.finder)
            if not os.path.isdir(pickle_path):
                os.mkdir(pickle_path)
            pickle.dump( mpi.unpack(dest), open("%s/time_int_fesc_all_halos_%05i.p" % (pickle_path, snap.ioutput), "wb") )

        mpi.msg("Waiting...")
        mpi.comm.Barrier()

if __name__ == "__main__":
    import sys, warnings
    path = sys.argv[1]
    pickle_path = sys.argv[2]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # main(path)
        # try:
        main(path, pickle_path)
        # except Exception as e:
        #    from seren3.analysis.parallel import mpi
        #    mpi.msg("Caught exception - terminating")
        #    mpi.terminate(500, e=e)

