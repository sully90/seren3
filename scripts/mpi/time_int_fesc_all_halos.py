def main(path):
    import random
    import numpy as np
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException
    from seren3.analysis.escape_fraction import time_integrated_fesc

    mpi.msg("Loading simulation...")
    sim = seren3.init(path)

    # for snap in sim:
    for iout in sim.numbered_outputs[::-1]:
        snap = sim[iout]
        mpi.msg("Working on snapshot %05i" % snap.ioutput)
        snap.set_nproc(1)
        halos = snap.halos(finder="ctrees")

        halo_ix = range(len(halos))
        random.shuffle(halo_ix)

        dest = {}
        for i, sto in mpi.piter(halo_ix, storage=dest):
            h = halos[i]

            age_delay = h.s["age"].f.in_units("Gyr") - h.dt.in_units("Gyr")

            # Check if all values are below zero i.e no stars dt ago
            if len(age_delay) > 0 and any(age_delay >= 0.):
                try:
                    mpi.msg("%05i \t %i" % (snap.ioutput, h.hid))
                    tint_fesc_hist, I1, I2, lbtime = time_integrated_fesc(h, 0., return_data=True)

                    fesc = I1/I2
                    sto.idx = h.hid
                    sto.result = {'tint_fesc_hist' : tint_fesc_hist, 'fesc' : fesc, 'I1' : I1, \
                            'I2' : I2, 'lbtime' : lbtime, 'Mvir' : h["Mvir"]}
                except NoParticlesException as e:
                    # mpi.msg(e.message)
                    continue

        if mpi.host:
            import pickle, os
            pickle_path = "%s/pickle/%s/" % (snap.path, halos.finder)
            if not os.path.isdir(pickle_path):
                os.mkdir(pickle_path)
            pickle.dump( mpi.unpack(dest), open("%s/time_int_fesc_all_halos_%05i.p" % (pickle_path, snap.ioutput), "wb") )

        mpi.msg("Waiting...")
        mpi.comm.Barrier()

if __name__ == "__main__":
    import sys, warnings
    path = sys.argv[1]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # main(path)
        try:
           main(path)
        except Exception as e:
           from seren3.analysis.parallel import mpi
           mpi.msg("Caught exception - terminating")
           mpi.terminate(500, e=e)

