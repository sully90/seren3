def main(path):
    import numpy as np
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException, CatalogueNotFoundException
    from seren3.analysis import time_integrated_fesc

    mpi.msg("Loading simulation...")
    sim = seren3.init(path)

    # for snap in sim:
    for iout in sim.numbered_outputs[::-1]:
        snap = sim[iout]
        mpi.msg("Working on snapshot %05i" % snap.ioutput)
        snap.set_nproc(1)
        try:
            halos = snap.halos(finder="ctrees")

            halo_ix = range(len(halos))

            dest = {}
            for i, sto in mpi.piter(halo_ix, storage=dest):
                h = halos[i]

                age_delay = h.s["age"].f.in_units("Gyr") - h.dt.in_units("Gyr")

                # Check if all values are below zero i.e no stars dt ago
                if len(age_delay) > 0 and any(age_delay >= 0.):
                    mpi.msg("*********************** %i, %s ***********************" % (i, h))

                    tint_fesc, I1, I2, lbtime = time_integrated_fesc(h, 1, return_data=True)
                    fesc = I1/I2
                    sto.idx = snap.ioutput
                    sto.result = {'tint_fesc' : tint_fesc, 'fesc' : fesc, 'I1' : I1, \
                            'I2' : I2, 'lbtime' : lbtime, 'Mvir' : h['Mvir'], 'hid' : h.hid}

            if mpi.host:
                import pickle, os
                pickle_path = "%s/pickle/%s/" % (snap.path, halos.finder)
                if not os.path.isdir(pickle_path):
                    os.mkdir(pickle_path)
                pickle.dump( mpi.unpack(dest), open("%s/time_int_fesc_%05i.p" % (pickle_path, snap.ioutput), "wb") )
        except CatalogueNotFoundException as e:
            print e.message
        mpi.msg("Waiting...")
        mpi.comm.Barrier()

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    # main(path)
    try:
       main(path)
    except Exception as e:
       from seren3.analysis.parallel import mpi
       mpi.msg("Caught exception - terminating")
       mpi.terminate(500, e=e)
