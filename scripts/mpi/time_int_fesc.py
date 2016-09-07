def main(path, ioutput):
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException
    from seren3.analysis import time_integrated_fesc

    mpi.msg("Loading snapshot...")
    snap = seren3.load_snapshot(path, ioutput)
    snap.set_nproc(1)
    halos = snap.halos(finder="ctrees")

    halo_ix = range(len(halos))

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        try:
            tint_fesc, I1, I2, lbtime = time_integrated_fesc(h, 1, return_data=True)
            fesc = I1/I2
            sto.idx = h.hid
            sto.result = {'tint_fesc' : tint_fesc, 'fesc' : fesc, 'I1' : I1, \
                    'I2' : I2, 'lbtime' : lbtime, 'Mvir' : h['Mvir']}
        except NoParticlesException as e:
            mpi.msg(e.message)

    if mpi.host:
        import pickle, os
        pickle_path = "%s/pickle/%s/" % (snap.path, halos.finder)
        if not os.path.isdir(pickle_path):
            os.mkdir(pickle_path)
        pickle.dump( mpi.unpack(dest), open("%s/time_int_fesc_%05i.p" % (pickle_path, snap.ioutput)) )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    ioutput = int(sys.argv[2])
    main(path, ioutput)