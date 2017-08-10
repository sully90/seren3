def load(snap):
    import pickle

    data = pickle.load( open("%s/pickle/ConsistentTrees/history_mass_flux_all_halos_half_rvir_%05i.p" % (snap.path, snap.ioutput), "rb") )
    # data = pickle.load( open("%s/pickle/ConsistentTrees/history_mass_flux_all_halos_no_denoise_%05i.p" % (snap.path, snap.ioutput), "rb") )
    return data


def load_halo(halo):
    data = load(halo.base)

    for i in range(len(data)):
        if (int(data[i].idx) == int(halo["id"])):
            return data[i].result


def main(path, pickle_path):
    import random
    import numpy as np
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.analysis.outflows import mass_flux_hist
    from seren3.scripts.mpi import write_mass_flux_hid_dict

    mpi.msg("Loading simulation...")
    sim = seren3.init(path)

    iout_start = max(sim.numbered_outputs[0], 60)
    back_to_aexp = sim[iout_start].info["aexp"]
    # iouts = range(iout_start, max(sim.numbered_outputs)+1)
    print "IOUT RANGE HARD CODED"
    iouts = range(iout_start, 109)
    # iouts = [109]

    for iout in iouts[::-1]:
        snap = sim[iout]
        mpi.msg("Working on snapshot %05i" % snap.ioutput)
        snap.set_nproc(1)
        halos = snap.halos(finder="ctrees")

        db = write_mass_flux_hid_dict.load_db(snap.path, snap.ioutput)

        halo_ids = None
        if mpi.host:
            halo_ids = db.keys()
            random.shuffle(halo_ids)

        dest = {}
        for i, sto in mpi.piter(halo_ids, storage=dest, print_stats=True):
            h = halos.with_id(i)
            
            F, age_arr, lbtime, hids, iouts = mass_flux_hist(h, back_to_aexp, return_data=True, db=db)

            sto.idx = h.hid
            sto.result = {'F' : F, 'age_array' : age_arr, 'lbtime' : lbtime, \
                    'hids' : hids, 'iouts' : iouts, 'Mvir' : h["Mvir"]}
        if mpi.host:
            import pickle, os
            # pickle_path = "%s/pickle/%s/" % (snap.path, halos.finder)
            if not os.path.isdir(pickle_path):
                os.mkdir(pickle_path)
            pickle.dump( mpi.unpack(dest), open("%s/history_mass_flux_all_halos_half_rvir_%05i.p" % (pickle_path, snap.ioutput), "wb") )

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

