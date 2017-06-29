'''
Writes out a pickle dictionary where keys and halo ids and values are fesc data
'''

import numpy as np

def load_db(path, ioutput):
    import pickle

    pickle_path = "%s/pickle/ConsistentTrees/" % path
    return pickle.load( open("%s/fesc_database_%05i.p" % (pickle_path, ioutput), "rb") )

def load_halo(halo):
    import pickle

    path = halo.base.path
    ioutput = halo.base.ioutput

    db = load_db(path, ioutput)

    hid = int(halo["id"])
    if (hid in db.keys()):
        return db[int(halo["id"])]
    else:
        return None

def main(path, iout, pickle_path):
    import seren3
    from seren3.core.serensource import DerivedDataset
    from seren3.utils import derived_utils
    from seren3.analysis.escape_fraction import fesc
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException
    import pickle, os

    mpi.msg("Loading snapshot")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)
    halos = None

    star_Nion_d_fn = derived_utils.get_derived_field(snap.s, "Nion_d")
    nIons = snap.info_rt["nIons"]

    halos = snap.halos(finder="ctrees")
    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest, print_stats=True):
        h = halos[i]
        star_dset = h.s[["age", "metal", "mass"]].flatten()
        if (len(star_dset["mass"]) > 0):
            dt = h.dt
            ix_young = np.where((star_dset["age"].in_units("Gyr") - dt.in_units("Gyr")) >= 0.)
            if len(ix_young[0] > 0):
                # Work on this halo
                sto.idx = int(h["id"])
                try:
                    dm_dset = h.d["mass"].flatten()
                    gas_dset = h.g["mass"].flatten()

                    star_mass = star_dset["mass"]
                    star_age = star_dset["age"]
                    star_metal = star_dset["metal"]

                    dict_stars = {"age" : star_age, "metal" : star_metal, "mass" : star_mass}
                    dset_stars = DerivedDataset(snap.s, dict_stars)

                    Nion_d_all_groups = snap.array(np.zeros(len(dset_stars["age"])), "s**-1 Msol**-1")

                    for ii in range(nIons):
                        Nion_d_now = star_Nion_d_fn(snap, dset_stars, group=ii+1, dt=0.)
                        Nion_d_all_groups += Nion_d_now

                    tot_mass = dm_dset["mass"].in_units("Msol h**-1").sum() + star_dset["mass"].in_units("Msol h**-1").sum()\
                                     + gas_dset["mass"].in_units("Msol h**-1").sum()

                    h_fesc = fesc(h.subsnap, nside=2**3, filt=True, do_multigroup=True)
                    mpi.msg("%1.2e \t %1.2e" % (h["Mvir"], h_fesc))
                    sto.result = {"fesc" : h_fesc, "tot_mass" : tot_mass, \
                        "Nion_d_now" : Nion_d_all_groups, "star_dict" : dict_stars,\
                        "hprops" : h.properties}

                except NoParticlesException as e:
                    mpi.msg(e.message)

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/%s/" % (path, halos.finder.lower())
        # pickle_path = "%s/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        unpacked_dest = mpi.unpack(dest)
        fesc_dict = {}
        for i in range(len(unpacked_dest)):
            fesc_dict[int(unpacked_dest[i].idx)] = unpacked_dest[i].result
        pickle.dump( fesc_dict, open( "%s/fesc_database_%05i.p" % (pickle_path, iout), "wb" ) )


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    iout = int(sys.argv[2])

    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, iout, pickle_path)