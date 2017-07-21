'''
Writes out a pickle dictionary where keys and halo ids and values are fesc data
'''

import numpy as np

def load_db(path, ioutput):
    import pickle

    pickle_path = "%s/pickle/ConsistentTrees/" % path
    # return pickle.load( open("%s/mass_flux_database_%05i.p" % (pickle_path, ioutput), "rb") )
    # return pickle.load( open("%s/mass_flux_database_no_filt_half_rvir_%05i.p" % (pickle_path, ioutput), "rb") )
    return pickle.load( open("%s/mass_flux_database_no_filt_half_rvir_denoise_%05i.p" % (pickle_path, ioutput), "rb") )

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

def plot_dm_by_dt_fesc(snapshot):
    import random
    import matplotlib.pylab as plt
    from seren3.scripts.mpi import write_fesc_hid_dict

    fesc_db = write_fesc_hid_dict.load_db(snapshot.path, snapshot.ioutput)
    mass_flux_db = load_db(snapshot.path, snapshot.ioutput)

    hids = fesc_db.keys()
    fesc = np.zeros(len(hids))
    outflow_dm_by_dt = np.zeros(len(hids))

    i = 0
    for hid in hids:
        ifesc = fesc_db[hid]["fesc"]
        if (ifesc > 1.):
            ifesc = random.uniform(0.9, 1.0)
        fesc[i] = ifesc
        F, F_plus, F_minus = mass_flux_db[hid]["F"]
        outflow_dm_by_dt[i] = F_plus
        i += 1

    plt.scatter(outflow_dm_by_dt, fesc)
    plt.ylabel(r"f$_{\mathrm{esc}}$")
    plt.xlabel(r"$dM/dt$ [M$_{\odot}$/h]")
    plt.show()


def main(path, iout, pickle_path):
    import seren3
    from seren3.core.serensource import DerivedDataset
    from seren3.utils import derived_utils
    from seren3.analysis import outflows
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException
    import pickle, os

    mpi.msg("Loading snapshot")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)
    halos = None

    halos = snap.halos(finder="ctrees")
    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest, print_stats=True):
        h = halos[i]

        if (len(h.s) > 0):
            sto.idx = int(h["id"])
            dm_dset = h.d["mass"].flatten()
            gas_dset = h.g["mass"].flatten()
            star_dset = h.s["mass"].flatten()

            tot_mass = dm_dset["mass"].in_units("Msol h**-1").sum() + star_dset["mass"].in_units("Msol h**-1").sum()\
                             + gas_dset["mass"].in_units("Msol h**-1").sum()

            F, h_im = outflows.dm_by_dt(h.subsnap, filt=False, nside=2**3, denoise=True)
            sto.result = {"F" : F, "h_im" : h_im, "tot_mass" : tot_mass, \
                "hprops" : h.properties}

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
        pickle.dump( fesc_dict, open( "%s/mass_flux_database_no_filt_half_rvir_denoise_%05i.p" % (pickle_path, iout), "wb" ) )


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    iout = int(sys.argv[2])

    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, iout, pickle_path)