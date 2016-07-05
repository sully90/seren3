import numpy as np
import pickle

def unpack(path, iout):
    fname = '%s/abundance_%05i.p' % (path, iout)
    dest = pickle.load( open(fname, 'rb') )
    tot_mass = []
    star_mass = []
    for rank in dest:
        for item in dest[rank]:
            res = item.result
            tot_mass.append(res["tot_mass"])
            star_mass.append(res["star_mass"])

    return np.array(star_mass), np.array(tot_mass)

def stars(dset):
    return np.where( np.logical_and(dset["epoch"] != 0, dset["id"] > 0.) )

def main(path, iout):
    import seren3
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading data")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)  # disable multiprocessing

    halos = snap.halos()
    halo_spheres = halos.mpi_spheres()

    dest = {}
    for h, sto in mpi.piter(halo_spheres, storage=dest):
        sphere = h["reg"]
        subsnap = snap[sphere]

        part_dset = subsnap.p[["mass", "epoch", "id"]].flatten()
        gas_mass = subsnap.g["mass"].in_units("Msol")

        star_idx = stars(part_dset)
        part_mass = part_dset["mass"].in_units("Msol")

        tot_mass = part_mass.sum() + gas_mass.sum()
        star_mass = part_mass[star_idx].sum()

        result = {"tot_mass" : tot_mass, "star_mass" : star_mass}
        sto.idx = h["id"]
        sto.result = result

    pickle.dump( dest, open('./abundance_%05i.p' % iout, 'wb') )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    main(path, iout)