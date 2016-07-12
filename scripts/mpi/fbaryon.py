import numpy as np

def main(path, iout):
    import seren3
    import pickle, os
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading data")
    snap = seren3.load_cwd(iout)
    snap.set_nproc(1)  # disbale multiprocessing

    halos = snap.halos()
    mpi_halos = halos.mpi_spheres()

    dest = {}
    for h, sto in mpi.piter(mpi_halos, storage=dest):
        sphere = h["reg"]
        subsnap = snap[sphere]

        # part_dset = subsnap.p[["mass", "epoch", "id"]].f
        part_mass = subsnap.p["mass"].f
        gas_mass = subsnap.g["mass"].f

        part_mass_tot = part_mass.in_units("Msol").sum()
        gas_mass_tot = gas_mass.in_units("Msol").sum()
        tot_mass = part_mass_tot + gas_mass_tot

        fb = gas_mass_tot/tot_mass
        sto.idx = h["id"]
        sto.result = {"fb" : fb, "tot_mass" : tot_mass}

    if mpi.host:
        pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        pickle.dump( mpi.unpack(dest), open( "%s/fbaryon_%05i.p" % (pickle_path, iout), "wb" ) )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    try:
        main(path, iout)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        mpi.terminate(500, e=e)