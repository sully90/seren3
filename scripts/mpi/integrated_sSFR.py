# Compute the instantaneous integrated sSFR per halo mass bin
import numpy as np

def current_age(snapshot):
    import cosmolopy.distance as cd
    import cosmolopy.constants as cc
    cosmo = snapshot.cosmo
    return cd.age(**cosmo) / cc.Gyr_s  # Gyr

def main(path, iout, nbins=50):
    import seren3, pickle
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException

    mpi.msg("Loading data...")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)  # disable multiprocessing

    halos = snap.halos()

    t = current_age(snap)
    agerange = [0, t]

    # halo_spheres = np.array( [ {'id' : h.hid, 'reg' : h.sphere, 'mvir' : h['mvir'].v} for h in halos ] )
    halo_spheres = halos.mpi_spheres()

    dest = {}
    for h, sto in mpi.piter(halo_spheres, storage=dest):
        sphere = h["reg"]
        subsnap = snap[sphere]

        try:
            dset = subsnap.s["sSFR"].flatten(nbins=nbins, agerange=agerange)
            dset["mvir"] = h["mvir"]
            sto.idx = h["id"]
            sto.result = dset
        except NoParticlesException as e:
            mpi.msg(e.message)
    pickle.dump( dest, open("./integrate_sSFR_%05i.p" % iout, "wb") )

if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    iout = int(sys.argv[2])
    # path = "/lustre/scratch/astro/ds381/simulations/bpass/ramses_sed_bpass/rbubble_200/"
    # path = "/lustre/scratch/astro/ds381/simulations/bpass/hopkins/"
    # iout = 61
    # iout = 60
    main(path, iout)