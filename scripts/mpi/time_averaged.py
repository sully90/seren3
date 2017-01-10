import numpy as np
from seren3.array import SimArray

# mem_opt = False
# do_mass_weighted = True

def volume_mass_weighted_average(snap, field):
    '''
    Computes volume and mass weighted averages of the desired AMR field
    '''

    length_unit = "pc"
    mass_unit = "Msol"
    boxmass = snap.quantities.box_mass('b').in_units(mass_unit)
    boxsize = SimArray(snap.info["boxlen"], snap.info["unit_length"]).in_units(length_unit)

    dset = snap.g[[field, "dx", "mass"]].flatten()
    dx = dset["dx"].in_units(length_unit)
    mass = dset["mass"].in_units(mass_unit)

    vw = np.sum(dset[field] * dx**3) / boxsize**3
    mw = np.sum(dset[field] * mass) / boxmass

    return vw, mw

def main(path, field, pickle_path):
    import seren3
    from seren3.analysis.parallel import mpi
    import pickle, os

    mpi.msg("Loading simulation")
    simulation = seren3.init(path)

    if mpi.host:
        mpi.msg("Averaging field: %s" % field)

    dest = {}
    for iout, sto in mpi.piter(simulation.numbered_outputs, storage=dest):
        mpi.msg("%05i" % iout)
        snapshot = simulation.snapshot(iout, verbose=True)
        snapshot.set_nproc(1)

        # vw = snapshot.quantities.volume_weighted_average(field, mem_opt=mem_opt)
        vw, mw = volume_mass_weighted_average(snapshot, field)
        sto.idx = iout
        sto.result = {"vw" : vw, "mw" : mw, "z" : snapshot.z}

        # if do_mass_weighted:
            # mw = snapshot.quantities.mass_weighted_average(field, mem_opt=mem_opt)
            # sto.result["mw"] = mw

    if mpi.host:
        if pickle_path == None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)

        pickle.dump( mpi.unpack(dest), open("%s/%s_time_averaged.p" % (pickle_path, field), "wb") )


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    field = sys.argv[2]
    pickle_path=None

    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, field, pickle_path)