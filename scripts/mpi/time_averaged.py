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

    vsum = 0.
    msum = 0.
    for dset in snap.g[[field, "dx", "mass"]]:
        dx = dset["dx"].in_units(length_unit)
        mass = dset["mass"].in_units(mass_unit)

        vsum += np.sum(dset[field] * dx**3)
        msum += np.sum(dset[field] * mass)

    vw = vsum / boxsize**3
    mw = msum / boxmass

    return vw, mw

def main(path, field, pickle_path):
    import seren3
    from seren3.analysis.parallel import mpi
    import pickle, os

    mpi.msg("Loading simulation")
    simulation = seren3.init(path)

    if mpi.host:
        mpi.msg("Averaging field: %s" % field)

    iout_start = max(simulation.numbered_outputs[0], 1)
    iouts = range(iout_start, max(simulation.numbered_outputs)+1)

    mpi.msg("Starting with snapshot %05i" % iout_start)
    dest = {}
    for iout, sto in mpi.piter(iouts, storage=dest, print_stats=True):
        mpi.msg("%05i" % iout)
        snapshot = simulation.snapshot(iout, verbose=False)
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