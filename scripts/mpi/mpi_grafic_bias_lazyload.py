'''
Script for applying sub-grid vbc dependent bias. Requires that the transfer.x binary (from CICsASS)
be in your $PATH. The code will fork additional processes to run transfer.x, so use a maximum of half
the available cores per node when allocation MPI processes.
'''

import sys
import numpy as np
from seren3.core.grafic_snapshot import GrafICSnapshot
from seren3.analysis.parallel import mpi
from seren3.analysis import drift_velocity
from seren3.utils import divisors

########################## MPI PARAMS ####################

comm = mpi.comm
size = comm.Get_size()
rank = comm.Get_rank()

host = (rank == 0)

debug = True
comm_debug = False

########################## SCRIPT PARAMS ####################

pad = 8  # number of cells to pad each region with
out_base_dir = "ics_ramses_vbc"  # the output directory for new ICs (level dir is handled below)

######################### BEGIN MAIN ########################

class Patch(object):

    def __init__(self, cube, dx, field):
        self.cube = cube
        self.dx = dx
        self.field = field

def apply_bias(ic, species, cube, dx, pad, field, ax=None):
    origin = np.array(cube - float(dx) / 2. - pad).astype(np.int32)
    dx_eps = float(dx) + float(2 * pad)

    if debug:
        mpi.msg("Loading fields")

    delta_cube = None
    if (field == 'vel'):
        # Velocity field
        delta_cube = ic.lazy_load_periodic(
            'vel%s%s' % (species, ax), origin, int(dx_eps))
    elif (field == 'rho'):
        delta_cube = ic.lazy_load_periodic(
            'delta%s' % species, origin, int(dx_eps))
    else:
        mpi.msg("Unknown species %s" % species)
        mpi.terminate(500)
    vbc_cube = ic.lazy_load_periodic('vbc', origin, int(dx_eps))

    if debug:
        mpi.msg("Done")

    # Compute the PS bias
    if debug:
        mpi.msg("Computing bias")
    k = bc = bb = None

    if (field == 'vel'):
        k, bc, bb = drift_velocity.compute_velocity_bias(ic, vbc_cube)
    else:
        k, bc, bb = drift_velocity.compute_bias(ic, vbc_cube)
    # Apply bias to this patch
    if debug:
        mpi.msg("Applying bias")

    if 'b' == species:
        modified_delta = drift_velocity.apply_density_bias(ic,
            k, bb, delta_cube.shape[0], delta_x=delta_cube)
    elif 'c' == species:
        modified_delta = drift_velocity.apply_density_bias(ic,
            k, bc, delta_cube.shape[0], delta_x=delta_cube)
    else:
        mpi.terminate("Unknown species %s" % species)

    x_shape, y_shape, z_shape = modified_delta.shape
    # Remove padding
    modified_delta = modified_delta[0 + pad:x_shape - pad,
                                    0 + pad:y_shape - pad, 0 + pad:z_shape - pad]

    return modified_delta


def get_patch_bounds(patch):
    x_min, x_max = (
        int((patch['cube'][0]) - (patch['dx'] / 2.)), int((patch['cube'][0]) + (patch['dx'] / 2.)))
    y_min, y_max = (
        int((patch['cube'][1]) - (patch['dx'] / 2.)), int((patch['cube'][1]) + (patch['dx'] / 2.)))
    z_min, z_max = (
        int((patch['cube'][2]) - (patch['dx'] / 2.)), int((patch['cube'][2]) + (patch['dx'] / 2.)))
    return x_min, x_max, y_min, y_max, z_min, z_max


def main(path, level, patch_size, species, field, ax=None):
    if host:
        mpi.msg("Loading initial conditions for level %d" % level)

    ic = GrafICSnapshot(path, level, sample_fft_spacing=False)
    if host:
        mpi.msg("Done")

    div = np.array([float(i) for i in divisors(ic.header.N, mode='yield')])
    # Fix to ~patch_size Mpc/h/a per patch
    idx = np.abs((ic.header.N / div) * ic.header.dx - patch_size).argmin()
    ncubes = int(div[idx])

    if host:
        if ic.field_exists_on_disk('vbc') is False:
            mpi.msg("Writing vbc field")
            vbc = ic['vbc']
            ic.write_field(vbc, 'vbc', out_dir=ic.level_dir)

        if (field == 'rho') and (species == 'c') and (ic.field_exists_on_disk('deltac') is False):
            mpi.msg("Writing deltac field")
            deltac = ic['deltac']
            ic.write_field(deltac, 'deltac', out_dir=ic.level_dir)

        mpi.msg("Using %d cubes per dimension" % ncubes)
        mpi.msg("Grid size is %d^3" % ic.header.N)
        mpi.msg("Scattering domains...")

    cubes, dx = drift_velocity.cube_positions(ic, ncubes)

    # Scatter the cubes across MPI processes
    if host:
        chunks = np.array_split(cubes, size)
    else:
        chunks = None

    local_cubes = comm.scatter(chunks, root=0)
    mpi.msg("Received %d cubes" % len(local_cubes))
    del chunks, cubes

    ##########################################################################
    patches = []
    if debug or host:
        mpi.msg("Profiling...")
    num_local_cubes = float(len(local_cubes))
    i = 0.
    for cube in local_cubes:
        i += 1.
        if np.round((i / num_local_cubes) * 100.) % 10. == 0.:
            mpi.msg("%1.1f %%" % ((i / num_local_cubes) * 100.))

        modified_delta = apply_bias(ic, species, cube, dx, pad, field, ax=ax)
        patches.append(Patch(cube, dx, modified_delta).__dict__)

    ##########################################################################

    patches = np.array(patches)

    if host:
        global_patches = []
        global_patches.append(patches)
    else:
        global_patches = None
    comm.Barrier()

    for i in range(1, size):  # this is not desirable, however gather fails for large messages (>1024^3 grids distributed over N cores)
        if rank == i:
            if comm_debug:
                mpi.msg("Sending")
            comm.send(patches, dest=0, tag=rank)
        elif host:
            if comm_debug:
                mpi.msg("Receiving")
            patches = comm.recv(source=i, tag=i)
            global_patches.append(patches)
        comm.Barrier()

    if host:
        mpi.msg("Constructing biased field")
        delta_biased_global = np.zeros(ic.header.nn)
        for i in range(size):
            for patch in global_patches[i]:
                x_min, x_max, y_min, y_max, z_min, z_max = get_patch_bounds(patch)
                delta_biased_global[x_min:x_max,\
                    y_min: y_max,\
                    z_min: z_max] = patch['field']

        # Write out the new ICs
        out_level_dir = "%s/level_%03i/" % (out_base_dir, level)

        field_name = None
        if (field == 'vel'):
            field_name = 'vel%s%s' % (species, ax)
        else:
            field_name = 'delta%s' % (species)
        fname = ic.output_fname(field_name, out_base_dir)
        mpi.msg("Writing new ICs to file: %s" % fname)
        ic.write_field(delta_biased_global, field_name, out_dir=out_level_dir)
        mpi.msg("Complete")

if __name__ == "__main__":
    if (len(sys.argv) < 6):
        if (mpi.host):
            print "Usage: mpirun -np ${NSLOTS} %s </path/to/ics_ramses/> <level> <patch size [Mpc/h/a]> <species (b or c)> <field (rho or vel)>" % sys.argv[0]
            mpi.terminate(1)
    else:
        path = sys.argv[1]
        level = int(sys.argv[2])
        patch_size = float(sys.argv[3])
        species = sys.argv[4]
        field = sys.argv[5]
        ax = None
        if (field == 'vel'):
            if (len(sys.argv) == 7):
                ax = sys.argv[6]
            else:
                if (mpi.host):
                    print "Velocity Usage: mpirun -np ${NSLOTS} %s </path/to/ics_ramses/> <level> <patch size [Mpc/h/a]> <species (b or c)> <field (rho or vel)> <velocity axis (x,y,z)>" % sys.argv[0]
                    mpi.terminate(1)

        try:
            main(path, level, patch_size, species, field, ax=ax)
        except Exception as e:
           mpi.terminate(500, e)