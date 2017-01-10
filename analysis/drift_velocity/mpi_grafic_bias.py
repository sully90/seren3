import numpy as np

VERBOSE = True

class Patch(object):

    def __init__(self, patch, dx, field):
        self.patch = patch
        self.dx = dx
        self.field = field

def main(path, level, patch_size):
    '''
    Writes a new set of grafIC initial conditions with a drift velocity dependent
    bias in the power spectrum
    '''
    from seren3.core import grafic_snapshot
    from seren3.analysis import drift_velocity as vbc_utils
    from seren3.analysis.parallel import mpi
    from seren3.utils import divisors

    mpi.msg("Loading initial conditions")
    ics = grafic_snapshot.load_snapshot(path, level, sample_fft_spacing=False)

    if mpi.host:
        # Make sure vbc field exists on disk
        if not ics.field_exists_on_disk("vbc"):
            ics.write_field(ics["vbc"], "vbc")

    div = np.array([float(i) for i in divisors(ics.header.N, mode='yield')])
    idx = np.abs((ics.header.N / div) * ics.dx - patch_size).argmin()
    ncubes = int(div[idx])

    # Compute cube positions in cell units
    cubes = dx = None
    if mpi.host:
        mpi.msg("Using %i cubes per dimension." % ncubes)
        cubes, dx = vbc_utils.cube_positions(ics, ncubes)
        cubes = np.array(cubes)
    dx = mpi.comm.bcast(dx, root=0)
    pad = 8


############################## WORK LOOP ######################################

    # Iterate over patch positions in parallel
    dest = {}
    for patch, sto in mpi.piter(cubes, storage=dest, print_stats=True):
        origin = np.array(patch - float(dx) / 2. - pad, dtype=np.int64)
        dx_eps = float(dx) + float(2 * pad)

        delta = vbc = None
        if (VERBOSE): mpi.msg("Loading patch: %s" % patch)
        delta = ics.lazy_load_periodic("deltab", origin, int(dx_eps))
        vbc = ics.lazy_load_periodic("vbc", origin, int(dx_eps))

        # Compute the bias
        if (VERBOSE): mpi.msg("Computing bias")
        k_bias, b_cdm, b_b = vbc_utils.compute_bias(ics, vbc)

        # Convolve with field
        if (VERBOSE): mpi.msg("Performing convolution")
        delta_biased = vbc_utils.apply_density_bias(ics, k_bias, b_b, delta.shape[0], delta_x=delta)

        # Remove the padded region
        x_shape, y_shape, z_shape = delta_biased.shape
        delta_biased = delta_biased[0 + pad:x_shape - pad,
                                        0 + pad:y_shape - pad, 0 + pad:z_shape - pad]

        # Store
        biased_patch = Patch(patch, dx, delta_biased)
        sto.result = biased_patch.__dict__

############################## END OF WORK LOOP ###############################

    if mpi.host:
        import os
        # Write new ICs
        dest = mpi.unpack(dest)

        output_field = np.zeros(ics.header.nn)

        for item in dest:
            result = item.result
            patch = result["patch"]
            dx = result["dx"]
            delta_biased = result["field"]

            # Bounds of this patch
            x_min, x_max = (int((patch[0]) - (dx / 2.)), int((patch[0]) + (dx / 2.)))
            y_min, y_max = (int((patch[1]) - (dx / 2.)), int((patch[1]) + (dx / 2.)))
            z_min, z_max = (int((patch[2]) - (dx / 2.)), int((patch[2]) + (dx / 2.)))

            # Place into output
            output_field[x_min:x_max, y_min:y_max, z_min:z_max] = delta_biased

        # Write the initial conditions
        ics_dir = "%s/ics_ramses_vbc/" % ics.level_dir
        if not os.path.isdir(ics_dir):
            os.mkdir(ics_dir)
        out_dir = "%s/level_%03i/" % (ics_dir, level)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        ics.write_field(output_field, "deltab", out_dir=out_dir)

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    level = int(sys.argv[2])
    patch_size = float(sys.argv[3])

    try:
        main(path, level, patch_size)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        mpi.msg("Caught exception: %s" % e.message)
        mpi.terminate(500, e=e)