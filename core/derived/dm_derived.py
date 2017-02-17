import seren3
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["epoch"])
def dm_age(context, dset, **kwargs):
    return part_age(context, dset, **kwargs)


@seren3.derived_quantity(requires=["pos"])
def dm_rho(context, dset, deconv_kernel=False, **kwargs):
    '''
    CIC dm density
    '''
    from seren3.utils.cython import cic

    L = 0  # boxsize in code units
    level = kwargs.get("level", 7)
    N = 2**level

    unit_l = context.array(context.info["unit_length"])
    pos = dset["pos"].in_units(unit_l)  # position in code units
    x,y,z = np.ascontiguousarray(pos.T)  # maintain C ordering
    rho = np.zeros(N**3)  # empty density array
    if hasattr(context.base, "region"):
        # Filtered dataset, normalise positions
        bbox = context.base.region.get_bounding_box()
        L = max(bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1], bbox[1][2] - bbox[0][2])
        offset = context.base.region.center - L/2.
        x-=offset[0]
        y-=offset[1]
        z-=offset[2]
    else:
        L = 1
    npart = len(x)

    # Do the CIC interpolation
    cic.cic(x,y,z,npart,L,N,rho)

    # Compute units
    boxmass = context.quantities.box_mass(species='cdm').in_units("kg")
    pm_mass = boxmass/npart

    boxsize = unit_l.in_units('m')
    dx = boxsize/N

    rhoc_unit = pm_mass/dx**3
    rho *= rhoc_unit

    # Low-level C I/O routines assemble data as a contiguous, C-ordered (nvars, twotondim, ngrids) numpy.ndarray
    # Swap data => shape : (ngrids, twotondim, nvars)
    ####### WARNING : must keep C-ordered contiguity !!! #######
    rho = np.ascontiguousarray(np.swapaxes(rho.reshape((N,N,N)), 0, 2))

    if deconv_kernel:
        from seren3.utils import deconvolve_cic
        print "Deconvolving CIC kernel"
        rho = deconvolve_cic(rho, N)
        # Correct for negaitve densities
        rho[np.where(rho < 0.)] = 0.
    return context.array(rho, "kg m**-3")
