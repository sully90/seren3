def fesc(subsnap, ret_flux_map=False, **kwargs):
    '''
    Computes halo escape fraction of hydrogen ionising photons
    '''
    import numpy as np
    from seren3.array import SimArray
    from seren3.analysis.render import render_spherical

    rvir = SimArray(subsnap.region.radius, subsnap.info["unit_length"])
    rt_c = SimArray(subsnap.info_rt["rt_c_frac"] * subsnap.C.c)
    dt = rvir / rt_c

    # Compute number of ionising photons from stars at time
    # rvir/rt_c (assumin halo is a point source)
    dset = subsnap.s[["Nion_d", "mass", "age"]].flatten(dt=dt)
    keep = np.where(dset["age"] - dt >= 0.)
    mass = dset["mass"][keep]
    nPhot = dset["Nion_d"] * mass

    # Computed integrated flux out of the virial sphere
    flux_map = render_spherical.render_quantity(subsnap.g, "rad_0_flux", units="s**-1 m**-2", ret_mag=False, filt=False, **kwargs)
    integrated_flux = render_spherical.integrate_surface_flux(flux_map, rvir)
    integrated_flux *= subsnap.info_rt["rt_c_frac"]  # scaled by reduced speed of light  -- is this right?

    # return the escape fraction
    if ret_flux_map:
        return nPhot.sum() / integrated_flux, flux_map
    return nPhot.sum() / integrated_flux