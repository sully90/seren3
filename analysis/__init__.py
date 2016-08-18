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

def time_integrated_fesc(halo, back_to_iout, return_data=False):
    '''
    Computes the time integrated escapte fraction across
    the history of the halo, a la Kimm & Cen 2014
    '''
    import numpy as np
    from scipy.integrate import trapz
    import random

    # Need to compute fesc(t) and \dot{Nion} at each snapshot
    catalogue = halo.catalogue
    fesc_dict = {}
    Nion_d_dict = {}
    age_dict = {}

    # This snapshot
    def _compute(h):
        dset = h.s[["Nion_d", "mass"]]
        fesc_h = fesc(h.subsnap)
        if fesc_h > 1.:
            fesc_h = random.uniform(0.9, 1.0)
        fesc_dict[h.base.ioutput] = fesc_h
        Nion_d_dict[h.base.ioutput] = (dset["Nion_d"] * dset["mass"]).sum()  # at t=0, not dt=rvir/c !!!
        age_dict[h.base.ioutput] = h.base.age
    
    _compute(halo)
    for prog in catalogue.iterate_progenitors(halo, back_to_iout=back_to_iout):
        _compute(prog)

    I1 = np.zeros(len(fesc_dict)); I2 = np.zeros(len(fesc_dict)); age_array = np.zeros(len(age_dict))

    for key, i in zip( sorted(fesc_dict.keys(), reverse=True), range(len(fesc_dict)) ):
        I1[i] = fesc_dict[key]*Nion_d_dict[key]
        I2[i] = Nion_d_dict[key]
        age_array[i] = age_dict[key]

    lbtime = halo.base.age - age_array

    result = trapz(I1, lbtime) / trapz(I2, lbtime)
    if return_data:
        return result, I1, I2, lbtime
    return result