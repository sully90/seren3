'''
Provides functions to compute the (photon production rate-weighted) escape fraction
of halos (or subsnaps in general).
Uses the pynbody python module to create healpix maps
'''

def fesc(subsnap, do_multigroup=True, ret_flux_map=False, **kwargs):
    '''
    Computes halo escape fraction of hydrogen ionising photons
    '''
    import numpy as np
    from seren3.array import SimArray
    # from seren3.core.derived import star_Nion_d
    from seren3.utils import derived_utils
    from seren3.analysis.render import render_spherical

    rvir = SimArray(subsnap.region.radius, subsnap.info["unit_length"])
    rt_c = SimArray(subsnap.info_rt["rt_c_frac"] * subsnap.C.c)
    dt = rvir / rt_c

    nIons = subsnap.info_rt["nIons"]

    integrated_flux = 0.
    nPhot = 0.

    dset = subsnap.s[["mass", "age", "metal"]].flatten()
    keep = np.where(dset["age"].in_units("Gyr") - dt.in_units("Gyr") >= 0.)
    mass = dset["mass"][keep]

    star_Nion_d = derived.get_derived_field("star", "Nion_d")

    if do_multigroup:
        for ii in range(nIons):
            # Compute number of ionising photons from stars at time
            # t - rvir/rt_c (assuming halo is a point source)
            # Nion_d = subsnap.s["Nion_d"].flatten(group=ii+1, dt=dt)
            Nion_d = star_Nion_d(subsnap, dset, dt=dt, group=ii+1)
            nPhot += (Nion_d * mass).sum()

            # Compute integrated flux out of the virial sphere
            flux_map = render_spherical.render_quantity(subsnap.g, "rad_%i_flux" % ii, units="s**-1 m**-2", ret_mag=False, filt=False, **kwargs)
            integrated_flux += render_spherical.integrate_surface_flux(flux_map, rvir)# * subsnap.info_rt["rt_c_frac"]  # scaled by reduced speed of light  -- is this right?
    else:
        # Compute number of ionising photons from stars at time
        # t - rvir/rt_c (assuming halo is a point source)
        # dset = subsnap.s[["Nion_d", "mass", "age"]].flatten(group=1, dt=dt)
        Nion_d = star_Nion_d(subsnap, dset, dt=dt, group=ii+1)
        keep = np.where(dset["age"] - dt >= 0.)
        mass = dset["mass"][keep]
        nPhot += (Nion_d * mass).sum()

        # Compute integrated flux out of the virial sphere
        flux_map = render_spherical.render_quantity(subsnap.g, "rad_0_flux", units="s**-1 m**-2", ret_mag=False, filt=False, **kwargs)
        integrated_flux += render_spherical.integrate_surface_flux(flux_map, rvir)# * subsnap.info_rt["rt_c_frac"]  # scaled by reduced speed of light  -- is this right?

    # fesc = nPhot.sum() / integrated_flux
    fesc = integrated_flux / nPhot.sum()

    # return the escape fraction
    if ret_flux_map:
        return fesc, flux_map
    return fesc

def integrate_fesc(I1, I2, lbtime):
    from scipy.integrate import trapz
    return trapz(I1, lbtime) / trapz(I2, lbtime)

def time_integrated_fesc(halo, back_to_aexp, nside=2**3, return_data=True, **kwargs):
    '''
    Computes the time integrated escapte fraction across
    the history of the halo, a la Kimm & Cen 2014
    '''
    import numpy as np
    import random    
    from seren3.exceptions import NoParticlesException

    # Need to compute fesc(t) and \dot{Nion} at each snapshot
    catalogue = halo.base.halos(finder="ctrees")

    # dicts to store results
    fesc_dict = {}
    Nion_d_dict = {}
    age_dict = {}

    def _compute(h):
        dset = h.s[["Nion_d", "mass"]]
        fesc_h = fesc(h.subsnap, nside=nside, **kwargs)
        # if fesc_h > 1.:
            # fesc_h = random.uniform(0.9, 1.0)
        fesc_dict[h.base.ioutput] = fesc_h
        Nion_d_dict[h.base.ioutput] = (dset["Nion_d"] * dset["mass"]).sum()  # at t=0, not dt=rvir/c !!!
        age_dict[h.base.ioutput] = h.base.age

    # Compute fesc for this halo (snapshot)
    _compute(halo)

    # Iterate through the most-massive progenitor line
    for prog in catalogue.iterate_progenitors(halo, back_to_aexp=back_to_aexp):
        if len(prog.s) > 0.:
            _compute(prog)

    # I1/I2 = numerator/denominator to be integrated
    I1 = np.zeros(len(fesc_dict)); I2 = np.zeros(len(fesc_dict)); age_array = np.zeros(len(age_dict))

    # Populate the arrays
    for key, i in zip( sorted(fesc_dict.keys(), reverse=True), range(len(fesc_dict)) ):
        I1[i] = fesc_dict[key] * Nion_d_dict[key]
        I2[i] = Nion_d_dict[key]
        age_array[i] = age_dict[key]

    # Calculate lookback-time
    lbtime = halo.base.age - age_array

    # Integrate back in time for each snapshot
    tint_fesc_hist = np.zeros(len(lbtime))
    for i in xrange(len(tint_fesc_hist)):
        tint_fesc_hist[i] = integrate_fesc( I1[i:], I2[i:], lbtime[i:] )

    # fesc at each time step can be computed by taking I1/I2
    if return_data:    
        return tint_fesc_hist, I1, I2, lbtime
    return tint_fesc_hist
    