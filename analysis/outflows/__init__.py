def integrate_surface_flux(flux_map, r):
    '''
    Integrates a healpix surface flux to compute the total
    net flux out of the sphere.
    r is the radius of the sphere in meters
    '''
    import numpy as np
    import healpy as hp
    from scipy.integrate import trapz
    from seren3.array import SimArray

    if not ((isinstance(flux_map, SimArray) or isinstance(r, SimArray))):
        raise Exception("Must pass SimArrays")

    # Compute theta/phi
    npix = len(flux_map)
    nside = hp.npix2nside(npix)
    # theta, phi = hp.pix2ang(nside, range(npix))
    theta, phi = hp.pix2ang(nside, range(npix))
    r = r.in_units("kpc")  # make sure r is in meters

    # Compute the integral
    integrand = np.zeros(len(theta))

    for i in range(len(theta)):
        th, ph = (theta[i], phi[i])
        integrand[i] = r**2 * np.sin(th) * flux_map[i]  # mass_flux_radial function already deals with unit vev

    integrand = integrand[:, None] + np.zeros(len(phi))  # 2D over theta and phi
    I = trapz(trapz(integrand, phi), theta)
    return SimArray(I, "Msol yr**-1")


def dm_by_dt(subsnap, filt=True, **kwargs):
    '''
    Compute mass flux at the virial sphere
    '''
    import numpy as np
    from seren3.array import SimArray
    from seren3.analysis.render import render_spherical

    reload(render_spherical)

    rvir = SimArray(subsnap.region.radius, subsnap.info["unit_length"])
    in_units = "kg s**-1 m**-2"
    s = subsnap.pynbody_snapshot(filt=filt)

    im = render_spherical.render_quantity(subsnap.g, "mass_flux_radial", s=s, in_units=in_units, out_units=in_units, **kwargs)
    im.convert_units("Msol yr**-1 kpc**-2")

    def _compute_flux(im, rvir, direction=None):
        im_tmp = im.copy()
        ix = None
        if ("out" == direction):
            ix = np.where(im_tmp < 0)
            im_tmp[ix] = 1e-12
        elif ("in" == direction):
            ix = np.where(im_tmp > 0)
            im_tmp[ix] = -1e-12
        else:
            return integrate_surface_flux(im, rvir)    

        return integrate_surface_flux(im_tmp, rvir)

    F = _compute_flux(im, rvir)
    F_plus = _compute_flux(im, rvir, direction="out")
    F_minus = _compute_flux(im, rvir, direction="in")
    return (F, F_plus, F_minus), im


def integrate_dm_by_dt(I1, I2, lbtime):
    from scipy.integrate import trapz
    return trapz(I1, lbtime) / trapz(I2, lbtime)


def time_integrated_dm_by_dt(halo, back_to_aexp, return_data=True, **kwargs):
    '''
    Computes the time integrated escapte fraction across
    the history of the halo, a la Kimm & Cen 2014
    '''
    import random
    import numpy as np
    from seren3.scripts.mpi import write_mass_flux_hid_dict
    from seren3.utils import derived_utils
    from seren3.core.serensource import DerivedDataset

    star_Nion_d_fn = derived_utils.get_derived_field(halo.s, "Nion_d")
    nIons = halo.base.info_rt["nIons"]

    db = write_mass_flux_hid_dict.load_db(halo.base.path, halo.base.ioutput)
    if (int(halo["id"]) in db.keys()):
        catalogue = halo.base.halos(finder="ctrees")

        # dicts to store results
        dm_by_dt_dict = {}
        Nphoton_dict = {}
        age_dict = {}

        def _compute(h, db):
            hid = int(h["id"])
            result = db[hid]

            dm_by_dt_h = result["out_dm_by_dt"]

            star_dset = h.s[["age", "metal", "mass"]].flatten()
            star_mass = star_dset["mass"]
            star_age = star_dset["age"]
            star_metal = star_dset["metal"]

            dict_stars = {"age" : star_age, "metal" : star_metal, "mass" : star_mass}
            dset_stars = DerivedDataset(h.s, dict_stars)

            Nion_d_all_groups = h.base.array(np.zeros(len(dset_stars["age"])), "s**-1 Msol**-1")

            for ii in range(nIons):
                Nion_d_now = star_Nion_d_fn(h.base, dset_stars, group=ii+1, dt=0.)
                Nion_d_all_groups += Nion_d_now

            Nphotons = (Nion_d_all_groups.in_units("s**-1 Msol**-1") * star_mass.in_units("Msol")).sum()

            dm_by_dt_dict[h.base.ioutput] = dm_by_dt_h
            Nphoton_dict[h.base.ioutput] = Nphotons # at t=0, not dt=rvir/c !!!
            age_dict[h.base.ioutput] = h.base.age

        # Compute dm_by_dt for this halo (snapshot)
        _compute(halo, db)
        # Iterate through the most-massive progenitor line
        for prog in catalogue.iterate_progenitors(halo, back_to_aexp=back_to_aexp):
            db = write_mass_flux_hid_dict.load_db(prog.base.path, prog.base.ioutput)
            # print prog

            if (int(prog["id"]) in db.keys()):
                _compute(prog, db)
            else:
                break

        # I1/I2 = numerator/denominator to be integrated
        I1 = np.zeros(len(dm_by_dt_dict)); I2 = np.zeros(len(dm_by_dt_dict)); age_array = np.zeros(len(age_dict))

        # Populate the arrays
        for key, i in zip( sorted(dm_by_dt_dict.keys(), reverse=True), range(len(dm_by_dt_dict)) ):
            I1[i] = dm_by_dt_dict[key] * Nphoton_dict[key]
            I2[i] = Nphoton_dict[key]
            age_array[i] = age_dict[key]

        # Calculate lookback-time
        lbtime = halo.base.age - age_array

        # Integrate back in time for each snapshot
        tint_dm_by_dt_hist = np.zeros(len(lbtime))
        for i in xrange(len(tint_dm_by_dt_hist)):
            tint_dm_by_dt_hist[i] = integrate_dm_by_dt( I1[i:], I2[i:], lbtime[i:] )

        # dm_by_dt at each time step can be computed by taking I1/I2
        if return_data:    
            return tint_dm_by_dt_hist, I1, I2, lbtime
        return tint_dm_by_dt_hist
    else:
        return None