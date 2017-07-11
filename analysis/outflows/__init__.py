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


def mass_flux_hist(halo, back_to_aexp, return_data=True):
    '''
    Compute history of in/outflows
    '''
    import numpy as np
    from seren3.scripts.mpi import write_mass_flux_hid_dict

    db = write_mass_flux_hid_dict.load_db(halo.base.path, halo.base.ioutput)
    if (int(halo["id"]) in db.keys()):
        catalogue = halo.base.halos(finder="ctrees")

        F = []
        age_arr = []
        hids = []

        def _compute(h, db):
            hid = int(h["id"])
            res = db[hid]

            F.append(res["F"])
            age_arr.append(h.base.age)
            hids.append(hid)

        _compute(halo, db)

        for prog in catalogue.iterate_progenitors(halo, back_to_aexp=back_to_aexp):
            db = write_mass_flux_hid_dict.load_db(prog.base.path, prog.base.ioutput)

            if (int(prog["id"]) in db.keys()):
                _compute(prog, db)
            else:
                break
        F = np.array(F)
        age_arr = np.array(age_arr)
        hids = np.array(hids)

        lbtime = halo.base.age - age_arr

        if return_data:    
            return F, age_arr, lbtime, hids
        return F
    else:
        return None