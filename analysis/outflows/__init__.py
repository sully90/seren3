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
    from seren3.array import SimArray
    from seren3.analysis.render import render_spherical

    rvir = SimArray(subsnap.region.radius, subsnap.info["unit_length"])
    in_units = "kg s**-1 m**-2"
    s = subsnap.pynbody_snapshot(filt=filt)

    im = render_spherical.render_quantity(subsnap.g, "mass_flux_radial", s=s, in_units=in_units, out_units=in_units, **kwargs)
    im.convert_units("Msol yr**-1 kpc**-2")

    dm_dt = integrate_surface_flux(im, rvir)
    return dm_dt, im