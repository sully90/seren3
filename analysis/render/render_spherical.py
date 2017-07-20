import pynbody
import numpy as np
from seren3.utils import unit_vec_r, heaviside

def render_quantity(family, qty, in_units=None, s=None, nside=2**5, kernel=pynbody.sph.Kernel(), filt=True, ret_mag=False, **kwargs):
    '''
    Renders a quantity on a healpix surface using pynbody.
    Must supply a subsnapshot
    '''
    import numpy as np
    from seren3.array import SimArray
    from seren3.core.snapshot import Family

    from pynbody import sph
    reload(sph)

    if not isinstance(family, Family):
        raise Exception("Must supply a Family level snapshot")

    _pymses_to_pynbody_family = {"amr" : "g", "dm" : "d", "star" : "s"}

    if (s is None):
        s = family.base.pynbody_snapshot(filt=filt)  # centered and filtered to virial sphere
    s.physical_units()  # change unit system

    # print "Rotating..."
    # pynbody.analysis.angmom.faceon(s.g)

    # family level specific pynbody snapshot
    s_family = getattr(s, _pymses_to_pynbody_family[family.family])

    # Radius from subsnap center to the healpix surface
    radius = kwargs.pop("radius", SimArray(family.base.region.radius, family.info["unit_length"]).in_units("kpc"))
    # print "radius = ", radius

    if (isinstance(radius, SimArray) is False):
        raise Exception("Radius must be a SimArray")

    radius.convert_units("kpc")
    # radius = "%f kpc" % (radius)

    if in_units is not None:
        s_family[qty].convert_units(in_units)
    out_units = kwargs.pop("out_units", None)

    ndim = len(s_family[qty].shape)

    if ("denoise" not in kwargs):
        kwargs["denoise"] = False
    kwargs["threaded"] = False
    kwargs["kernel"] = kernel
    if ndim == 1:
        # Scalar
        im = sph.render_spherical_image(s_family, qty=qty, \
                 distance=radius, out_units=out_units, nside=nside, **kwargs)
        return im
    else:
        # Vector
        im = {}
        for i in 'xyz':
            qty_i = "%s_%s" % (qty, i)
            im[i] = sph.render_spherical_image(s_family, qty=qty_i, \
                 distance=radius, out_units=out_units, nside=nside, **kwargs)
        if ret_mag:
            return np.sqrt( im['x']**2 + im['y']**2 + im['z']**2 )
        return SimArray( [im['x'], im['y'], im['z']], in_units, dtype=np.float32 ).T

def integrate_surface_flux(flux_map, r, smooth=False, ret_map=False, **smooth_kwargs):
    '''
    Integrates a healpix surface flux to compute the total
    net flux out of the sphere.
    r is the radius of the sphere in meters
    '''
    import healpy as hp
    from scipy.integrate import trapz
    from seren3.array import SimArray

    raise Exception("Function deprecated")

    if not ((isinstance(flux_map, SimArray) or isinstance(r, SimArray))):
        raise Exception("Must pass SimArrays")

    # Compute theta/phi
    npix = len(flux_map)
    nside = hp.npix2nside(npix)
    # theta, phi = hp.pix2ang(nside, range(npix))
    theta, phi = hp.pix2ang(nside, range(npix))
    r = r.in_units("m")  # make sure r is in meters

    # Smoothing?
    if smooth:
        flux_map = hp.smoothing(flux_map, **smooth_kwargs)

    # Compute the integral
    integrand = np.zeros(len(theta))

    for i in range(len(theta)):
        th, ph = (theta[i], phi[i])
        unit_r = unit_vec_r(th, ph)
        integrand[i] = r**2 * np.sin(th)\
         * np.dot(flux_map[i], unit_r)\
         * heaviside(np.dot(flux_map[i], unit_r))

    integrand = integrand[:, None] + np.zeros(len(phi))  # 2D over theta and phi

    I = trapz(trapz(integrand, phi), theta)

    return SimArray(I, "s**-1")
