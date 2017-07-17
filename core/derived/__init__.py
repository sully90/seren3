"""
Registry for derived fields
"""
from amr_derived import *
from part_derived import *
from dm_derived import *
from star_derived import *
from gmc_derived import *

import pynbody

@pynbody.derived_array
def deltab(sim):
    from seren3 import cosmology
    from seren3.array import SimArray

    omega_b_0 = 0.045
    cosmo = sim.properties.copy()
    cosmo["z"] = (1. / cosmo['a']) - 1.

    rho_mean = SimArray(cosmology.rho_mean_z(omega_b_0, **cosmo), "kg m**-3")
    rho = sim.g["rho"].in_units("kg m**-3")

    db = (rho-rho_mean)/rho_mean
    return db

@pynbody.derived_array
def nH(sim):
    from pymses.utils import constants as C
    from seren3.array import SimArray

    rho = sim.g["rho"]
    mH = SimArray(C.mH)
    X_fraction = 0.76
    H_frac = mH/X_fraction

    nH = rho/H_frac
    return nH

@pynbody.derived_array
def nHI(sim):
    return sim.g["nH"] * (1. - sim.g["xHII"])

@pynbody.derived_array
def rho_HI(sim):
    return sim.g["rho"] * (1. - sim.g["xHII"])

@pynbody.derived_array
def rho_HII(sim):
    return sim.g["rho"] * sim.g["xHII"]

@pynbody.derived_array
def mfx(sim):
    return sim.g["rho"].in_units("Msol km**-3") * sim.g["vx"].in_units("km s**-1")

@pynbody.derived_array
def mfy(sim):
    return sim.g["rho"].in_units("Msol km**-3") * sim.g["vy"].in_units("km s**-1")

@pynbody.derived_array
def mfz(sim):
    return sim.g["rho"].in_units("Msol km**-3") * sim.g["vz"].in_units("km s**-1")

@pynbody.derived_array
def mass_flux_radial(sim):
    import numpy as np
    from pynbody.array import SimArray
    from seren3.utils import unit_vec_r, heaviside

    flux = []
    units = None
    for i in 'xyz':
        #print i
        fi = sim.g["mf%s" % i]
        flux.append(fi)
        units = fi.units
    flux = np.array(flux).T

    x,y,z = sim.g["pos"].T
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta = sim.g["sg_theta"]
    # phi = sim.g["sg_az"]
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    mass_flux_scalar = np.zeros(len(theta))
    for i in range(len(theta)):
        th, ph = (theta[i], phi[i])
        unit_r = unit_vec_r(th, ph)
        mass_flux_scalar[i] = np.dot(flux[i], unit_r)

    return SimArray( mass_flux_scalar, units )

@pynbody.derived_array
def outflow_rate(sim):
    import numpy as np
    from pynbody.array import SimArray
    from seren3.utils import unit_vec_r, heaviside

    flux = []
    units = None
    for i in 'xyz':
        #print i
        fi = sim.g["mf%s" % i]
        flux.append(fi)
        units = fi.units
    flux = np.array(flux).T

    x,y,z = sim.g["pos"].T
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta = sim.g["sg_theta"]
    # phi = sim.g["sg_az"]
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    mass_flux_scalar = np.zeros(len(theta))
    for i in range(len(theta)):
        th, ph = (theta[i], phi[i])
        unit_r = unit_vec_r(th, ph)
        mass_flux_scalar[i] = np.dot(flux[i], unit_r)\
                * heaviside(np.dot(flux[i], unit_r))

    return SimArray( mass_flux_scalar, units )

def rad_radial(sim, group):
    import numpy as np
    from pynbody.array import SimArray
    from seren3.utils import unit_vec_r, heaviside

    flux = []
    units = None
    for i in 'xyz':
        #print i
        fi = sim.g["rad_%i_flux_%s" % (group, i)]
        flux.append(fi)
        units = fi.units
    flux = np.array(flux).T

    x,y,z = sim.g["pos"].T
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta = sim.g["sg_theta"]
    # phi = sim.g["sg_az"]
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    flux_scalar = np.zeros(len(theta))
    for i in range(len(theta)):
        th, ph = (theta[i], phi[i])
        unit_r = unit_vec_r(th, ph)
        # Compute outward flux (should always be positive)
        flux_scalar[i] = np.dot(flux[i], unit_r)\
                * heaviside(np.dot(flux[i], unit_r))

    return SimArray( flux_scalar, units )

@pynbody.derived_array
def rad_flux_radial(sim):
    return sim["rad_0_flux_radial"] + sim["rad_1_flux_radial"] + sim["rad_2_flux_radial"]

@pynbody.derived_array
def rad_0_flux_radial(sim):
    return rad_radial(sim, 0)

@pynbody.derived_array
def rad_1_flux_radial(sim):
    return rad_radial(sim, 1)

@pynbody.derived_array
def rad_2_flux_radial(sim):
    return rad_radial(sim, 2)

@pynbody.derived_array
def sg_theta(sim):
    import numpy as np
    from pynbody.array import SimArray

    x,y,z = sim["pos"].transpose()
    r = np.sqrt(x**2 + y**2 + z**2)
    return SimArray(np.arccos(z / r))

@pynbody.derived_array
def sg_az(sim):
    import numpy as np
    from pynbody.array import SimArray

    x,y,z = sim.g["pos"].transpose()
    return SimArray(np.arctan2(y, x))


