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