import seren3
from seren3.utils import constants
from pynbody.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    from pymses.utils import constants as C
    rho = dset["rho"].in_units("kg m**-3")
    mH = constants.from_pymses_constant(context.C.mH)
    H_cc = mH / 0.76  # Hydrogen mass fraction
    nH = (rho/H_cc).in_units("cm**-3")
    return nH

@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
def amr_nHe(context, dset):
    '''
    Return Helium number density
    '''
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    nHe = 0.25 * amr_nH(context, dset) * (Y_frac/X_frac)
    return result

@seren3.derived_quantity(requires=["xHII"], unit=C.none)
def amr_xHI(context, dset):
    '''
    Hydrogen neutral fraction
    '''
    val = 1. - dset['xHII']
    result = SimArray(val)
    return result

@seren3.derived_quantity(requires=["P", "rho"], unit=C.m/C.s)
def amr_cs(context, dset):
    '''
    Gas sound speed in m/s (units are convertabke)
    '''
    result = np.sqrt(1.66667 * dset['P'] / dset['rho'])
    return result

