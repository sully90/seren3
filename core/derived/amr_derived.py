import seren3
from seren3.utils import constants
from seren3.utils.derived_utils import _get_field
from pynbody.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["rho", "dx"], unit=C.Msun)
def amr_mass(context, dset):
    '''
    Return cell mass in solar masses
    '''
    rho = _get_field(context, dset, "rho").in_units("Msol pc**-3")
    dx = _get_field(context, dset, "dx").in_units("pc")
    mass = rho * (dx**3)
    return mass

@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    from pymses.utils import constants as C
    rho = _get_field(context, dset, "rho").in_units("kg m**-3")
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
    val = 1. - _get_field(context, dset, "xHII")
    result = SimArray(val)
    return result

@seren3.derived_quantity(requires=["P", "rho"], unit=C.m/C.s)
def amr_cs(context, dset):
    '''
    Gas sound speed in m/s (units are convertabke)
    '''
    rho = _get_field(context, dset, "rho")
    P = _get_field(context, dset, "P")
    result = np.sqrt(1.66667 * P / rho)
    return result

@seren3.derived_quantity(requires=["P", "rho"], unit=C.K)
def amr_T2(context, dset):
    '''
    Gas Temperature in units of K/mu
    '''
    rho = _get_field(context, dset, "rho")
    P = _get_field(context, dset, "P")
    mH = constants.from_pymses_constant(context.C.mH)
    kB = constants.from_pymses_constant(context.C.kB)
    return P/rho * (mH / kB)
    
############################################### RAMSES-RT ###############################################
@seren3.derived_quantity(requires=["Np1", "Np2", "Np3"], unit=1./C.s)
def amr_Gamma(context, dset, iIon=0):
    '''
    Gas photoionization rate in [s^-1]
    '''
    from seren3.utils import constants

    emi = 0.
    for i in range(1, context.info_rt["nGroups"] + 1):
        Np = _get_field(context, dset, "Np%i" % i)
        csn = constants.from_pymses_constant(context.info_rt["group%i" % i]["csn"][iIon])
        emi += Np * csn

    return emi
