import seren3
from seren3.utils import constants
# from seren3.utils.derived_utils import _get_field
from seren3.array import SimArray
import numpy as np
from pymses.utils import constants as C

def check_dset(derived_fn):
    '''
    Ensures tracked fields always have unit information
    '''
    def _check_dset(context, dset, **kwargs):
        parsed_dset = {}
        for field in dset:
            print field
            if not isinstance(dset[field], SimArray):
                field_info = seren3.info_for_tracked_field(field)
                unit_key = field_info["info_key"]

                unit = context.info[unit_key]
                parseddset[field] = SimArray(field, unit)

                if "default_unit" in field_info:
                    dset[field] = dset[field].in_units(field_info["default_unit"])
            else:
                parsed_dset[field] = dset[field]

            return derived_fn(context, dset, **kwargs)
    return _check_dset

@check_dset
@seren3.derived_quantity(requires=["rho", "dx"], unit=C.Msun)
def amr_mass(context, dset):
    '''
    Return cell mass in solar masses
    '''
    rho = dset["rho"].in_units("Msol pc**-3")
    dx = dset["dx"].in_units("pc")
    mass = rho * (dx**3)
    return mass

@check_dset
@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    rho = dset["rho"].in_units("kg m**-3")
    mH = SimArray(context.C.mH)
    H_cc = mH / 0.76  # Hydrogen mass fraction
    nH = (rho/H_cc).in_units("cm**-3")
    return nH

@check_dset
@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
def amr_nHe(context, dset):
    '''
    Return Helium number density
    '''
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    nHe = 0.25 * amr_nH(context, dset) * (Y_frac/X_frac)
    return result

@check_dset
@seren3.derived_quantity(requires=["xHII"], unit=C.none)
def amr_xHI(context, dset):
    '''
    Hydrogen neutral fraction
    '''
    val = 1. - dset["xHII"]
    result = SimArray(val)
    return result

@check_dset
@seren3.derived_quantity(requires=["P", "rho"], unit=C.m/C.s)
def amr_cs(context, dset):
    '''
    Gas sound speed in m/s (units are convertabke)
    '''
    rho = dset["rho"]
    P = dset["P"]
    result = np.sqrt(1.66667 * P / rho).in_units("m s**-1")
    return result

@check_dset
@seren3.derived_quantity(requires=["P", "rho"], unit=C.K)
def amr_T2(context, dset):
    '''
    Gas Temperature in units of K/mu
    '''
    rho = dset["rho"]
    P = dset["P"]
    mH = SimArray(context.C.mH)
    kB = SimArray(context.C.kB)
    # mH = constants.from_pymses_unit(context.C.mH)
    # kB = constants.from_pymses_unit(context.C.kB)
    return (P/rho * (mH / kB)).in_units("K")
    
############################################### RAMSES-RT ###############################################
@check_dset
@seren3.derived_quantity(requires=["Np1", "Np2", "Np3"], unit=1./C.s)
def amr_Gamma(context, dset, iIon=0):
    '''
    Gas photoionization rate in [s^-1]
    '''
    from seren3.utils import constants

    emi = 0.
    for i in range(1, context.info_rt["nGroups"] + 1):
        Np = dset["Np%i" % i]
        csn = SimArray(context.info_rt["group%i" % i]["csn"][iIon])
        emi += Np * csn

    return emi
