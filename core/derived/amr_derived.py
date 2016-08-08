import seren3
from seren3.utils import constants
from seren3.utils.derived_utils import check_dset
from seren3.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["pos"], unit=C.Msun)
@check_dset
def amr_r(context, dset, center=None):
    '''
    Radial position
    '''
    pos = dset["pos"]

    if center is not None:
        pos -= center
    return ((pos ** 2).sum(axis=1)) ** (1,2)

@seren3.derived_quantity(requires=["rho", "dx"], unit=C.Msun)
@check_dset
def amr_mass(context, dset):
    '''
    Return cell mass in solar masses
    '''
    rho = dset["rho"].in_units("Msol pc**-3")
    dx = dset["dx"].in_units("pc")
    mass = rho * (dx**3)
    return mass

@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
@check_dset
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    rho = dset["rho"].in_units("kg m**-3")
    mH = SimArray(context.C.mH)
    H_cc = mH / 0.76  # Hydrogen mass fraction
    nH = (rho/H_cc).in_units("cm**-3")
    return nH

@seren3.derived_quantity(requires=["rho"], unit=C.H_cc)
@check_dset
def amr_nHe(context, dset):
    '''
    Return Helium number density
    '''
    X_frac, Y_frac = (context.info['X_fraction'], context.info['Y_fraction'])
    nHe = 0.25 * amr_nH(context, dset) * (Y_frac/X_frac)
    return result.in_units("cm**-3")

@seren3.derived_quantity(requires=["xHII"], unit=C.none)
@check_dset
def amr_xHI(context, dset):
    '''
    Hydrogen neutral fraction
    '''
    val = 1. - dset["xHII"]
    result = SimArray(val)
    return result

@seren3.derived_quantity(requires=["P", "rho"], unit=C.m/C.s)
@check_dset
def amr_cs(context, dset):
    '''
    Gas sound speed in m/s (units are convertabke)
    '''
    rho = dset["rho"]
    P = dset["P"]
    result = np.sqrt(1.66667 * P / rho).in_units("m s**-1")
    return result.in_units("m s**-1")

@seren3.derived_quantity(requires=["P", "rho"], unit=C.K)
@check_dset
def amr_T2(context, dset):
    '''
    Gas Temperature in units of K/mu
    '''
    rho = dset["rho"]
    P = dset["P"]
    mH = SimArray(context.C.mH)
    kB = SimArray(context.C.kB)
    T2 = (P/rho * (mH / kB))
    return T2.in_units("K")
    
############################################### RAMSES-RT ###############################################
@seren3.derived_quantity(requires=["Np1", "Np2", "Np3"], unit=1./C.s)
@check_dset
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

    return emi.in_units("s**-1")
