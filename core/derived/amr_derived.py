import seren3
from seren3.core.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["rho"])
def amr_nH(context, dset):
    '''
    Return hydrogen number density
    '''
    from pymses.utils import constants as C
    nH = dset["rho"] * context.info["unit_density"].express(C.H_cc)
    result = SimArray(nH, 'cm**-3')
    result.set_field_name(r'n$_{\mathrm{H}}$')
    return result

@seren3.derived_quantity(requires=["xHII"])
def amr_xHI(context, dset):
    '''
    Hydrogen neutral fraction
    '''
    val = 1. - dset['xHII']
    unit = C.none
    result = SimArray(val * unit)
    result.set_field_name(r"x$_{\mathrm{HI}}$")
    return result

@seren3.derived_quantity(requires=["P", "rho"])
def amr_cs(context, dset):
    val = np.sqrt(1.66667 * dset['P'] / dset['rho'])
    unit = context.info['unit_velocity']
    result = SimArray(val * unit.express(context.C.m / context.C.s)\
        , 'm s**-1')
    result.set_field_name("cs")
    return result
