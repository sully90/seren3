import seren3
from seren3.core.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["P", "rho"])
def amr_cs(context, dset):
    val = np.sqrt(1.66667 * dset['P'] / dset['rho'])
    unit = context.info['unit_velocity']
    result = SimArray(val * unit.express(context.C.m / context.C.s)\
        , 'm s**-1')
    result.set_field_name("cs")
    return result
