import seren3
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["epoch"], unit=C.Gyr)
def dm_age(context, dset, **kwargs):
    return part_age(context, dset, **kwargs)