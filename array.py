import numpy as np
from seren3.utils.constants import unit_string
from pynbody import array
from pynbody import units as units
_units = units
from pymses.utils.constants.unit import Unit as pymses_Unit

class SimArray(array.SimArray):
    '''
    Wrapper to support use of pymses units
    '''
    def __new__(subtype, data, units=None, sim=None, **kwargs):
        if isinstance(data, pymses_Unit):
            units = data._decompose_base_units().replace("^", "**").replace(".", " ")
            data = data.coeff
        if isinstance(units, pymses_Unit):
            units = unit_string(units)

        new = np.array(data, **kwargs).view(subtype)
        if hasattr(data, 'units') and hasattr(data, 'sim') and units is None and sim is None:
            units = data.units
            sim = data.sim

        if hasattr(data, 'family'):
            new.family = data.family

        if isinstance(units, str):
            units = _units.Unit(units)

        new._units = units

        # Always associate a SimArray with the top-level snapshot.
        # Otherwise we run into problems with how the reference should
        # behave: we don't want to lose the link to the simulation by
        # storing a weakref to a SubSnap that might be deconstructed,
        # but we also wouldn't want to store a strong ref to a SubSnap
        # since that would keep the entire simulation alive even if
        # deleted.
        #
        # So, set the sim attribute to the top-level snapshot and use
        # the normal weak-reference system.

        if sim is not None:
            new.sim = sim.ancestor
            # will generate a weakref automatically

        new._name = None

        return new
