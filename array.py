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

    # def __init__(self, array, ptr, **kwargs):
    #     super(SimArray, self).__init__(array, ptr)
    #     self._context = {}

    #     if "snapshot" in kwargs:
    #         cosmo = kwargs.pop("snapshot").cosmo
    #         self._context["h"] = cosmo["h"]
    #         self._context["a"] = cosmo["aexp"]

    def __new__(subtype, data, units=None, snapshot=None, **kwargs):
        if isinstance(data, pymses_Unit):
            units = data._decompose_base_units().replace("^", "**").replace(".", " ")
            data = data.coeff
        if isinstance(units, pymses_Unit):
            units = unit_string(units)

        new = np.array(data, **kwargs).view(subtype)
        new._context = {}
        if hasattr(data, 'units') and hasattr(data, 'snapshot') and units is None and sim is None:
            units = data.units
            snapshot = data.snapshot

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

        if snapshot is not None:
            new.snapshot = snapshot.ancestor            
            new._context["h"] = snapshot.cosmo["h"]
            new._context["a"] = snapshot.cosmo["aexp"]
            # will generate a weakref automatically

        new._name = None

        return new

    def in_units(self, new_unit, **context_overrides):
        """Return a copy of this array expressed relative to an alternative
        unit."""

        context = self.conversion_context()
        context.update(context_overrides)

        if isinstance(new_unit, SimArray):
            new_unit = "{val} {unit}".format(val=str(new_unit), unit=str(new_unit.units))

        if self.units is not None:
            r = self * self.units.ratio(new_unit,
                                        **context)
            r.units = new_unit
            return r
        else:
            raise ValueError, "Units of array unknown"

    def conversion_context(self):
        return self._context

    def add_conversion_context(self, key, value):
        self._context[key] = value

    def remove_conversion_context(self, key):
        if key in self._context:
            del self._context[key]
        else:
            raise Exception("No entry with key: %s" % key)