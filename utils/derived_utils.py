"""
Location of the derived field registry
"""
import seren3
from seren3.array import SimArray
from seren3.core.serensource import DerivedDataset

_derived_field_registry = {}  # Automatically filled by annotations
_pynbody_to_pymses_registry = {"Msol" : "Msun"}

def pymses_units(unit_string):
    '''
    Returns pymses compatible units from a string
    '''
    import numpy as np
    from pymses.utils import constants as C
    unit = 1.
    compontents = str(unit_string).split(' ')
    for c in compontents:
        if '**' in c:
            dims = c.split('**')
            pymses_unit = np.power(C.Unit(dims[0]), float(dims[1]))
            unit *= pymses_unit
        else:
            if c in _pynbody_to_pymses_registry:
                c = _pynbody_to_pymses_registry[c]
            unit *= C.Unit(c)
    return unit

def derived_quantity(requires):
    def _check_dset(fn, context, dset, **kwargs):
        '''
        Ensures a DerivedDataset is also handed off to the worker function
        '''
        if not isinstance(dset, DerivedDataset):
            dset = DerivedDataset(context, dset)
        return fn(context, dset, **kwargs)
    def wrap(fn):
        _derived_field_registry[fn.__name__] = lambda context, dset, **kwargs: _check_dset(fn, context, dset, **kwargs)
        _derived_field_registry["%s_required" % fn.__name__] = requires
        # _derived_field_registry["%s_unit" % fn.__name__] = unit
        return fn
    return wrap


def add_derived_quantity(fn, requires):
    _derived_field_registry[fn.__name__] = fn
    _derived_field_registry["%s_required" % fn.__name__] = requires


def required_for_field(family, field):
    return _derived_field_registry["%s_%s_required" % (family.family, field)]


def is_derived(family, field):
    return "%s_%s" % (family.family, field) in _derived_field_registry


def get_derived_field(family, field):
    return _derived_field_registry["%s_%s" % (family.family, field)]
