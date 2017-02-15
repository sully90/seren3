'''
Modules to handle images with pymses
'''
import seren3
from enum import Enum


class ImageMode(Enum):
    SPLATTING = 1
    RAYTRACING = 2
    SLICE = 3


def get_operator(family, field, mode, out_units=None):
    '''
    Returns an appropriate operator for the desired projection method
    '''
    from seren3.array import units
    from seren3.utils import units as unit_utils
    from seren3.core.serensource import DerivedDataset
    from pymses.analysis import ScalarOperator, FractionOperator

    # Look up the base unit for this field
    base_unit = seren3.get_field_unit(family, field)  # string
    if type(base_unit) != str:
        raise Exception("Detected deprecated unit information for field (%s, %s). Should be string." % (family, field))


    def field_fn(family, dset, field):
        field_unit = units.Unit(seren3.get_field_unit(family, field))
        new_unit = unit_utils.in_unit_system(DerivedDataset.BASE_UNITS)
        return DerivedDataset(family, dset)[field].in_units(new_unit)

    # Make sure we are consistent with the DerivedDataset unit base
    # If this is an intensive variable (i.e density), and we are using the 
    # splatting method, we need to convert to extensive (and vise versa for ray tracing)
    op = None
    if (mode == ImageMode.SPLATTING) and (base_unit.dimensions[0] != 0):
        # Return a mass-weighted density operator
        up_fn = lambda dset: field_fn(field)**2 * field_fn("dx")**3
        up_fn = lambda dset: field_fn(field) * field_fn("dx")**3




def get_operator(family, field, extensive=True):
    from pymses.analysis import ScalarOperator

    func = lambda dset: dset[field]
    if seren3.is_derived(family, field):
        fn = seren3.get_derived_field(family, field)
        func = lambda dset: fn(family, dset)

    base_unit = seren3.get_field_unit(family, field)
    if extensive is True and base_unit.dimensions[0] != 0:
        '''
        Extensive variable requested but intensive variable provided
        '''
        return ScalarOperator(lambda dset: func(dset) * dset.get_sizes()**3, base_unit*family.C.m)
    else:
        return ScalarOperator(func, base_unit*family.C.m)



def image(family, field, mode="splatting", camera=None, op=None, out_units=None, **kwargs):
    '''
    Makes an image of the desired field.
    Determines how to handle intensive/extensive quantities
    '''
    from seren3.core.snapshot import Family
    from seren3.utils.derived_utils import pymses_units

    if not isinstance(family, Family):
        raise Exception("Require Family specific snapshot, got %s. Use snap.g, .d or .s" % family) 

    if camera is None:
        camera = family.camera()

    process = None
    if mode == "splatting":
        from pymses.analysis import splatting

        kwargs["random_shift"] = True
        if op is None:
            op = get_operator(family, field, extensive=True)
            
        disable_multi_processing = kwargs.pop("disable_multi_processing", False)
        source = family[field].pymses_source
        sp = splatting.SplatterProcessor(source, family.ro.info, op)
        if disable_multi_processing:
           sp.disable_multiprocessing()
        process = sp.process
    else:
        raise Exception("Unknown mode: %s" % mode)

    return process(camera, **kwargs)