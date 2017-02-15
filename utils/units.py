'''
Routines to handle unit conversion
'''
def tracked_field_in_unit_system(family, dset, dims):
    '''
    Converts all fields in a dictionary to a given unit system
    '''
    from seren3.utils import derived_utils

    output_dset = {}
    for field in dset.fields:
        unit = derived_utils.get_field_unit(family, field)
        unit = family.array(unit)
        new_unit = in_unit_system(unit, dims)
        output_dset[field] = family.array(dset[field], unit).in_units(new_unit)

    return output_dset


def in_unit_system(unit, dims):
    from seren3.array import units

    if not hasattr(dims, "__iter__"):
        dims = [dims]
    if type(dims[0]) == str:
        dims = [units.Unit(x) for x in dims]

    new_unit = unit.dimensional_project(dims)
    new_unit = reduce(lambda x,y: x * y, [a ** b for a,b in zip(dims, new_unit[:3])])
    return new_unit