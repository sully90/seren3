"""
Location of the derived field registry
"""

_derived_field_registry = {}

def derived_quantity(requires):
    def wrap(fn):
        _derived_field_registry[fn.__name__] = fn
        _derived_field_registry["%s_required" % fn.__name__] = requires
        return fn
    return wrap

def add_derived_quantity(fn, requires):
    _derived_field_registry[fn.__name__] = fn
    _derived_field_registry["%s_required" % fn.__name__] = requires

def required_for_field(family, field):
    return _derived_field_registry["%s_%s_required" % (family, field)]

def is_derived(family, field):
    return "%s_%s" % (family, field) in _derived_field_registry

def get_derived_field(family, field):
    return _derived_field_registry["%s_%s" % (family, field)]

def get_derived_field_latex(family, field):
    return _derived_field_registry["%s_%s_latex" % (family, field)]

def LambdaOperator(family, field, power=1., vol_weighted=False):
    '''
    Return a lambda function for this field
    '''
    # If this is a derived field then grab the approp. function
    if is_derived(family.family, field):
        fn = get_derived_field(family.family, field)
        if vol_weighted:
            op = lambda dset: fn(family.base, dset)**power * dset.get_sizes()**3
        else:
            op = lambda dset: fn(family.base, dset)**power
    else:
        if vol_weighted:
            op = lambda dset: dset[field]**power * dset.get_sizes()**3
        else:
            op = lambda dset: dset[field]**power
    return op