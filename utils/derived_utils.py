"""
Location of the derived field registry
"""

_derived_field_registry = {}

def derived_quantity(requires, latex):
    def wrap(fn):
        _derived_field_registry[fn.__name__] = fn
        _derived_field_registry["%s_required" % fn.__name__] = requires
        _derived_field_registry["%s_latex" % fn.__name__] = latex
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