import seren3

def lambda_function(family, field, vol_weighted, power):
    '''
    Returns a lambda function to derive this field for projections
    '''
    from seren3.core.serensource import DerivedDataset
    derived_dset = DerivedDataset(family, dset)

    def _fn(family, field, vol_weighted, power, dset):
        with DerivedDataset(family, dset) as derived_dset:
            if vol_weighted:
                return derived_dset[field]**power * derived_dset["dx"]**3
            else:
                return derived_dset[field]**power

    return lambda dset: _fn(family, field, vol_weighted, power, dset)

def ScalarOperator(family, field, vol_weighted):
    '''
    Returns a projection operator for scalar fields
    '''
    from pymses.analysis import ScalarOperator as ScalarOp

    # Gather unit information
    unit = family.C.none
    if (seren3.is_derived(family.family, field)):
        unit = seren3.get_derived_field_unit(family.family, field)

    elif (seren3.in_tracked_field_registry(field)):
        field_info = seren3.info_for_tracked_field(field)
        info_key = field_info["info_key"]
        unit = family.info[info_key]

    fn = lambda_function(family, field, vol_weighted, 1)
    op = ScalarOp(fn, unit)
    return op

def FractionOperator(family, field, vol_weighted):
    '''
    Returns a projection operator for scalar fields
    '''
    from pymses.analysis import FractionOperator as FractonOp

    # Gather unit information
    unit = family.C.none
    if (seren3.is_derived(family.family, field)):
        unit = seren3.get_derived_field_unit(family.family, field)

    elif (seren3.in_tracked_field_registry(field)):
        field_info = seren3.info_for_tracked_field(field)
        info_key = field_info["info_key"]
        unit = family.info[info_key]

    up_fn = lambda_function(family, field, vol_weighted, 2)
    down_fn = lambda_function(family, field, vol_weighted, 1)
    op = FractonOp(up_fn, down_fn, unit)
    return op

def Projection(family, field, mode='fft', camera=None, op=None,\
         vol_weighted=False, **kwargs):
    '''
    Takes a projection of the supplied family
    '''
    from seren3.core.snapshot import Family

    if not isinstance(family, Family):
        raise Exception("Require Family specific snapshot, got %s. Use snap.g, .d or .s" % family)    

    if camera is None:
        camera = family.camera()
    if op is None:
        if (vol_weighted):
            op = FractionOperator(family, field, vol_weighted)
        else:
            op = ScalarOperator(family, field)

    field = [field, "dx"] if vol_weighted else [field]
    process = None
    if mode == "fft":
        from pymses.analysis import splatting
        sp = splatting.SplatterProcessor(family[fields].pymses_source, family.info, op)
        process = sp.process
    else:
        raise Exception("Unknown mode: %s" % mode)

    return process(camera, **kwargs)