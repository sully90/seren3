import seren3

def ScalarOperator(family, field):
    '''
    family - seren3 family object
    '''
    from pymses.analysis import ScalarOperator as pymScalarOperator
    unit = family.base.C.none
    if seren3.is_derived(family.family, field):
        unit = seren3.get_derived_field_unit(family.family, field)
    else:
        if seren3.in_tracked_field_registry(field):
            field_info = seren3.info_for_tracked_field(field)
            info_key = field_info["info_key"]
            unit = family.info[info_key]
    lambda_op = seren3.LambdaOperator(family, field)
    # print "Unit: ", unit
    op = pymScalarOperator(lambda_op, unit)
    return op


def FractionOperator(family, field, vol_weighted=False):
    from pymses.analysis import FractionOperator as pymFractionOperator
    unit = family.base.C.none
    if seren3.is_derived(family.family, field):
        unit = seren3.get_derived_field_unit(family.family, field)
    else:
        if seren3.in_tracked_field_registry(field):
            field_info = seren3.info_for_tracked_field(field)
            info_key = field_info["info_key"]
            unit = family.info[info_key]
    up_func = seren3.LambdaOperator(family, field, power=2., vol_weighted=vol_weighted)
    down_func = seren3.LambdaOperator(family, field, power=1., vol_weighted=vol_weighted)

    # print "Unit: ", unit
    op = pymFractionOperator(up_func, down_func, unit)
    return op

def ProjectionPlot(family, field, mode='splatting', map_unit='Mpc', vol_weighted=False, camera=None, op=None, **kwargs):
    '''
    Make a projection plot of the (sub)snapshot

    Parameters:
        field (seren2.Field) : the desired field to take the projection of.
            Must be a scalar.
        mode (string) : fft (default) or rt (ray-tracing)
        camera (pymses.analysis.visualization.Camera) : None (default), uses
            snapshot camera
        op (pymses.analyis.visualization.Operator) : None (default), uses
            ScalarOperator
        vol_weighted (bool) : Make volume weighted projection (switches to fft
            mode and FractionOperator)
    '''
    from seren3.core.snapshot import Family

    if not isinstance(family, Family):
        raise Exception("Require Family specific snapshot, got %s. Use snap.g, .d or .s" % family)

    if camera is None:
        camera = family.camera()

    # Disbale volume weighting
    if mode == 'rt' and vol_weighted:
        print "Volume weighting not supported by ray-tracing, disabling"
        vol_weighted = False

    if op is None:
        if vol_weighted:
            op = FractionOperator(family, field, vol_weighted=vol_weighted)
        else:
            op = ScalarOperator(family, field)

    map_unit = seren3.pymses_units(map_unit)
    surf_qty = kwargs.pop('surf_qty', False)
    process = None

    if mode == 'splatting':
        from pymses.analysis import splatting
        sp = splatting.SplatterProcessor(family[field].source, family.info, op)
        process = sp.process
    elif mode == 'rt':
        from pymses.analysis import raytracing
        rt = raytracing.RayTracer(family[field].source, family.ro.info, op)
        process = rt.process
    else:
        raise Exception("Unknown projection mode: %s" % mode)
    proj = process(camera, surf_qty=surf_qty, **kwargs)
    return proj
