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
            info_key = seren3.get_tracked_field_info_key(field)
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
            info_key = seren3.get_tracked_field_info_key(field)
            unit = family.info[info_key]
    up_func = seren3.LambdaOperator(family, field, power=2., vol_weighted=vol_weighted)
    down_func = seren3.LambdaOperator(family, field, power=1., vol_weighted=vol_weighted)

    # print "Unit: ", unit
    op = pymFractionOperator(up_func, down_func, unit)
    return op


def ProjectionPlot(family, field, mode='splatting', map_unit='Mpc', vol_weighted=True, camera=None, op=None):
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
    if mode != 'splatting':
        raise NotImplementedError("splatting only projections implemented")

    if camera is None:
        camera = family.base.camera()

    if op is None:
        if vol_weighted:
            op = FractionOperator(family, field, vol_weighted=vol_weighted)
        else:
            op = ScalarOperator(family, field)

    map_unit = seren3.pymses_units(map_unit)

    if mode == 'splatting':
        from pymses.analysis import splatting
        fn = splatting.SplatterProcessor(family[field].source, family.info, op)
        proj = fn.process(camera)
        return proj
