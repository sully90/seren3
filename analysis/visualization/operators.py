from pymses.analysis.operator import AbstractOperator, FractionOperator

# Map operator : minimum temperature along line-of-sight
class MinTempOperator(AbstractOperator):
    def __init__(self, unit):
        '''
        Computes minimum temperature along line of sight.
        snap - seren3.core.snapshot
        '''
        def invT_func(dset):
            P = dset["P"]
            rho = dset["rho"]
            r = rho/P
            # print r[(rho<=0.0)+(P<=0.0)]
            # r[(rho<=0.0)*(P<=0.0)] = 0.0
            return r
        d = {"invTemp": invT_func}
        super(MinTempOperator, self).__init__(d, unit, is_max_alos=True)
    def operation(self, int_dict):
        map = int_dict.values()[0]
        mask = (map == 0.0)
        mask2 = map != 0.0
        map[mask2] = 1.0 / map[mask2]
        map[mask] = 0.0
        return map


# Map operator : minimum hydrogon ionised fraction along line-of-sight
class MinxHIIOperator(AbstractOperator):
    def __init__(self, unit):
        '''
        Computes minimum temperature along line of sight.
        snap - seren3.core.snapshot
        '''
        def invT_func(dset):
            return 1./dset["xHII"]
        d = {"invxHII": invT_func}
        super(MinxHIIOperator, self).__init__(d, unit, is_max_alos=True)
    def operation(self, int_dict):
        map = int_dict.values()[0]
        mask = (map == 0.0)
        mask2 = map != 0.0
        map[mask2] = 1.0 / map[mask2]
        map[mask] = 0.0
        return map


# Simple density weighted operator for ray-tracing for scalar fields
class DensityWeightedOperator(FractionOperator):
    def __init__(self, field, unit, **kwargs):
        up_fn = lambda dset: dset[field] * dset["rho"]
        down_fn = lambda dset: dset["rho"]
        super(DensityWeightedOperator, self).__init__(up_fn, down_fn, unit, **kwargs)


# Simple mass weighted operator for ray-tracing for scalar fields
class MassWeightedOperator(FractionOperator):
    def __init__(self, field, unit, **kwargs):
        up_fn = lambda dset: dset[field] * (dset["rho"] * dset.get_sizes()**3)
        down_fn = lambda dset: dset["rho"] * dset.get_sizes()**3
        super(MassWeightedOperator, self).__init__(up_fn, down_fn, unit, **kwargs)