from pymses.analysis.operator import AbstractOperator

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