'''
Collection of projection engines. Raytracing requires intensive variables, splatter (FFT) requires
extensive variables. The MassWeightedDensitySplatterEngine shows how to make projections of intensive
variables with the splatter engine.

The stock RayTraceEngine and SplatterEngine can analyse any quantity (raytracing for AMR only), but
do not ensure the variables are intensive/extensive respectively. They use simple ScalarOperators,
but derive all fields and units.
'''
import abc
from seren3.core.serensource import DerivedDataset
from seren3.exceptions import IncompatibleUnitException

class ProjectionEngine(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, family, field):
        """
            Base class for collecting necessary information to make visualizations of quantities
        """
        self.family = family
        self.field = field
        self.info = self.family.base.ro.info


    def _units(self):
        '''
        Load a single domain so we can collect full unit information
        '''
        unit_dict = {}
        dset = self.family[self.field].get_domain_dset(1)
        unit_dict[self.field] = dset[self.field].units

        for key in dset.keys():
            unit_dict[key] = dset[key].units

        return unit_dict


    @abc.abstractmethod
    def get_process(self):
        return NotImplementedError()


    def get_source(self):
        return self.family[self.field].pymses_source


    def get_map_unit(self):
        """
        Returns the correct unit for this field
        """
        from seren3.utils import derived_utils
        units = self._units()
        unit = str(units[self.field])
        return derived_utils.pymses_units(unit)

    def get_field(self, dset, field):
        dset_gen = lambda dset: DerivedDataset(self.family, dset)
        return dset_gen(dset)[field]


    def get_operator(self):
        '''
        Return a simple scalar operator for this field, provided it is intensive
        '''
        from pymses.analysis import ScalarOperator

        op = ScalarOperator( lambda dset: self.get_field(dset, self.field), self.get_map_unit() )
        return op


    def process(self, camera, **kwargs):
        process_func = self.get_process()
        if "surf_qty" not in kwargs:
            kwargs["surf_qty"] = self.IsSurfQuantity()
        return process_func(camera, **kwargs)


    def IsSurfQuantity(self):
        return False

        
'''
Ray tracing engine for INTENSIVE scalar fields
'''
class RayTraceEngine(ProjectionEngine):
    def __init__(self, family, field):
        super(RayTraceEngine, self).__init__(family, field)


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "intensiveScalarRT"


    def get_process(self):
        from pymses.analysis import raytracing
        op = self.get_operator()
        source = self.get_source()

        rt = raytracing.RayTracer(source, self.info, op)
        return rt.process


class RayTraceMaxLevelEngine(RayTraceEngine):
    def __init__(self, family):
        super(RayTraceMaxLevelEngine, self).__init__(family, None)


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "levelmax"


    def get_map_unit(self):
        return self.family.C.none


    def get_source(self):
        return self.family.base.amr_source([])


    def get_operator(self):
        from pymses.analysis import MaxLevelOperator
        return MaxLevelOperator()


class RayTraceMinTemperatureEngine(RayTraceEngine):
    '''
    Computes the minimum temperature along each ray
    '''
    def __init__(self, family):
        super(RayTraceMinTemperatureEngine, self).__init__(family, "T2")


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "mintemperatureScalarRT"


    def get_map_unit(self):
        return self.info["unit_temperature"]


    def get_operator(self):
        from seren3.analysis.visualization.operators import MinTempOperator
        return MinTempOperator(self.get_map_unit())

'''
FFT convolution for EXTENSIVE scalar fields
'''
class SplatterEngine(ProjectionEngine):
    def __init__(self, family, field):
        super(SplatterEngine, self).__init__(family, field)


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "intensiveScalarFFT"


    def get_process(self):
        from pymses.analysis import splatting

        op = self.get_operator()
        source = self.get_source()

        sp = splatting.SplatterProcessor(source, self.info, op)
        return sp.process


class MassWeightedSplatterEngine(SplatterEngine):
    '''
    Example of how to process intensive variable with the splatter engine
    '''
    def __init__(self, family, field):
        super(MassWeightedSplatterEngine, self).__init__(family, field)


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "customMassWeightedFFT"


    def get_operator(self):
        from pymses.analysis import FractionOperator

        up_func = lambda dset: self.get_field(dset, self.field)**2 * self.get_field(dset, "dx")**3
        down_func = lambda dset: self.get_field(dset, self.field) * self.get_field(dset, "dx")**3
        unit = self.get_map_unit()

        op = FractionOperator(up_func, down_func, unit)
        return op


class MassWeightedDensitySplatterEngine(MassWeightedSplatterEngine):
    '''
    Example of how to process intensive variable with the splatter engine.
    NB - Only works for Family<amr>
    '''
    def __init__(self, amr_family):
        super(MassWeightedDensitySplatterEngine, self).__init__(amr_family, "rho")


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "densityMassWeightedFFT"


class SurfaceDensitySplatterEngine(SplatterEngine):
    '''
    Example of a surface density engine
    '''
    def __init__(self, family):
        super(SurfaceDensitySplatterEngine, self).__init__(family, "rho")


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "Sigma"


    def get_map_unit(self):
        """
        Returns the correct unit for this field
        """
        return self.family.C.kg * self.family.C.m**-2


    def IsSurfQuantity(self):
        return True


class MassWeightedTemperatureSplatterEngine(SplatterEngine):
    def __init__(self, family):
        super(MassWeightedTemperatureSplatterEngine, self).__init__(family, "T2")


    @classmethod
    def is_map_engine_for(cls, map_type):
        return map_type == "temperatureMassWeightedFFT"


    def get_operator(self):
        from pymses.analysis import FractionOperator
        '''
        Here will won't pass the dset through the DerivedDataset for the projection,
        we'll just handle the raw dataset ourselves
        '''

        up_func = lambda dset: dset["P"] * dset.get_sizes()**3
        down_func = lambda dset: dset["rho"] * dset.get_sizes()**3
        unit = self.info["unit_temperature"]

        op = FractionOperator(up_func, down_func, unit)
        return op

