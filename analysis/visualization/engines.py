'''
Collection of projection engines. Raytracing supports INTENSIVE amr variables, splatter (FFT) supports
EXTENSIVE variables, but can be used to make projections of intensive variables
(see MassWeightedDensitySplatterEngine example).

The stock RayTraceEngine and SplatterEngine can process any quantity (raytracing for AMR only), but
do not ensure the variables are intensive/extensive respectively. They use simple ScalarOperators,
but derive all fields and units.

The ProjectionEngine class is abstract and cannot be instantiated, but provides the base logic for 
visualizations. Typically one would create an engine and call process with a camera object:
from seren3.analysis.visualization import engines
engine = engines.RayTraceEngine(snapshot.g, "rho")
projection = engine.process(snapshot.camera())
It is down to the desired engine to provide the correct AMR source and operator.
'''
import abc, weakref
from seren3.core.serensource import DerivedDataset
from seren3.exceptions import IncompatibleUnitException

class ProjectionEngine(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, family, field):
        """
            Base class for collecting necessary information to make visualizations of quantities
        """
        self._family = weakref.ref(family)
        self.field = field
        self.info = self.family.base.ro.info


    @property
    def family(self):
        if self._family is None:
            raise Exception("Lost reference to base family")
        return self._family()


    def _units(self):
        '''
        Load a single domain so we can collect full unit information
        '''
        unit_dict = {}
        source = self.family[self.field]

        iout = source.get_cpu_list()[0]
        dset = source.get_domain_dset(iout)
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
        from seren3.array import units
        from seren3.utils import derived_utils

        unit_dict = self._units()
        unit = unit_dict[self.field]

        if unit == units.NoUnit():
            return self.family.C.none
        return derived_utils.pymses_units(str(unit))

    def get_field(self, dset, field):
        dset_gen = lambda dset: DerivedDataset(self.family, dset)
        return dset_gen(dset)[field]


    def get_operator(self):
        '''
        Return a simple scalar operator for this field, provided it is intensive
        '''
        from pymses.analysis import ScalarOperator

        is_max_alos = self.IsMaxAlos()
        op = ScalarOperator( lambda dset: self.get_field(dset, self.field), self.get_map_unit(), is_max_alos=is_max_alos)
        return op


    def process(self, camera, **kwargs):
        processor = self.get_process()

        # Always ensure we respect the IsMaxAlos property of a visualization engine
        if processor._operator.is_max_alos() != self.IsMaxAlos():
            processor._operator._max_alos = self.IsMaxAlos()

        if "surf_qty" not in kwargs:
            kwargs["surf_qty"] = self.IsSurfQuantity()
        return processor.process(camera, **kwargs)


    def IsSurfQuantity(self):
        return False


    def IsMaxAlos(self):
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
        return rt


    def IsMaxAlos(self):
        '''
        True for raytracers
        '''
        return True


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
        return sp


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

