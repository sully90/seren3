'''
Module to handle implementation of derived fields and I/O
'''
import seren3
from seren3 import config
from seren3.array import units
from pynbody.units import UnitsException

_tracked_field_unit_registry = {"rho" : "unit_density",\
                                "vel" : "unit_velocity",\
                                "P" : "unit_pressure",\
                                "pos" : "unit_length",\
                                "dx" : "unit_length",\
                                "size" : "unit_length",\
                                "Np" : "unit_photon_flux_density",\
                                "Fp" : "unit_photon_flux_density",\
                                "mass" : "unit_mass"}

def is_tracked(field):
    if field[-1].isdigit():
        return field[:-1] in _tracked_field_unit_registry
    return field in _tracked_field_unit_registry

def get_tracked_field_unit(family, field):
    unit = None
    if field[-1].isdigit():
        unit = family.info[_tracked_field_unit_registry[field[:-1]]]    
    else:
        unit = family.info[_tracked_field_unit_registry[field]]
    return units.Unit(str(unit))

class DerivedDataset(object):
    '''
    Class to handle indexing/deriving of fields
    '''
    BASE_UNITS = config.BASE_UNITS
    INFO_KEY = "info_key"
    DEFAULT_UNIT = "default_unit"
    def __init__(self, family, dset, **kwargs):
        self._dset = dset
        self.dims = [units.Unit(x) for x in self.BASE_UNITS["length"],\
                 self.BASE_UNITS["velocity"], self.BASE_UNITS["mass"], "h", "a"]  # default length, velocity, mass units
        self.family = family
        self.kwargs = kwargs

        # Index RAMSES fields with unit information
        self.indexed_fields = {}
        # keys = indexed_fields.fields if hasattr(indexed_fields, "fields") else indexed_fields.keys()

        for field in dset.fields:
            if is_tracked(field):
                unit = get_tracked_field_unit(self.family, field)
                self.indexed_fields[field] = self.family.array(dset[field], unit)
            else:
                self.indexed_fields[field] = self.family.array(dset[field])

        if hasattr(dset, "points"):
            self["pos"] = self.family.array(dset.points, get_tracked_field_unit(self.family, "pos"))

        if hasattr(dset, "get_sizes"):
            self["dx"] = self.family.array(dset.get_sizes(), get_tracked_field_unit(self.family, "dx"))

        # Set units
        self.original_units()

    def __getitem__(self, field):
        if field not in self.indexed_fields:
            # Derive the field and add it to our index
            self.indexed_fields[field] = self.derive_field(field, self, **self.kwargs)

        return self.indexed_fields[field]


    def __setitem__(self, item, value):
        if item in self.indexed_fields:
            raise Exception("Cowardly refusing to override item: %s" % item)
        self.indexed_fields[item] = value


    def __contains__(self, field):
        return field in self.indexed_fields


    def keys(self):
        return self.indexed_fields.keys()


    def original_units(self):
        self.physical_units(length=self.BASE_UNITS["length"], velocity=self.BASE_UNITS["velocity"], mass=self.BASE_UNITS["mass"])


    def physical_units(self, length="kpc", velocity="km s**-1", mass="Msol"):
        '''
        Convert all units to desired physical system
        '''
        from seren3.utils import units as unit_utils

        self.dims = [units.Unit(x) for x in length, velocity, mass, "h", "a"]
        cosmo = self.family.cosmo
        conversion_context = {"a" : cosmo["aexp"], "h" : cosmo["h"]}

        for k in self.keys():
            v = self[k].units
            try:
                new_unit = unit_utils.in_unit_system(v, self.dims)
                self.indexed_fields[k].convert_units(new_unit)
            except UnitsException:
                continue


    @property
    def points(self):
        return self["pos"]


    @property
    def get_sizes(self):
        if self.family == "amr":
            return self["dx"]
        else:
            raise Exception("Field dx does not exist for family %s" % self.family)


    def derive_field(self, field, dset, **kwargs):
        '''
        Recursively derives a field using the existing indexed fields
        '''
        from seren3.utils import units as unit_utils

        # First we must collect a list of known RAMSES fields which we require to derive field
        required = seren3.required_for_field(self.family, field)

        # Recursively Derive any required fields as necessary
        for r in required:
            if r in dset:
                continue
            elif seren3.is_derived(self.family, r):
                self[r] = self.derive_field(r, dset)
            else:
                raise Exception("Field not indexed or can't derive: (%s, %s)" % (self.family, r))

        # dset contains everything we need
        fn = seren3.get_derived_field(self.family, field)
        var = fn(self.family, dset, **kwargs)

        # Apply unit system
        try:
            new_unit = unit_utils.in_unit_system(var.units, self.dims)
            return var.in_units(new_unit)
        except UnitsException:
            return var


class SerenSource(object):
    '''
    Class to expose data reading routines/pass dsets to DerivedDataset
    '''
    def __init__(self, family, source):
        self.family = family
        self.source = source
        self._cpu_list = None

        if hasattr(self.family.base, "region"):
            bbox = self.family.base.region.get_bounding_box()
            self._cpu_list = self.family.base.cpu_list(bbox)

    def __len__(self):
        '''
        Returns the number of CPU domains to be read for this source
        '''
        return len(self._cpu_list)

    def __iter__(self):
        '''
        Iterate over cpu domains
        '''
        cpu_list = None
        if self._cpu_list is not None:
            cpu_list = self._cpu_list
        else:
            cpu_list = range(1, self.family.info['ncpu'] + 1)
        for idomain in cpu_list:
            yield self.get_domain_dset(idomain)

    def get_cpu_list(self):
        if self._cpu_list is None:
            return range(1, self.family.info["ncpu"]+1)
        return self._cpu_list

    @property
    def pymses_source(self):
        '''
        The base RamsesAmrSource
        '''
        from pymses.sources.ramses.sources import RamsesAmrSource, RamsesParticleSource

        src = self.source
        if isinstance(src, RamsesAmrSource) or isinstance(src, RamsesParticleSource):
            return src
        else:
            while True:
                src = src.source
                if isinstance(src, RamsesAmrSource) or isinstance(src, RamsesParticleSource):
                    return src

    def get_domain_dset(self, idomain, **kwargs):
        dset = self.source.get_domain_dset(idomain)
        return DerivedDataset(self.family, dset, **kwargs)

    @property
    def f(self):
        return self.flatten()

    def flatten(self, **kwargs):
        dset = self.source.flatten()
        return DerivedDataset(self.family, dset, **kwargs)

    def generate_uniform_points(self, ngrid):
        import numpy as np
        
        x = y = z = None
        if hasattr(self.family.base, "region"):
            bbox = self.family.base.region.get_bounding_box()
            x, y, z = np.mgrid[bbox[0][0]:bbox[1][0]:complex(ngrid), bbox[0][1]:bbox[1][1]:complex(ngrid), bbox[0][2]:bbox[1][2]:complex(ngrid)]
        else:
            x, y, z = np.mgrid[0:1:complex(ngrid), 0:1:complex(ngrid), 0:1:complex(ngrid)]

        # Reshape
        npoints = np.prod(x.shape)
        x1 = np.reshape(x, npoints)
        y1 = np.reshape(y, npoints)
        z1 = np.reshape(z, npoints)

        # Arrange for Pymses
        pxyz = np.array([x1, y1, z1])
        pxyz = pxyz.transpose()
        return pxyz

    def sample_points(self, points, use_multiprocessing=True, **kwargs):
        '''
        Samples points to cartesian mesh of size 2**level
        '''
        from pymses.analysis.point_sampling import PointSamplingProcessor

        source = self.pymses_source
        psampler = PointSamplingProcessor(source)

        if (use_multiprocessing is False):
            psampler.disable_multiprocessing()
            
        dset = psampler.process(points)
        return DerivedDataset(self.family, dset, **kwargs)
