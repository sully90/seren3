'''
Module to handle implementation of derived fields and I/O
'''
import seren3
from seren3.array import SimArray

class DerivedDataset(object):
    '''
    Class to handle indexing/deriving of fields
    '''
    INFO_KEY = "info_key"
    DEFAULT_UNIT = "default_unit"
    def __init__(self, family, dset, **kwargs):
        self._dset = dset
        self.family = family
        self.kwargs = kwargs

        # Index RAMSES fields with unit information
        self.indexed_fields = {}
        # keys = indexed_fields.fields if hasattr(indexed_fields, "fields") else indexed_fields.keys()
        keys = dset.fields

        for field in keys:
            if seren3.in_tracked_field_registry(field):
                info_for_field = seren3.info_for_tracked_field(field)
                unit_key = info_for_field[self.INFO_KEY]
                unit = self.family.info[unit_key]
                self.indexed_fields[field] = SimArray(dset[field], unit)

                if self.DEFAULT_UNIT in info_for_field:
                    self.indexed_fields[field].convert_units(info_for_field[self.DEFAULT_UNIT])
            else:
                self.indexed_fields[field] = SimArray(dset[field])

        # Setup position and dx fields
        if hasattr(self._dset, "points"):
            self["pos"] = SimArray(self._dset.points, family.info["unit_length"])
        if hasattr(self._dset, "get_sizes")
            self["dx"] = SimArray(self._dset.get_sizes(), family.info["unit_length"])

    def __getitem__(self, field):
        if field not in self.indexed_fields:
            # Derive the field and add it to our index
            self.indexed_fields[field] = self.derive_field(self.family.family, field, **self.kwargs)

        return self.indexed_fields[field]


    def __setitem__(self, item, value):
        if item in self.indexed_fields:
            raise Exception("Cowardly refusing to override item: %s" % item)
        self.indexed_fields[item] = value


    def __contains__(self, field):
        return field in self.indexed_fields


    @property
    def points(self):
        return self["pos"]


    @property
    def get_sizes(self):
        if self.family.family == "amr":
            return self["dx"]
        else:
            raise Exception("Field dx does not exist for family %s" % self.family.family)


    def derive_field(self, family, field, **kwargs):
        '''
        Recursively derives a field using the existing indexed fields
        '''

        # dset = self.indexed_fields.copy()  # dict to store non-indexed fields required to derive our field

        # First we must collect a list of known RAMSES fields which we require to derive field
        required = seren3.required_for_field(family, field)

        # Recursively Derive any required fields as necessary
        for r in required:
            if r in self:
                continue
            elif (seren3.is_derived(family, r)):
                self[r] = self.derive_field(family, r)
            else:
                raise Exception("Field not indexed or can't derive: (%s, %s)" % (self.family.family, r))

        # dset contains everything we need
        fn = seren3.get_derived_field(family, field)
        var = fn(self.family, self, **kwargs)

        return var

class SerenSource(object):
    '''
    Class to expose data reading routines/pass dsets to DerivedDataset
    '''
    def __init__(self, family, source):
        self.family = family
        self.source = source
        self._cpu_list = None

        if hasattr(self.family, "region"):
            bbox = self.family.region.get_bounding_box()
            self._cpu_list = self.family.cpu_list(bbox)

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
