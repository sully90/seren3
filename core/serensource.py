import seren3
from seren3.array import SimArray
from pymses.sources.ramses.sources import RamsesAmrSource, RamsesParticleSource
from pymses.core import sources

from seren3 import config
verbose = config.get("general", "verbose")

class SerenSource(sources.DataSource):
    """
    Class to extend pymses source and implement derived fields
    """
    def __init__(self, family, source, required_fields, requested_fields, cpu_list=None):
        super(SerenSource, self).__init__()
        self._source = source
        self._dset = None
        self._cpu_list = cpu_list
        self.family = family
        self.required_fields = required_fields
        self.requested_fields = requested_fields

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.flatten()[item]
        elif str(item).isdigit():
            return self.get_domain_dset(item)
        else:
            raise Exception("Can't get item: %s" % item)

    def __iter__(self):
        cpu_list = None
        if self._cpu_list is not None:
            cpu_list = self._cpu_list
        else:
            cpu_list = range(1, self.family.info['ncpu'] + 1)
        for idomain in cpu_list:
            yield self.get_domain_dset(idomain)

    def field_latex(self, field):
        icpu = self._cpu_list[0]
        return self.get_domain_dset(icpu).latex

    @property
    def points(self):
        if self._dset is None:
            return self._source.flatten().points
        return self._dset.points

    def get_sizes(self):
        if self._dset is None:
            return self._source.flatten().get_sizes()
        return self._dset.get_sizes()

    @property
    def point_dset_source(self):
        return self._source

    @property
    def source(self):
        '''
        The base RamsesAmrSource
        '''
        src = self._source
        if isinstance(src, RamsesAmrSource) or isinstance(src, RamsesParticleSource):
            return src
        else:
            while True:
                src = src.source
                if isinstance(src, RamsesAmrSource) or isinstance(src, RamsesParticleSource):
                    return src

    @property
    def f(self):
        return self.flatten()

    def flatten(self, **kwargs):
        self._dset = self._source.flatten(verbose=self.family.base.verbose)
        return self._derived_dset(**kwargs)

    def get_domain_dset(self, idomain, **kwargs):
        self._dset = self._source.get_domain_dset(idomain)
        return self._derived_dset(**kwargs)

    def sample_points(self, pxyz, **kwargs):
        import pymses

        # source = self.source
        # if hasattr(source, "source"):
        #     source = source.source
        source = self.source
        self._dset = pymses.analysis.sample_points(source, pxyz, use_C_code=True)
        return self._derived_dset(**kwargs)

    def _derived_dset(self, **kwargs):
        if(verbose): print "Deriving dataset..."

        # Setup dicts to hold fields
        if self._dset is None:
            return {}
        derived_dset = {}
        dset = self._dset
        family = self.family.family

        # User requested dx or pos fields?
        if "dx" in self.required_fields:
            dset.add_scalars("dx", dset.get_sizes())
        if "pos" in self.required_fields:
            dset.add_vectors("pos", dset.points)

        # Deal with tracked fields / units
        tracked_fields = {}
        for f in self.required_fields:
            # Deal with group specific fields i.e Np1
            s = f
            if f[-1].isdigit():
                s = f[:-1]
            if seren3.in_tracked_field_registry(s):
                # Field is tracked by RAMSES -> Get unit information
                info_for_field = seren3.info_for_tracked_field(s)
                unit_key = info_for_field["info_key"]

                unit = self.family.info[unit_key]
                val = SimArray(dset[f], unit)

                # User defined default unit?
                if "default_unit" in info_for_field:
                    val = val.in_units(info_for_field["default_unit"])
                tracked_fields[f] = val
            else:
                # We have no unit information -> return dimensionless SimArray
                # tracked_fields[f] = SimArray(dset[f], 1)
                tracked_fields[f] = dset[f]

        # Derive fields
        def _get_derived(field, temp):
            '''
            Recursively derive all the fields we need
            '''
            # List of fields required to derive requested field
            rules = [r for r in seren3.required_for_field(family, field)]
            for r in rules:
                if seren3.is_derived(family, r) and r not in dset.fields:
                    # Recursively derive required field
                    _get_derived(r, temp)

                elif r in tracked_fields:
                    temp[r] = tracked_fields[r]

            # Get the appropriate function from the registry and call it
            fn = seren3.get_derived_field(family, field)
            temp[field] = fn(self.family.base, temp, **kwargs)

        for f in self.requested_fields:
            if f in dset.fields:  # tracked field
                derived_dset[f] = tracked_fields[f]
            elif f not in derived_dset and seren3.is_derived(family, f):
                temp = {}
                for r in self.required_fields:
                    if r == 'pos':
                        temp["pos"] = dset.points

                    elif r == 'dx':
                        temp["dx"] = dset.get_sizes()

                    else:
                        temp[r] = dset[r]

                # Populate temp with the required fields to derive f
                _get_derived(f, temp)
                derived_dset[f] = temp[f]
            else:
                raise Exception("Don't know what to do with non-tracked and non-derived field: %s", f)

        if(verbose): print "Done"
        if len(self.requested_fields) == 1:
            return derived_dset[self.requested_fields[0]]
        return derived_dset
