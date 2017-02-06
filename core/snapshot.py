import abc
import numpy as np
from namelist import load_nml, NML

from seren3 import config

class Snapshot(object):
    """
    Base class for loading RAMSES snapshots
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, path, ioutput, **kwargs):
        import os
        from snapshot_quantities import Quantity
        from pymses.utils import constants as C

        self.path = os.getcwd() if path.strip('/') == '.' else path
        if ioutput != int(ioutput):
            raise Exception("Must provide integer output number (got %f)" % ioutput)
        self.ioutput = int(ioutput)
        self.C = C

        # Known particles
        self.known_particles = ["part", "dm", "star", "gmc"]
        self.particle_field_list = ["mass", "pos", "vel", "epoch", "id"]

        # Load the namelist file
        self.nml = load_nml(self)
        # Tracking metals?
        self.metals = self.nml[NML.PHYSICS_PARAMS]['metal'] == '.true.' or kwargs.pop("metal", False)
        if self.metals:
            self.particle_field_list.append("metal")
        # Patch?
        self.patch = None
        if os.path.isfile("%s/output_%05i/info_rt_%05i.txt" % (self.path, self.ioutput, self.ioutput)):
            self.patch = "rt"
        elif os.path.isfile("%s/output_%05i/rad_%05i.out00001" % (self.path, self.ioutput, self.ioutput)):
            self.patch = "aton"

        # Init friedmann dict variable
        self._friedmann = None

        # Quantities object
        self.quantities = Quantity(self)

    def ancestor(self):
        import weakref
        return weakref.ref(self)

    def array(self, array, units=None, **kwargs):
        from seren3.array import SimArray
        return SimArray(array, units, snapshot=self, **kwargs)

    @abc.abstractmethod
    def g(self):
        return

    @abc.abstractmethod
    def p(self):
        return

    @abc.abstractmethod
    def d(self):
        return

    @abc.abstractmethod
    def s(self):
        return

    @abc.abstractmethod
    def get_io_source(self, family):
        return

    @abc.abstractmethod
    def get_sphere(self, pos, r):
        return

    @property
    def boxsize(self):
        return self.array(self.info["unit_length"]).in_units("Mpc a h**-1")

    def pickle_dump(self, fname, data):
        '''
        Dumps data (safely) to a pickle database
        '''
        import pickle
        with open(fname, "wb") as f:
            pickle.dump(data, f)

    def pickle_load(self, fname):
        '''
        Loads data (safely) from a pickle databse
        '''
        import pickle
        data = None
        with open(fname, "rb") as f:
            data = pickle.load(f)

        return data

    def halos(self, finder=config.get("halo", "default_finder"), **kwargs):
        if finder.lower() == 'ahf':
            from seren3.halos.halos import AHFCatalogue
            return AHFCatalogue(self, **kwargs)
        elif finder.lower() == 'rockstar':
            from seren3.halos.halos import RockstarCatalogue
            return RockstarCatalogue(self, **kwargs)
        elif finder.lower() == 'ctrees':
            from seren3.halos.halos import ConsistentTreesCatalogue
            return ConsistentTreesCatalogue(self, **kwargs)
        else:
            raise Exception("Unknown halo finder: %s" % finder)

    @property
    def h(self):
        return self.halos()

    @abc.abstractmethod
    def camera(self, **kwargs):
        return

    @property
    def friedmann(self):
        if self._friedmann is None:
            self._friedmann = self.integrate_friedmann()
        return self._friedmann


    def detect_rt_module(self):
        '''
        Checks if RAMSES-RT or RAMSES-CUDATON simulation.
        Retuns string 'rt' or 'cudaton'
        '''
        import os
        if os.path.isfile(self.info_rt_fname):
            return 'rt'
        elif os.path.isfile("%s/output_%05i/rad_%05i.out00001" % (self.path, self.ioutout, self.ioutout)):
            return 'cudaton'
        else:
            return 'ramses'
    

    @property
    def info_fname(self):
        return '%s/output_%05d/info_%05d.txt' % (self.path, self.ioutput, self.ioutput)

    @property
    def info_rt_fname(self):
        return '%s/output_%05d/info_rt_%05d.txt' % (self.path, self.ioutput, self.ioutput)

    @property
    def info(self):
        '''
        Expose info API
        '''
        fname = self.info_fname
        from pymses.sources.ramses import info
        info_dict = info.read_ramses_info_file(fname)
        if self.patch == 'rt':
            full_info = info_dict.copy()
            full_info.update(self.info_rt)
            return full_info
        return info_dict

    @property
    def info_rt(self):
        '''
        Expose RT info API
        '''
        from pymses.sources.ramses import info
        
        fname = self.info_rt_fname
        return info.read_ramses_rt_info_file(fname)

    @property
    def unit_l(self):
        return self.array(self.info["unit_length"])

    @property
    def hilbert_dom_decomp(self):
        '''
        Expose Hilbert domain decomposition API
        '''
        from pymses.sources.ramses.hilbert import HilbertDomainDecomp
        info = self.info
        keys = info['dom_decomp_Hilbert_keys']
        dom_decomp = HilbertDomainDecomp(
            info['ndim'], keys[:-1], keys[1:], (info['levelmin'], info['levelmax']))
        return dom_decomp

    def cpu_list(self, bounding_box):
        '''
        Return the list of CPUs which cover the bounding box
        - bounding box: (2, ndim) ndarray containing min/max bounding box
        '''
        return self.hilbert_dom_decomp.map_box(bounding_box)

    @property
    def z(self):
        return self.cosmo["z"]

    @property
    def age(self):
        from seren3.array import SimArray

        fr = self.friedmann
        age_simu = fr["age_simu"]
        return SimArray(age_simu, "Gyr")


    @property
    def cosmo(self):
        """
        Returns a cosmolopy compatible dict
        """
        par = self.info
        cosmo = {'omega_M_0': round(par['omega_m'], 3),
                       'omega_lambda_0': round(par['omega_l'], 3),
                       'omega_k_0': round(par['omega_k'], 3),
                       'omega_b_0': round(par['omega_b'], 3),
                       'h': par['H0'] / 100.,
                       'aexp': par['aexp'],
                       'z': (1. / par['aexp']) - 1.,
                       'omega_n_0': 0.,
                       'N_nu': 0.,
                       'n': 0.96}  # TODO - read this from somewhere
        return cosmo

    @property
    def z(self):
        return (1. / self.info['aexp']) - 1.

    def integrate_friedmann(self, aexp=None):
        from seren3.utils.f90 import friedmann as fm

        cosmology = self.cosmo
        omega_m_0 = cosmology['omega_M_0']
        omega_l_0 = cosmology['omega_lambda_0']
        omega_k_0 = cosmology['omega_k_0']
        if aexp is None:
            aexp = cosmology['aexp']
        H0 = cosmology['h'] * 100

        alpha = 1e-6
        axpmin = 1e-3
        ntable = 1000

        axp_out, hexp_out, tau_out, t_out, age_tot = fm.friedmann(
            omega_m_0, omega_l_0, omega_k_0, alpha, axpmin, ntable)

        # Find neighbouring expansion factor
        i = 1
        while ((axp_out[i] > aexp) and (i < ntable)):
            i += 1

        # Interpolate time
        time_simu = t_out[i] * (aexp - axp_out[i - 1]) / (axp_out[i] - axp_out[i - 1]) + \
            t_out[i - 1] * (aexp - axp_out[i]) / (axp_out[i - 1] - axp_out[i])

        age_simu = (time_simu + age_tot) / \
            (H0 * 1e5 / 3.08e24) / (365 * 24 * 3600 * 1e9)

        friedmann = {
            'axp_out': axp_out,
            'hexp_out': hexp_out,
            'tau_out': tau_out,
            't_out': t_out,
            'age_tot': age_tot,
            'age_simu': age_simu,
            'time_simu': time_simu
        }

        return friedmann

    def pynbody_snapshot(self, **kwargs):
        raise NotImplementedError("Conversion not implemented for base snapshot")

class Family(object):
    """
    Class to load family specific fields
    """
    def __init__(self, snapshot, family):
        import weakref
        self._base = weakref.ref(snapshot)
        self.path = snapshot.path
        self.ioutput = snapshot.ioutput
        self.quantities = snapshot.quantities
        self.family = family.lower()
        self.C = snapshot.C

    def __str__(self):
        return "Family<%s>" % self.family

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        dset = self["pos"].flatten()["pos"]
        return len(dset)

    @property
    def base(self):
        return self._base()

    @property
    def ro(self):
        return self.base.ro

    def array(self, *args, **kwargs):
        return self.base.array(*args, **kwargs)

    @property
    def info(self):
        return self.base.info

    @property
    def cosmo(self):
        return self.base.cosmo

    @property
    def friedmann(self):
        return self.base.friedmann

    @property
    def patch(self):
        return self.base.patch

    @property
    def nml(self):
        return self.base.nml

    def camera(self, **kwargs):
        return self.base.camera(**kwargs)

    def compute_required_fields(self, fields):
        """
        Computes which of the tracked scalar quantities are needed to fully derive a field
        """
        from seren3.utils import derived_utils as derived
        if not hasattr(fields, "__iter__"):
            fields = [fields]

        field_list = None  # Fields RAMSES knows about
        if self.family == 'amr':
            field_list = self.ro._amr_fields().field_name_list
            field_list.extend(["pos", "dx"])
        else:
            field_list = self.base.particle_field_list
            field_list.append("pos")

        known_fields = set()

        def _get_rules(field):
            if derived.is_derived(self.family, field):
                required_fields = [r for r in derived.required_for_field(self.family, field)]
                for r in required_fields:
                    if derived.is_derived(self.family, r):
                        _get_rules(r)
                    else:
                        known_fields.add(r)
            else:
                if field in field_list:
                    known_fields.add(field)
                else:
                    raise Exception("Unknown %s field: %s" % (self.family, field))

        for f in fields:
            _get_rules(f)

        return list(known_fields)

    def get_source(self, fields, return_required_fields=False):
        """
        Data access via pymses for family specific tracked/derived fields
        """
        from serensource import SerenSource
        required_fields = self.compute_required_fields(fields)

        if self.family in self.base.known_particles:
            required_fields.append("level")  # required for fft projections of particle fields
            if "epoch" not in required_fields:
                required_fields.append("epoch")  # required for particle filtering
            if "id" not in required_fields:
                required_fields.append("id")  # required for particle filtering

        source = None
        if "dx" in required_fields or "pos" in required_fields:
            source = self.base.get_io_source(self.family, [r for r in required_fields if r != "dx" and r != "pos"])
        else:
            source = self.base.get_io_source(self.family, required_fields)
        if return_required_fields:
            return source, required_fields
        return source

    def __getitem__(self, fields):
        """
        Data access via pymses for family specific tracked/derived fields
        """
        from serensource import SerenSource
        if not hasattr(fields, "__iter__"):
            fields = [fields]
            
        source, required_fields = self.get_source(fields, return_required_fields=True)

        if self.family in ['amr', 'rt']:
            from pymses.filters import CellsToPoints
            source = CellsToPoints(source)

        cpu_list = None
        if hasattr(self.base, "region"):
            from pymses.filters import RegionFilter
            source = RegionFilter(self.base.region, source)

        return SerenSource(self, source)

    def bin_spherical(self, field, r_units='pc', ncell_per_dim=128, prof_units=None, profile_func=None, center=None, radius=None, nbins=200, divide_by_counts=True):
        '''
        Spherical binning function
        '''
        from seren3.array import SimArray
        from seren3.utils.derived_utils import LambdaOperator, is_derived, get_field_unit
        from seren3.analysis.profile_binners import SphericalProfileBinner

        if center is None:
            if hasattr(self.base, "region"):
                center = self.base.region.center
            else:
                raise Exception("center not specified")

        if radius is None:
            if hasattr(self.base, "region"):
                radius = self.base.region.radius
            else:
                raise Exception("radius not specified")

        if profile_func is None:
            if is_derived(self, field):
                profile_func = LambdaOperator(self, field)
            else:
                profile_func = lambda dset: dset[field]

        _ = np.linspace(0., radius, nbins)

        unit = get_field_unit(self, field)
        source = self[[field, "pos"]]
        binner = SphericalProfileBinner(field, center, profile_func, _, divide_by_counts)
        # prof = self.base.array(binner.process(source), unit)
        prof = binner.process(source, ncell_per_dim)

        if prof_units is not None:
            prof = prof.in_units(prof_units)

        r_bins = (_[1:] + _[:-1])/2.
        r_bins = SimArray(r_bins, self.info["unit_length"]).in_units(r_units)

        return prof, r_bins

    def projection(self, field, **kwargs):
        from seren3.analysis import visualization

        cam = self.camera()
        proj = visualization.Projection(self, field, camera=cam, **kwargs)
        return proj
