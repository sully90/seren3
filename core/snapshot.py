import abc
import numpy as np
from namelist import load_nml, NML

class Snapshot(object):
    """
    Base class for loading RAMSES snapshots
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, path, ioutput, **kwargs):
        import os
        from pymses.utils import constants as C

        self.path = os.getcwd() if path.strip('/') == '.' else path
        self.ioutput = ioutput
        self.C = C

        # Known particles
        self.known_particles = ["dm", "star", "gmc"]

        # Load the namelist file
        self.nml = load_nml(self)
        # Tracking metals?
        self.metals = self.nml[NML.PHYSICS_PARAMS]['metal'] == '.true.' or kwargs.pop("metal", False)
        # Patch?
        self.patch = None
        if os.path.isfile("%s/output_%05i/info_rt_%05i.txt" % (self.path, self.ioutput, self.ioutput)):
            self.patch = "rt"
        elif os.path.isfile("%s/output_%05i/rad_%05i.out00001" % (self.path, self.ioutput, self.ioutput)):
            self.patch = "aton"

        # Init friedmann dict variable
        self._friedmann = None

    @abc.abstractmethod
    def ro(self):
        return

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
    def get_source(self, family):
        return

    @abc.abstractmethod
    def get_sphere(self, pos, r):
        return

    def halos(self, finder='ahf', **kwargs):
        if finder.lower() == 'ahf':
            from seren3.halos.halos import AHFCatalogue
            return AHFCatalogue(self, **kwargs)

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
        fname = self.info_rt_fname
        from pymses.sources.ramses import info
        return info.read_ramses_rt_info_file(fname)

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
    def cosmo(self):
        """
        Returns a cosmolopy compatible dict
        """
        par = self.info
        cosmo = {'omega_M_0': par['omega_m'],
                       'omega_lambda_0': par['omega_l'],
                       'omega_k_0': par['omega_k'],
                       'omega_b_0': par['omega_b'],
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

class Family(object):
    """
    Class to load family specific fields
    """
    def __init__(self, snapshot, family):
        self.base = snapshot
        self.family = family.lower()

    def __str__(self):
        return "Family<%s>" % self.family

    def __repr__(self):
        return self.__str__()

    @property
    def ro(self):
        return self.base.ro

    @property
    def info(self):
        return self.base.info

    @property
    def nml(self):
        return self.base.nml

    def camera(self, **kwargs):
        return self.base.camera(**kwargs)

    def compute_required_fields(self, fields):
        """
        Computes which of the tracked scalar quantities are needed to fully derive a field
        Parameters
        ----------
        fields - array/list of Field objects
        """
        from seren3.utils import derived_utils as derived
        if not hasattr(fields, "__iter__"):
            fields = [fields]

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
                known_fields.add(field)

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

        source = None
        if "dx" in required_fields or "pos" in required_fields:
            source = self.base.get_source(self.family, [r for r in required_fields if r != "dx" and r != "pos"])
        else:
            source = self.base.get_source(self.family, required_fields)
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
            cpu_list = self.base.cpu_list(self.base.region.get_bounding_box())

        return SerenSource(self, source, required_fields, fields, cpu_list=cpu_list)

    def bin_spherical(self, field, r_units='pc', prof_units=None, profile_func=None, center=None, r=None, nbins=200, divide_by_counts=False):
        '''
        Spherical binning function
        '''
        from seren3.array import SimArray
        from seren3.utils.constants import from_pymses_unit
        from seren3.utils.derived_utils import LambdaOperator, is_derived
        from seren3.analysis.profile_binners import SphericalProfileBinner

        if center is None:
            if hasattr(self.base, "region"):
                center = from_pymses_unit(self.base.region.center * self.info["unit_length"])
            else:
                raise Exception("center not specified")

        if r is None:
            if hasattr(self.base, "region"):
                r = from_pymses_unit(self.base.region.radius * self.info["unit_length"])
            else:
                raise Exception("radius not specified")

        if profile_func is None:
            if is_derived(self, field):
                profile_func = LambdaOperator(self, field)
            else:
                profile_func = lambda dset: dset[field]

        r_bins = SimArray(np.linspace(0., r, nbins), r.units)

        source = self[[field, "pos"]]
        binner = SphericalProfileBinner(center, profile_func, r_bins, divide_by_counts)


        return binner.process(source)