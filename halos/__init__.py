'''
Heavily based on halos.py from pynbody, so credit to those guys
Rewritten to allow loading of Rockstar catalogues when using any module

@author dsullivan, bthompson
'''
import seren3
from seren3 import config
from seren3.core.snapshot import Family
from seren3.array import SimArray
import numpy as np
import logging
import abc
import sys
import weakref

verbose = config.get("general", "verbose")
logger = logging.getLogger('seren3.halos.halos')

class Halo(object):
    """
    Object to represent a halo and allow filtered data access
    """
    def __init__(self, properties, snapshot, units, boxsize):
        self.hid = properties["id"]
        self.properties = properties
        self.units = units
        self.boxsize = boxsize

        self._base = weakref.ref(snapshot)
        self._subsnap = None

    def __str__(self):
        return "halo_" + str(self.hid)

    def __repr__(self):
        # pos, r = self.pos_r_code_units
        # return "pos: %s \t r: %s" % (pos, r)
        return "halo_%i" % self.hid

    def __getitem__(self, item):
        # Return the requested property of this halo, i.e Mvir
        item = item.lower()
        unit_dict = self.units
        unit = None  # Default to dimensionless
        if item in unit_dict:
            unit = unit_dict[item]
        return self.base.array(self.properties[item], unit)

    @property
    def base(self):
        return self._base()

    @property
    def info(self):
        return self.base.info

    @property
    def g(self):
        return Family(self.subsnap, 'amr')

    @property
    def p(self):
        return Family(self.subsnap, "part")

    @property
    def s(self):
        return Family(self.subsnap, 'star')

    @property
    def d(self):
        return Family(self.subsnap, 'dm')

    @property
    def gmc(self):
        return Family(self.subsnap, "gmc")

    @property
    def pos_r_code_units(self):
        return self.pos, self.rvir

    @property
    def pos(self):
        return self["pos"].in_units(self.boxsize)

    @property
    def rvir(self):
        return self["rvir"].in_units(self.boxsize)

    @property
    def dt(self):
        '''
        Time delay for photons to escape, assuming point source
        '''
        rvir = self.rvir.in_units("m")
        rt_c = SimArray(self.base.info_rt["rt_c_frac"] * self.base.C.c)

        dt = rvir / rt_c
        return self.base.array(dt, dt.units)

    @property
    def sphere(self):
        pos, r = self.pos_r_code_units
        return self.base.get_sphere(pos, r)

    @property
    def subsnap(self):
        if self._subsnap is None:
            self._subsnap = self.base[self.sphere]
        return self._subsnap

    def camera(self, **kwargs):
        return self.subsnap.camera(**kwargs)

    def annotate_rvir(self, proj, color="lightsteelblue", facecolor="none", alpha=1, ax=None):
        '''
        Draw the virial radius on a projection plot
        '''
        import matplotlib.pylab as plt
        from matplotlib.patches import Circle

        if ax is None:
            ax = plt.gca()

        camera = proj.camera
        region_size = camera.region_size[0]  # code length
        map_max_size = camera.map_max_size  # projection size in pixels

        unit_l = self.base.array(self.base.info["unit_length"])
        rvir = self.rvir.in_units(unit_l)

        rvir_pixels = (rvir/region_size) * map_max_size
        xy = (map_max_size/2, map_max_size/2)
        e = Circle(xy=xy, radius=rvir_pixels)

        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_edgecolor( color )
        e.set_facecolor( facecolor )  # "none" not None
        e.set_alpha( alpha )

        return e

    @property
    def Vc(self):
        '''
        Returns the circular velocity of the halo
        '''
        G = SimArray(self.base.C.G)
        M = self["mvir"].in_units("kg")
        Rvir = self["rvir"].in_units("m")
        Vc = np.sqrt( (G*M)/Rvir )

        return self.base.array(Vc, Vc.units)

    @property
    def Tvir(self):
        '''
        Returns the virial Temperature of the halo
        '''
        mu = 0.59  # Okamoto 2008
        mH = SimArray(self.base.C.mH)
        kB = SimArray(self.base.C.kB)
        Vc = self.Vc

        Tvir = 1./2. * (mu*mH/kB) * Vc**2
        return self.base.array(Tvir, Tvir.units)

    def fesc(self, **kwargs):
        from seren3.analysis import escape_fraction
        return escape_fraction.fesc(self.subsnap, **kwargs)

    def pynbody_snapshot(self, **kwargs):
        return self.subsnap.pynbody_snapshot(**kwargs)


class SortedHaloCatalogue(object):

    '''
    Simple wrapper to sort halos and return Halo
    objects
    '''
    def __init__(self, halo_catalogue, sort_key):
        self._halo_catalogue = halo_catalogue
        self._sorted_halos = np.sort(halo_catalogue._haloprops, order=sort_key)[::-1]

    def __getitem__(self, item):
        hprops = self._sorted_halos[item]
        return Halo(hprops, self._halo_catalogue.base, self._halo_catalogue.units, self._halo_catalogue.get_boxsize())

    def __len__(self):
        return len(self._sorted_halos)


class HaloCatalogue(object):
    """
    Abstract halo catalogue
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, pymses_snapshot, finder, filename=None, **kwargs):
        # import weakref
        # self._base = weakref.ref(pymses_snapshot)
        self._base = weakref.ref(pymses_snapshot)
        self.finder = finder
        self.finder_base_dir = "%s/%s" % (self.base.path, config.get("halo", "%s_base" % self.finder.lower()))

        can_load, message = self.can_load(**kwargs)
        if can_load is False:
            print "Unable to load catalogue: %s" % message
            logger.error("Unable to load catalogue: %s" % message)
            return

        self.filename = self.get_filename(**kwargs) if filename is None else filename
        self.boxsize = self.get_boxsize(**kwargs)  # Mpccm/h

        if(verbose):
            print "%sCatalogue: loading halos..." % self.finder,
            sys.stdout.flush()

        self.load(**kwargs)
        if(verbose):
            print 'Loaded %d halos' % len(self)

    def __len__(self):
        return len(self._haloprops)

    def __str__(self):
        return "%sCatalogue - Snapshot - %s" % (self.finder, self.base)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return self._halo_generator()

    def __getitem__(self, item):
        return self._get_halo(item)

    @property
    def base(self):
        return self._base()

    @abc.abstractmethod
    def get_boxsize(self, **kwargs):
        return

    @abc.abstractmethod
    def _get_halo(self, item):
        return

    @abc.abstractmethod
    def can_load(self):
        return False

    @abc.abstractmethod
    def get_filename(self, **kwargs):
        return

    @abc.abstractmethod
    def load(self):
        return

    @property
    def mvir_array(self):
        mvir = np.zeros(len(self._haloprops))
        for i in range(len(self._haloprops)):
            mvir[i] = self._haloprops[i]['mvir']

        return mvir

    @property
    def mass_sigma_relation(self):
        '''
        Returns the mass-sigma relation required to compute the
        multipllicity function (f(sigma))
        Requires transfer functions to exist for this redshift in the camb/MUSIC dir
        '''
        from seren3.cosmology import transfer_function
        from seren3.cosmology import lingrowthfac

        # Dict describing our cosmology
        cosmo = self.base.cosmo
        print 'Rounding redshift: %1.1f -> %1.1f' % (cosmo['z'], np.round(cosmo['z']))
        cosmo['z'] = np.round(cosmo['z'])
        cosmo['aexp'] = 1./(1. + cosmo['z'])

        # PowerSpectrum & routines to compute species specific ps using
        # transfer functions
        ps = transfer_function.PowerSpectrumCamb(**cosmo)
        
        # Out TopHat filter kernel
        f_filter = transfer_function.TophatFilter(**cosmo)

        mvir = sorted(self.mvir_array)
        # Integrate k**2 * P(k) * W(k,R)**2
        # lingrowth = lingrowthfac(self.base.z, **cosmo)
        # variance func. includes factor 1/(2*pi**2)
        var = np.array( [transfer_function.variance(m, ps, f_filter=f_filter, arg_is_R=False) for m in mvir] )
        # var = lingrowth**2 * np.array( [transfer_function.variance(m, ps, f_filter=f_filter, arg_is_R=False) for m in mvir] )

        sigma = np.sqrt(var)
        return sigma, mvir

    def halo_ix(self, shuffle=False):
        '''
        Return list of indicies to halos, which can be scattered with MPI
        shuffle (bool) - whether to shuffle the array
        '''

        halo_ix = range(len(self))
        if shuffle:
            import random
            random.shuffle(halo_ix)
        return halo_ix

    def mpi_spheres(self):
        '''
        Returns iterable which can be scattered/gathered
        '''
        halo_spheres = np.array( [ {'id' : h.hid, 'reg' : h.sphere, 'mvir' : h['mvir']} for h in self ] )
        return halo_spheres

    def _halo_generator(self):
        i = 0
        while True:
            try:
                yield self[i]
                i += 1
                if i > len(self._haloprops) - 1:
                    break
            except RuntimeError:
                break

    def kdtree(self, bounds=[1., 1., 1.]):
        '''
        Return a KDTree with all halos, accounting for periodic boundaries
        '''
        if not hasattr(self, '_ctree'):
            from periodic_kdtree import PeriodicCKDTree
            points = np.array([halo['pos'].in_units("Mpc") / self.boxsize.in_units("Mpc")
                               for halo in self])
            T = PeriodicCKDTree(bounds, points)
            self._ctree = T
        return self._ctree

    def search(self, condition):
        '''
        Search halos for matches
        condition - function to evaluate matches
        e.g condition = lambda halos: halos[:]['Mvir'] > 1e9 NB : Unit conversion wont work here
        Kinda messy
        '''
        idx = np.where(condition(self._haloprops))[0]
        found = []
        for i in idx:
            found.append(self[i])
        return found

    def from_id(self, hid):
        idx = np.where(self._haloprops[:]["id"] == hid)
        return self[idx]

    def closest_halos(self, point, n_halos=1):
        '''
        Return the closest halo to the point
        n_halos - Number of halos to find - default to 1
        '''
        # Build the periodic tree
        T = self.kdtree()
        neighbours = T.query(point, n_halos)
        return neighbours

    def from_indicies(self, idx):
        '''
        Return list of halos specified by their index
        '''
        return np.array([self[i] for i in range(len(self)) if i in idx])

    # def sort(self, field, halos=None, reverse=True):
    #     '''
    #     Sort halos by a given field
    #     '''
    #     if halos is None:
    #         halos = self
    #     return sorted(halos, key=lambda x: x[field], reverse=reverse)

    def sort(self, key):
        return SortedHaloCatalogue(self, key.lower())

    def plot_mass_function(self, units='Msol h**-1', kern='ST', ax=None,\
                     plot_Tvir=True, label_z=False, nbins=100, label=None, show=True, **kwargs):
        '''
        Plot the Halo mass function and (optionally) Tvir on twinx axis

        Params:
            kern: The analytical kernal to use
            plot_Tvir: Calculates the Virial temperature in Kelvin for a halo of a given mass using equation 26 of Barkana & Loeb.
        '''
        import matplotlib.pylab as plt
        from seren3.analysis.plots import fit_scatter

        snapshot = self.base
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if label_z:
            label = "%s z=%1.3f" % (label, self.base.z)

        # Compute HMF from realization and plot
        # boxsize = self.boxsize  # Mpccm/h
        boxsize = self.base.boxsize.in_units("Mpc a h**-1")

        mbinmps, mhist, mbinsize = self.mass_function(units=units, nbins=nbins)
        y = self.base.array(mhist/(boxsize**3)/mbinsize, "Mpc**-3 h**3 a**-3")
        ax.semilogy(mbinmps, y, 'o', label=label, color='b')
        # ax.set_xlim(6, 8)

        bin_centers, mean, std = fit_scatter(mbinmps, y)
        ax.errorbar(bin_centers, mean, yerr=std, color='b')

        if plot_Tvir:
            import cosmolopy.perturbation as cp
            cosmo = snapshot.cosmo
            M_ticks = np.array(ax.get_xticks())

            mass = self.base.array(10**M_ticks, units)
            Tvir = [float("%1.3f" % np.log10(v)) for v in cp.virial_temp(mass, **cosmo)]

            ax2 = ax.twiny()
            ax2.set_xticks(M_ticks-M_ticks.min())
            ax2.set_xticklabels(Tvir)
            ax2.set_xlabel(r"log$_{10}$(T$_{\mathrm{vir}}$ [K])")

            # ax.set_xlim(M_ticks[0], M_ticks[-1])

        if kern is not None:
            import pynbody

            # We only need to load one CPU to setup the params dict
            s = pynbody.snapshot.ramses.RamsesSnap("%s/output_%05d" % (snapshot.path, snapshot.ioutput), cpus=[1])
            M_kern, sigma_kern, N_kern = pynbody.analysis.halo_mass_function(s, kern=kern)

            # ax.semilogy(np.log10(M_kern*(snapshot.info['H0']/100)), N_kern, label=kern)
            ax.semilogy(np.log10(M_kern*self.base.cosmo["h"]), N_kern, label=kern, color="k")

        ax.set_xlabel(r'log$_{10}$(M [M$_{\odot}$/h])')
        ax.set_ylabel('dN / dlog$_{10}$(M [Mpc$^{-3}$ h$^{3}$])')

        if "title" in kwargs:
            ax.set_title(kwargs.get("title"))

        ax.set_xlim(mbinmps.min(), mbinmps.max())
        ax.set_ylim(1e-1, 1e3)
        # ax.set_ylim(y.min(), y.max())

        if show:
            ax.legend()
            plt.show()


    def mass_function(self, units='Msun/h', nbins=100):
        '''
        Compute the halo mass function for the given catalogue
        '''
        masses = []
        for halo in self:
            Mvir = halo['Mvir'].in_units(units)
            masses.append(Mvir)

        mhist, mbin_edges = np.histogram(np.log10(masses), bins=nbins)
        mbinmps = np.zeros(len(mhist))
        mbinsize = np.zeros(len(mhist))
        for i in np.arange(len(mhist)):
            mbinmps[i] = np.mean([mbin_edges[i], mbin_edges[i + 1]])
            mbinsize[i] = mbin_edges[i + 1] - mbin_edges[i]

        return mbinmps, mhist, mbinsize

    def dump(self, fname):
        '''
        Dump positions and mass for splotting
        '''
        with open(fname, 'w') as f:
            for i in range(len(self)):
                halo = self[i]
                pos = halo['pos'].in_units('Mpccm/h')
                f.write('%f  %f  %f  %e' %
                        (pos[0], pos[1], pos[2], halo['Mvir'].in_units('Msun')))
                if i < len(self):
                    f.write('\n')