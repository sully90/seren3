'''
Heavily based on halos.py from pynbody, so credit to those guys
Rewritten to allow loading of Rockstar catalogues when using any module

NOTE: The new Halo catalogue will only work when loaded through yt for unit coherence.

@author dsullivan, bthompson
=======
TODO -> Use SimArray instead of YTArray
'''
import seren3
from seren3.core.snapshot import Family
from seren3.array import SimArray
import numpy as np
import logging
import abc
import sys

logger = logging.getLogger('seren3.halos.halos')

class Halo(object):
    """
    Object to represent a halo and allow filtered data access
    """
    def __init__(self, halo_id, catalogue, properties):
        self.hid = halo_id
        self.catalogue = catalogue
        self.ds = catalogue.ds
        self.base = catalogue.base
        self.properties = properties

        self._subhalos = None

    def __str__(self):
        return "halo_" + str(self.hid)

    def __repr__(self):
        pos, r = self.pos_r_code_units
        return "pos: %s \t r: %s" % (pos, r)

    def __getitem__(self, item):
        # Return the requested property of this halo, i.e Mvir
        item = item.lower()
        unit_dict = self.catalogue.units
        unit = 'dimensionless'  # Default to dimensionless
        if item in unit_dict:
            unit = unit_dict[item]
        return self.ds.arr(self.properties[item], unit)

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
        pos_units = self.catalogue.units['pos'].lower()
        boxsize = self.catalogue.boxsize
        pos, r = (None, None)
        if pos_units == 'mpccm / h':
            pos = self['pos'].v / boxsize
        elif pos_units == 'kpccm / h':
            pos = self['pos'].v / (boxsize * 1.e3)
        else:
            raise Exception("Cannot handle pos units: %s" % pos_units)

        return pos, self['rvir'].v / (boxsize * 1.e3)

    @property
    def pos(self):
        pos_units = self.catalogue.units['pos'].lower()
        boxsize = self.catalogue.boxsize
        pos = None
        if pos_units == 'mpccm / h':
            pos = self['pos'].v / boxsize
        elif pos_units == 'kpccm / h':
            pos = self['pos'].v / (boxsize * 1.e3)
        else:
            raise Exception("Cannot handle pos units: %s" % pos_units)

        return SimArray(pos, self.base.info["unit_length"])

    @property
    def rvir(self):
        boxsize = self.catalogue.boxsize
        rvir = self["rvir"].v / (boxsize * 1.e3)
        return SimArray(rvir, self.base.info["unit_length"])

    @property
    def sphere(self):
        pos, r = self.pos_r_code_units
        return self.base.get_sphere(pos, r)

    @property
    def subsnap(self):
        return self.base[self.sphere]

    def camera(self, **kwargs):
        return self.subsnap.camera(**kwargs)

    @property
    def Vc(self):
        '''
        Returns the circular velocity of the halo
        '''
        G = SimArray(self.base.C.G)
        M = self["mvir"].in_units("kg")
        M = SimArray(M.v, str(M.units))
        Rvir = self["rvir"].in_units("m")
        Rvir = SimArray(Rvir.v, str(Rvir.units))

        return np.sqrt( (G*M)/Rvir )

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
        return Tvir

    # def fesc(self, **kwargs):
    #     '''
    #     Computes halo escape fraction of hydrogen ionising photons
    #     '''
    #     from seren3.analysis.render import render_spherical

    #     rvir = self.rvir
    #     rt_c = SimArray(self.base.info_rt["rt_c_frac"] * self.base.C.c)
    #     dt = rvir / rt_c

    #     # Compute number of ionising photons from stars at time
    #     # rvir/rt_c (assumin halo is a point source)
    #     dset = self.s[["Nion_d", "mass", "age"]].flatten(dt=dt)
    #     keep = np.where(dset["age"] - dt >= 0.)
    #     mass = dset["mass"][keep]
    #     nPhot = dset["Nion_d"] * mass

    #     # Computed integrated flux out of the virial sphere
    #     integrated_flux = render_spherical.integrate_surface_flux( 
    #         render_spherical.render_quantity(self.g, "rad_0_flux", units="s**-1 m**-2", ret_mag=False, **kwargs), rvir )
    #     integrated_flux *= self.base.info_rt["rt_c_frac"]  # scaled by reduced speed of light  -- is this right?

    #     # return the escape fraction
    #     return nPhot.sum() / integrated_flux

    def fesc(self, **kwargs):
        from seren3 import analysis
        return analysis.fesc(self.subsnap, **kwargs)

    def pynbody_snapshot(self, **kwargs):
        return self.subsnap.pynbody_snapshot(**kwargs)


class HaloCatalogue(object):
    """
    Abstract halo catalogue
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, pymses_snapshot, finder, filename=None, **kwargs):
        self.base = pymses_snapshot
        self.finder = finder

        if self.can_load(**kwargs) is False:
            print "Unable to load catalogue for dataset with filename %s" % pymses_snapshot.path
            logger.info("Unable to load catalogue for dataset with filename %s" % pymses_snapshot.path)
            return

        import yt
        self.ds = yt.load(\
            "%s/output_%05d/info_%05d.txt" % \
            (pymses_snapshot.path, pymses_snapshot.ioutput, pymses_snapshot.ioutput))
        self.filename = self.get_filename(**kwargs) if filename is None else filename
        self.boxsize = self.get_boxsize(**kwargs)  # Mpccm/h

        print "%sCatalogue: loading halos..." % self.finder,
        sys.stdout.flush()
        self.load(**kwargs)
        print 'Loaded %d halos' % len(self)

    def __len__(self):
        return len(self._haloprops)

    def __str__(self):
        return "%sCatalogue - Snapshot - %s" % (self.finder, self.snapshot)

    def __iter__(self):
        return self._halo_generator()

    def __getitem__(self, item):
        return self._get_halo(item)

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

    def mpi_spheres(self):
        '''
        Returns iterable which can be scattered/gathered
        '''
        halo_spheres = np.array( [ {'id' : h.hid, 'reg' : h.sphere, 'mvir' : h['mvir'].v} for h in self ] )
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
            points = np.array([halo['pos'].in_units('code_length')
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

    def with_id(self, id):
        '''
        Returns halo(s) with the desired id.
        Slow, but preserves id order
        '''
        # halos = []
        # for i in id:
        #     ix = np.where(self._haloprops[:]['id'] == i)
        #     halos.append(self[ix])
        # return halos
        if hasattr(id, "__iter__"):
            keep = []
            for h in self:
                if h['id'] in id:
                    keep.append(h)
            return keep
        else:
            func = lambda h: h['id'] == id
            idx = np.where(func(self._haloprops))[0][0]
            return self[idx]

    def closest_halos(self, point, n_halos=1, units='code_length'):
        '''
        Return the closest halo to the point
        n_halos - Number of halos to find - default to 1
        '''
        # Build the periodic tree
        T = self.kdtree()
        neighbours = T.query(point, n_halos)
        return neighbours

    def within_radius(self, pos, rad, convert_units=True):
        '''
        Find halos within a given radius of pos
        '''
        T = self.kdtree()
        neighbors = None

        if convert_units:
            neighbors = T.query_ball_point(
                pos.in_units('code_length').v, r=rad.in_units('code_length').v)
        else:
            neighbors = T.query_ball_point(
                pos, r=rad)
        # found = self._haloprops[neighbors]
        # return found
        return self.from_indicies(neighbors)

    def from_id(self, hid):
        for h in self:
            if h.hid == hid:
                return h

    def from_indicies(self, idx):
        '''
        Return list of halos specified by their index
        '''
        return np.array([self[i] for i in range(len(self)) if i in idx])

    def sort(self, field, halos=None, reverse=True):
        '''
        Sort halos by a given field
        '''
        if halos is None:
            halos = self
        return sorted(halos, key=lambda x: x[field], reverse=reverse)

    def custom_sort(self, func, reverse=True):
        '''
        Sort halos based on a custom function
        func - a function which returns a number (i.e Mass) to sort by
        '''
        return sorted(self, key=func, reverse=reverse)

    def plot_mass_function(self, units='Msun/h', kern='ST', ax=None,\
                     plot_Tvir=True, label_z=False, nbins=100, label=None, show=True, **kwargs):
        '''
        Plot the Halo mass function and (optionally) Tvir on twinx axis

        Params:
            kern: The analytical kernal to use
            plot_Tvir: Calculates the Virial temperature in Kelvin for a halo of a given mass using equation 26 of Barkana & Loeb.
        '''
        import matplotlib.pylab as plt
        from seren2.analysis.plots import fit_scatter

        snapshot = self.snapshot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        if label_z:
            label = "%s z=%1.3f" % (label, self.snapshot.z)

        # Compute HMF from realization and plot
        boxsize = self.boxsize  # Mpccm/h

        mbinmps, mhist, mbinsize = self.mass_function(units=units, nbins=nbins)
        y = mhist/(boxsize**3)/mbinsize
        ax.semilogy(mbinmps, y, 'o', label=label, color='b')
        ax.set_xlim(6, 8)

        bin_centers, mean, std = fit_scatter(mbinmps, y)
        ax.errorbar(bin_centers, mean, yerr=std, color='b')

        if plot_Tvir:
            import cosmolopy.perturbation as cp
            cosmo = snapshot.cosmo
            M_ticks = np.array(ax.get_xticks())

            mass = self.ds.arr(10**M_ticks, units).in_units("Msun").v
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

            ax.semilogy(np.log10(M_kern*(snapshot.info['H0']/100)), N_kern, label=kern)

        ax.set_xlabel(r'log$_{10}$(M [M$_{\odot}$/h])')
        ax.set_ylabel('dN / dlog$_{10}$(M [Mpc$^{-3}$ h$^{3}$])')

        if "title" in kwargs:
            ax.set_title(kwargs.get("title"))

        # ax.set_xlim(mbinmps.min(), mbinmps.max())
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

    def dbscan(self, eps=0.4, min_samples=20):
        '''
        DBSCAN halo catalogue to identify clusters
        '''
        # Get the positions in code units
        if super_verbose:
            import time
            t0 = time.time()
            print 'Loading positions...'
        pos = np.array([h['pos'].in_units('code_length') for h in self])
        ind = np.array([h['id'] for h in self])

        if super_verbose:
            print 'Got positions'
            t1 = time.time()
            print 'Took %f s' % (t1 - t0)

        # We have the data to run dbscan
        from scipy.spatial import distance
        #from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import DBSCAN
        #from sklearn import metrics
        # Compute similarities
        if super_verbose:
            print 'Computing similarities'
            t0 = time.time()
        D = distance.squareform(distance.pdist(pos))
        S = np.max(D) - D
        # print pos
        # print S
        if super_verbose:
            print 'Got similarities'
            t1 = time.time()
            print 'Took %f s' % (t1 - t0)

        if super_verbose:
            print 'Starting DBSCAN'
            t0 = time.time()
        #db = DBSCAN(eps=eps, min_samples=min_samples).fit(pos)
        db = DBSCAN(eps=eps * np.max(D), min_samples=10).fit(S)
        if super_verbose:
            print 'Finished DBSCAN'
            t1 = time.time()
            print 'Took %f s' % (t1 - t0)
        del(D)
        del(S)
        #core_samples = db.core_sample_indices_
        labels = db.labels_

        # add the particle indices to the origional array
        Y = np.insert(pos, 3, ind, axis=1)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print 'Found %d groups' % (n_clusters_)
        return db, Y

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