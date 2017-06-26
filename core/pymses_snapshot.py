import seren3
import pymses
from snapshot import Snapshot, Family

class PymsesSnapshot(Snapshot):
    """
    Class for handling pymses snapshots
    """
    def __init__(self, path, ioutput, ro=None, verbose=False, **kwargs):
        super(PymsesSnapshot, self).__init__(path, ioutput, **kwargs)

        from pymses import rcConfig as pymsesrc
        self.verbose = verbose
        if ro is None:
            self._ro = pymses.RamsesOutput(path, ioutput, metals=self.metals, verbose=self.verbose, **kwargs)
        else:
            self._ro = ro
        self._nproc_multiprocessing = pymsesrc.multiprocessing_max_nproc

    def __getitem__(self, item):
        from seren3.halos import Halo
        if hasattr(item, '__module__') and (item.__module__ == 'pymses.utils.regions'):
            return PymsesSubSnapshot(self, item)
        elif isinstance(item, Halo):
            sphere = item.sphere
            return PymsesSubSnapshot(self, sphere)
        else:
            raise ValueError("Unknown item: ", item)

    def __str__(self):
        return "PymsesSnapshot: %05i:z=%1.2f" % (self.ioutput, self.z)

    def __repr__(self):
        return self.__str__()

    @property
    def ro(self):
        return self._ro

    def get_io_source(self, family, fields):
        source = getattr(self, "%s_source" % family)
        return source(fields)

    def get_sphere(self, pos, r):
        from pymses.utils.regions import Sphere
        return Sphere(pos, r)

    def get_cube(self, pos, l):
        '''
        Return a cube region object encasing this halo
        '''
        from pymses.utils.regions import Cube
        return Cube(pos, l)

    def amr_source(self, fields):
        if not hasattr(fields, "__iter__"):
            fields = [fields]
        return self.ro.amr_source(fields)

    def part_source(self, fields):
        if not hasattr(fields, "__iter__"):
            fields = [fields]
        return self.ro.particle_source(fields)

    def dm_source(self, fields):
        if not hasattr(fields, "__iter__"):
            fields = [fields]
        return self.ro.particle_source(fields, select_stars=False)

    def star_source(self, fields):
        if not hasattr(fields, "__iter__"):
            fields = [fields]
        return self.ro.particle_source(fields, select_dark_matter=False)

    def gmc_source(self, fields):
        if not hasattr(fields, "__iter__"):
            fields = [fields]
        return self.ro.particle_source(fields, select_stars=False, select_dark_matter=False, select_gmc=True)


    @property
    def g(self):
        return Family(self, "amr")

    @property
    def p(self):
        return Family(self, "part")

    @property
    def d(self):
        return Family(self, "dm")

    @property
    def s(self):
        return Family(self, "star")

    @property
    def gmc(self):
        return Family(self, "gmc")

    def camera(self, **kwargs):
        from pymses.analysis import Camera
        if "map_max_size" not in kwargs:
            kwargs["map_max_size"] = 2**self.info["levelmin"]
        if "size_unit" not in kwargs:
            kwargs["size_unit"] = self.info["unit_length"]

        return Camera(**kwargs)

    def get_nproc(self):
        return self._nproc_multiprocessing

    def set_nproc(self, nproc):
        self._nproc_multiprocessing = nproc
        return self.get_nproc()

class PymsesSubSnapshot(PymsesSnapshot):
    def __init__(self, pymses_snapshot, region):
        super(PymsesSubSnapshot, self).__init__(pymses_snapshot.path, pymses_snapshot.ioutput, ro=pymses_snapshot.ro)
        self._base = pymses_snapshot
        self.region = region

    @property
    def info(self):
        return self._base.info

    @property
    def friedmann(self):
        return self._base.friedmann

    def camera(self, **kwargs):
        from pymses.analysis import Camera

        width = None
        if hasattr(self.region, "width"):
            width = self.region.width
        elif hasattr(self.region, "radius"):
            width = 2 * self.region.radius
        else:
            raise Exception("Could not determine width for camera object (region is: %s)" % self.region)

        kwargs["center"] = self.region.center
        if "region_size" not in kwargs:
            kwargs["region_size"] = [width, width]
        if "distance" not in kwargs:
            kwargs["distance"] = width/2
        if "far_cut_depth" not in kwargs:
            kwargs["far_cut_depth"] = width/2
        if "map_max_size" not in kwargs:
            kwargs["map_max_size"] = min(2**self.info["levelmax"], 1024)

        return super(PymsesSubSnapshot, self).camera(**kwargs)

        # return Camera(center=center, region_size=region_size, \
        #         distance=distance, far_cut_depth=far_cut_depth, \
        #         map_max_size=map_max_size, **kwargs)

    def pynbody_snapshot(self, filt=False, remove_gmc=True):
        '''
        Load a pynbody snapshot using only the CPUs that bound this subsnapshot
        '''
        import pynbody
        import numpy as np

        bbox = self.region.get_bounding_box()
        cpus = self.cpu_list(bbox)

        s = pynbody.load("%s/output_%05i/" % (self.path, self.ioutput), cpus=cpus)

        # Center on region center
        s['pos'] -= self.region.center

        # Alias metals field
        if s.has_key('metal'):
            s['metals'] = s['metal']
        else:
            s['metals'] = np.ones(len(s)) * 1e-6

        if filt:
            # Filter data to a sphere
            s = s[pynbody.filt.Sphere(self.region.radius)]

        # Deal with GMC particles and stellar age
        # if all(i > 0 for i in s.s['iord']) and remove_gmc:
            # We have GMC particles -> setup new family
            # gmc = pynbody.family.Family("gmc")  # not working
            # s.gmc = s.s[np.where(s.s['iord'] < 0)]

        # Just remove GMC for now, until a better solution is found
        s.s = s.s[s.s['iord'] >= 0]

        # Create age field for star particles
        if len(s.s) > 0:
            from pynbody.array import SimArray
            from seren3.core.derived.part_derived import part_age
            tform = s.s['tform']
            age = part_age(self, {'epoch':tform})
            age = SimArray(age, 'Gyr')
            s.s['age'] = age

        return s
