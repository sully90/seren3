import seren3
import pymses
from snapshot import Snapshot, Family

class PymsesSnapshot(Snapshot):
    """
    Class for handling pymses snapshots
    """
    def __init__(self, path, ioutput, ro=None, **kwargs):
        super(PymsesSnapshot, self).__init__(path, ioutput, **kwargs)
        if ro is None:
            self._ro = pymses.RamsesOutput(path, ioutput, metals=self.metals, **kwargs)
        else:
            self._ro = ro

    def __getitem__(self, item):
        if hasattr(item, '__module__') and (item.__module__ == 'pymses.utils.regions'):
            return PymsesSubSnapshot(self, item)
        elif isinstance(item, seren3.halos.Halo):
            sphere = item.sphere
            return PymsesSubSnapshot(self, sphere)
        else:
            raise ValueError("Unknown item: ", item)

    @property
    def ro(self):
        return self._ro

    def get_source(self, family, fields):
        source = getattr(self, "%s_source" % family)
        return source(fields)

    def get_sphere(self, pos, r):
        from pymses.utils.regions import Sphere
        return Sphere(pos, r)

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
        raise NotImplementedError("gmc particle selection not implemented for pymses_4.1.3")


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

    def camera(self, **kwargs):
        from pymses.analysis import Camera
        return Camera(**kwargs)

    def get_nproc(self):
        return pymses.utils.misc.NUMBER_OF_PROCESSES_LIMIT

    def set_nproc(self, nproc):
        pymses.utils.misc.NUMBER_OF_PROCESSES_LIMIT = nproc
        return self.get_nproc()

class PymsesSubSnapshot(PymsesSnapshot):
    def __init__(self, pymses_snapshot, region):
        super(PymsesSubSnapshot, self).__init__(pymses_snapshot.path, pymses_snapshot.ioutput, ro=pymses_snapshot.ro)
        self.region = region

    def camera(self, **kwargs):
        from pymses.analysis import Camera

        center = self.region.center
        radius = self.region.radius

        region_size = [radius, radius]
        distance = radius
        far_cut_depth = radius
        map_max_size = kwargs.pop("map_max_size", 1024)

        return Camera(center=center, region_size=region_size, \
                distance=distance, far_cut_depth=far_cut_depth, \
                map_max_size=map_max_size, **kwargs)

    def pynbody_snapshot(self, filt=False):
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

        return s