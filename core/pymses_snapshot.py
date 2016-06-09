from pymses import RamsesOutput
from snapshot import Snapshot, Family

class PymsesSnapshot(Snapshot):
    """
    Class for handling pymses snapshots
    """
    def __init__(self, path, ioutput, **kwargs):
        super(PymsesSnapshot, self).__init__(path, ioutput, **kwargs)
        self._ro = RamsesOutput(path, ioutput, metals=self.metals, **kwargs)

    @property
    def ro(self):
        return self._ro

    def get_source(self, family, fields):
        source = getattr(self, "%s_source" % family)
        return source(fields)

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
    def s(self):
        return Family(self, "star")

    def camera(self, **kwargs):
        from pymses.analysis import Camera
        return Camera(**kwargs)