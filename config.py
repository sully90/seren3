RAMSES_F90_DIR = '/home/d/ds/ds381/Code/ramses-rt/trunk/ramses/utils/f90/'
DATA_DIR = '/home/d/ds/ds381/apps/seren2/data/'
DEFAULT_HALO_FINDER = 'rockstar'
ROCKSTAR_BASE = 'rockstar/'
# ROCKSTAR_BASE = '../rt/rockstar/keep/'
VERBOSE = True


class Params(object):

    def __init__(self):
        self.params = {'MPI': False}

    def get(self, key):
        return self.params.get(key, None)

    def set(self, key, value):
        self.params[key] = value
        return self.get(key)


PARAMS = Params()


def get(key, default=None):
    return globals().get(key, default)
