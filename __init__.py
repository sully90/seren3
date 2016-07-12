import abc
from seren3.utils.derived_utils import *
from seren3.core import derived

_jet_black_cdict = {
        'red'   : ((0.,0.,0.),(0.30, 0.000,0.000),(0.40,0.2778,0.2778),(0.52, 0.2778,0.2778),(0.64, 1.000,1.000),(0.76, 1.000,1.000),(0.88,  0.944,0.944),(0.98, 0.500,0.500),(1.,1.,1.)),
        'green' : ((0.,0.,0.),(0.10, 0.000,0.000),(0.25,  0.389,0.389),(0.32,   0.833,0.833),(0.40, 1.000,1.000),(0.52, 1.000,1.000),(0.64,  0.803,0.803),(0.76, 0.389,0.389),(0.88, 0.000,0.000),(1.,0.,0.)),
        'blue'  : ((0.,0.00,0.00),(0.001,0.,0.),(0.07, 0.500,0.500),(0.12,  0.900,0.900),(0.23,   1.000,1.000),(0.28, 1.000,1.000),(0.40, 0.722,0.722),(0.52,0.2778,0.2778),(0.64, 0.000,0.000),(1.,0.,0.))
}

import matplotlib
import matplotlib.pylab as plt
_jet_black_cm = matplotlib.colors.LinearSegmentedColormap('jet_black', _jet_black_cdict, 1e5)
plt.register_cmap(cmap=_jet_black_cm)

matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.labelsize'] = 20

# Custom mpl styling
import config
# import matplotlib, json
# mpljson = json.load(open("%s/styles/bmh_matplotlibrc.json" % config.DATA_DIR))
# # mpljson = json.load(open("%s/styles/538.json" % config.DATA_DIR))
# matplotlib.rcParams.update(mpljson)

def init(path="./"):
    from seren3.core.simulation import Simulation
    return Simulation(path)

def load_cwd(ioutput, **kwargs):
    return load_snapshot('./', ioutput, **kwargs)

def load_snapshot(path, ioutput, **kwargs):
    from seren3.core.pymses_snapshot import PymsesSnapshot
    return PymsesSnapshot(path, ioutput, **kwargs)

class SerenIterable(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, base):
        self.base = base
        self.end = 1 if base != 0 else 0

    def __iter__(self):
        for i in range(self.base, len(self)+self.end):
            yield self[i]

    @abc.abstractmethod
    def __len__(self):
        return