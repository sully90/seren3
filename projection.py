import seren3
from seren3.analysis import visualization
from seren3.utils import camera_utils
import sys
import numpy as np

iout = int(sys.argv[1])
field = sys.argv[2]

snap = seren3.load_cwd(iout)
cam = snap.camera(region_size=np.array([.5, .5]), distance=.5, far_cut_depth=.5)
gal_ax = camera_utils.find_galaxy_axis(snap, camera=cam)
cam.los_axis = gal_ax

proj = visualization.Projection(snap.g, field, camera=cam, vol_weighted=False)
im = proj.save_plot(fraction=0.01)
im.show()

