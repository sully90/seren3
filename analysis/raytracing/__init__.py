r"""
:mod:`seren3.analysis.raytracing` --- 2D map ray-casting module
===============================================================

"""
from optical_depth_tracer import GunnPetersonOpticalDepthTracer

def generate_tau_map(snap, show=False):
    '''
    Ray-traces optical depth and plots as a 2D map
    '''
    import numpy as np
    from pymses.analysis.splatting.map_bin2d import histo2D

    odt = GunnPetersonOpticalDepthTracer(snap)
    cam = snap.camera()
    cells = odt.process(cam)

    tau = cells["tau"]
    pts = np.zeros_like(cells.points)
    pts[:, :2] , pts[:, 2] = cam.project_points(cells.points)
    centered_map_box = cam.get_map_box()
    map_range = np.array([[centered_map_box.min_coords[0], centered_map_box.max_coords[0]],
                             [centered_map_box.min_coords[1], centered_map_box.max_coords[1]],
                             [centered_map_box.min_coords[2], centered_map_box.max_coords[2]]])

    map = histo2D(pts, [256, 256], map_range, {"tau": tau})

    map_unit = snap.info['unit_density'] * snap.info['unit_length']
    map = np.log10(map["tau"]*map_unit.express(snap.C.Msun / snap.C.kpc**2))

    if show:
        from matplotlib import pyplot as plt
        from matplotlib.ticker import FormatStrFormatter, IndexLocator
        vmin = np.min(map[map > 0.])
        plt.imshow(map, origin='lower')
        fo = FormatStrFormatter("$10^{%d}$")
        offset = np.ceil(vmin) - vmin
        lo = IndexLocator(1.0, offset)
        cb = plt.colorbar(ticks=lo, format=fo)
        # Set colorbar lable
        cb.set_label("$M_{\odot}/kpc^{2}$")
        plt.show(block=False)

    return map