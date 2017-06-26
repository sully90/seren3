def load():
    import pickle
    proj_list1 = pickle.load( open("./proj_list1.p", "rb") )
    proj_list2 = pickle.load( open("./proj_list2.p", "rb") )

    vrange1 = pickle.load( open("./vrange1.p", "rb") )
    vrange2 = pickle.load( open("./vrange2.p", "rb") )

    extent = pickle.load( open("./extent.p", "rb") )
    zlist = pickle.load( open("./zlist.p", "rb") )

    return proj_list1, proj_list2, vrange1, vrange2, extent, zlist

def write_both(context, proj_list1, proj_list2, z, extent, vrange1, vrange2):
    import matplotlib
    matplotlib.use('Agg')

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from seren3.utils import plot_utils
    cm = plot_utils.load_custom_cmaps('blues_black_test')

    field = "rho_Gamma"

    cwd = os.getcwd()
    out_dir = "%s/images/" % cwd

    C = context.C
    info = context.info
    unit_sd = (info["unit_density"] * info["unit_length"]).express(C.Msun * C.kpc**-2)  # surface density unit

    count=0
    for proj1, proj2 in zip(proj_list1, proj_list2):
        print count
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        proj_map1 = proj1.map.T * unit_sd
        proj_map2 = proj2.map.T

        idx = np.where(proj_map2 <= 0)
        proj_map2[idx] = 1e-6

        im1 = axs[0].imshow(np.log10(proj_map1), cmap="RdBu_r", extent=extent, vmin=vrange1[0], vmax=vrange1[1])
        im2 = axs[1].imshow(np.log10(proj_map2), cmap=cm, extent=extent, vmin=vrange2[0], vmax=vrange2[1])

        cbar1 = fig.colorbar(im1, ax=axs[0])
        cbar1.set_label(r"log$_{10}$ $\rho$ [M$_{\odot}$ kpc$^{-2}$]")

        cbar2 = fig.colorbar(im2, ax=axs[1])
        cbar2.set_label(r"log$_{10}$ $\Gamma$ [s$^{-1}$]")

        text_pos = (-75, 75)
        for ax in axs.flatten():
            ax.set_xlabel(r"x [kpc a h$^{-1}$]")
            ax.set_ylabel(r"y [kpc a h$^{-1}$]")
            ax.text(text_pos[0], text_pos[1], "z = %1.2f" % z[count], color="w", size="x-large")

        fig.tight_layout()

        plt.savefig("%s/%s_%05i.png" % (out_dir, field, count))
        count += 1

        plt.close(fig)
        # plt.clf()


def write_rho(context, proj_list, z, extent, vrange):
    import matplotlib
    matplotlib.use('Agg')

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    field = "rho"

    cwd = os.getcwd()
    out_dir = "%s/images/" % cwd

    C = context.C
    info = context.info
    unit_sd = (info["unit_density"] * info["unit_length"]).express(C.Msun * C.kpc**-2)  # surface density unit

    count=0
    for proj in proj_list:
        fig = plt.figure()

        proj_map = proj.map.T * unit_sd

        plt.imshow(np.log10(proj_map), cmap="RdBu_r", extent=extent, vmin=vrange[0], vmax=vrange[1])
        cbar = plt.colorbar()
        cbar.set_label(r"log$_{10}$ $\rho$ [M$_{\odot}$ kpc$^{-2}$]")

        plt.xlabel(r"x [kpc a h$^{-1}$]")
        plt.ylabel(r"y [kpc a h$^{-1}$]")

        text_pos = (-75, 75)
        plt.text(text_pos[0], text_pos[1], "z = %1.2f" % z[count], color="w", size="x-large")

        plt.savefig("%s/%s_%05i.png" % (out_dir, field, count))
        count += 1

        plt.close(fig)


def write_Gamma(context, proj_list, z, extent, vrange):
    import matplotlib
    matplotlib.use('Agg')

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from seren3.utils import plot_utils
    cm = plot_utils.load_custom_cmaps('blues_black_test')

    field = "Gamma"

    cwd = os.getcwd()
    out_dir = "%s/images/" % cwd

    C = context.C
    info = context.info
    unit = C.s**-1

    count=0
    for proj in proj_list:
        fig = plt.figure()

        proj_map = proj.map.T
        idx = np.where(proj_map <= 0)
        proj_map[idx] = 1e-6

        print proj_map.min()

        plt.imshow(np.log10(proj_map), cmap=cm, extent=extent, vmin=vrange[0], vmax=vrange[1])
        cbar = plt.colorbar()
        cbar.set_label(r"log$_{10}$ $\Gamma$ [s$^{-1}$]")

        plt.xlabel(r"x [kpc a h$^{-1}$]")
        plt.ylabel(r"y [kpc a h$^{-1}$]")

        text_pos = (-75, 75)
        plt.text(text_pos[0], text_pos[1], "z = %1.2f" % z[count], color="w", size="x-large")

        plt.savefig("%s/%s_%05i.png" % (out_dir, field, count))
        count += 1

        plt.close(fig)


def get_rho_proj_list(halo_catalogue, h, back_to_aexp=0., **kwargs):
    '''
    Iterate through progenitors and make projections
    '''
    import seren3
    import numpy as np
    from seren3.analysis.visualization import engines, operators
    from seren3.utils import camera_utils

    field = "rho"
    proj_list = []

    C = h.base.C
    info = h.info

    length_fac = 2.
    camera = h.camera()
    camera.region_size *= length_fac
    camera.distance *= length_fac
    camera.far_cut_depth *= length_fac

    extent = camera_utils.extent(h.base, camera)

    unit_sd = (info["unit_density"] * info["unit_length"]).express(C.Msun * C.kpc**-2)  # surface density unit
    kwargs["surf_qty"] = True

    for iout in [116, 115, 114, 113, 112, 111, 110, 109, 108, 107]:
        print iout
        tmp = seren3.load_snapshot(h.base.path, iout)
        
        eng = engines.SurfaceDensitySplatterEngine(tmp.g)
        # op = operators.MassWeightedOperator(field, info["unit_density"])
        # eng = engines.CustomSplatterEngine(h.g, field, op, extra_fields=['rho'])
        proj = eng.process(camera, **kwargs)
        proj_list.append(proj)

    eng = engines.SurfaceDensitySplatterEngine(h.g)
    # op = operators.MassWeightedOperator(field, info["unit_density"])
    # eng = engines.CustomSplatterEngine(h.g, field, op, extra_fields=['rho'])
    proj = eng.process(camera, **kwargs)

    proj_map = proj.map.T * unit_sd
    vmin, vmax = (np.log10(proj_map.min()), np.log10(proj_map.max()))

    proj_list.append(proj)

    for prog in halo_catalogue.iterate_progenitors(h, back_to_aexp=back_to_aexp):
        # camera.center = prog.subsnap.region.center
        info = prog.info

        # op = operators.MassWeightedOperator(field, info["unit_density"])
        eng = engines.SurfaceDensitySplatterEngine(prog.g)
        # eng = engines.CustomSplatterEngine(prog.g, field, op, extra_fields=['rho'])
        proj_list.append(eng.process(camera, **kwargs))

    return proj_list[::-1], extent, (vmin, vmax)

def get_Gamma_proj_list(halo_catalogue, h, back_to_aexp=0., **kwargs):
    '''
    Iterate through progenitors and make projections
    '''
    import seren3
    import numpy as np
    from seren3.analysis.visualization import engines, operators
    from seren3.utils import camera_utils

    field = "Gamma"
    proj_list = []

    C = h.base.C
    info = h.info

    length_fac = 2.
    camera = h.camera()
    camera.region_size *= length_fac
    camera.distance *= length_fac
    camera.far_cut_depth *= length_fac

    extent = camera_utils.extent(h.base, camera)

    unit_Gamma = C.s**-1
    kwargs["surf_qty"] = True

    for iout in [116, 115, 114, 113, 112, 111, 110, 109, 108, 107]:
        print iout
        tmp = seren3.load_snapshot(h.base.path, iout)
        # eng = engines.SurfaceDensitySplatterEngine(h.g)
        eng = engines.MassWeightedSplatterEngine(tmp.g, field)
        proj = eng.process(camera, **kwargs)
        proj_list.append(proj)

    # op = operators.MassWeightedOperator(field, unit_Gamma)
    # eng = engines.SurfaceDensitySplatterEngine(h.g)
    eng = engines.MassWeightedSplatterEngine(h.g, field)
    proj = eng.process(camera, **kwargs)

    proj_map = proj.map.T
    vmin, vmax = (np.log10(proj_map.min()), np.log10(proj_map.max()))

    proj_list.append(proj)

    for prog in halo_catalogue.iterate_progenitors(h, back_to_aexp=back_to_aexp):
        # camera.center = prog.subsnap.region.center
        info = prog.info

        # op = operators.MassWeightedOperator(field, unit_Gamma)
        # eng = engines.SurfaceDensitySplatterEngine(prog.g)
        eng = engines.MassWeightedSplatterEngine(prog.g, field)
        proj_list.append(eng.process(camera, **kwargs))

    return proj_list[::-1], extent, (vmin, vmax)
