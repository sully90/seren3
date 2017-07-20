def tmp(h1, h2, av_z=False, vc='k', rotate=False, out_units="Msol yr**-1 pc**-2", **kwargs):
    import matplotlib.pylab as plt
    from matplotlib.ticker import IndexLocator, FormatStrFormatter
    from matplotlib.colors import Colormap, LinearSegmentedColormap
    from matplotlib.patches import Circle
    import pynbody
    from pynbody.plot import sph

    def anno(ax, xy, rvir, color="lightsteelblue", facecolor="none", alpha=1):
            e = Circle(xy=xy, radius=rvir)

            ax.add_artist( e )
            e.set_clip_box( ax.bbox )
            e.set_edgecolor( color )
            e.set_facecolor( facecolor )  # "none" not None
            e.set_alpha( alpha )

    xy=(0,0)
    nrows=3
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12,16))

    # fig.subplots_adjust(hspace=0.001)
    # fig.subplots_adjust(wspace=0.9)

    def _load_s(h):
        s = h.pynbody_snapshot(filt=False)
        return s

    if "s" in kwargs:
        s = kwargs.pop("s")
    else:
        s = [_load_s(hi) for hi in [h1, h2]]
        if rotate:
            # tx = pynbody.analysis.angmom.sideon(s[1].s[pynbody.filt.Sphere(radius='2 kpc')])
            # tx.apply_to(s[0])
            for s_h in s:
                pynbody.analysis.angmom.sideon(s_h.s[pynbody.filt.Sphere(radius='2 kpc')])

    for row, field in zip(range(nrows), ["rho", "mass_flux_radial", "rad_flux_radial"]):
        axs = axes[row,:]
        print len(axs.flatten()), field
        for h, s_h, ax in zip([h1, h2], s, axs.flatten()):
                
            s_h.physical_units()
            rvir = h.rvir.in_units(s_h.g['pos'].units)
            width = '%1.2f kpc' % (2.1*rvir)
            s_h.g['rad_flux_radial'].convert_units("s**-1 m**-2")
            s_h.g['rad_0_rho'].convert_units("s**-1 m**-2")

            print "Rvir = %1.2f kpc" % rvir.in_units("kpc")

            if field == "mass_flux_radial":
                sph.velocity_image(s_h.g, qty="outflow_rate", units=out_units,\
                             width=width, av_z=av_z, cmap="RdBu_r", qtytitle=r"$\vec{F}_{\mathrm{M}_{\mathrm{gas}}}$",\
                             subplot=ax, quiverkey=False, vector_color=vc, vector_resolution=20, key_length="1000 km s**-1", **kwargs)
            elif field == "rad_flux_radial":
                # sph.velocity_image(s_h.g, qty="rad_flux_radial", units="s**-1 m**-2",\
                #              width=width, cmap="RdBu_r", qtytitle=r"$\vec{F}_{\mathrm{ion}}$",\
                #              subplot=ax, quiverkey=False, vector_color=vc, vector_resolution=20, key_length="1000 km s**-1")

                sph.velocity_image(s_h.g, qty="rad_0_rho", units="s**-1 m**-2",\
                             width=width, cmap="RdBu_r", qtytitle=r"$\vec{F}_{\mathrm{ion}}$",\
                             subplot=ax, quiverkey=False, vector_color=vc, vector_resolution=20, key_length="1000 km s**-1", vmin=1e8, vmax=1e12)

                # sph.velocity_image(s_h.g, qty="temp", units="K",\
                #              width=width, av_z=av_z, cmap="jet", qtytitle=r"T",\
                #              subplot=ax, quiverkey=False, vector_color="k", vector_resolution=20, key_length="1000 km s**-1")
            else:
                sph.velocity_image(s_h.g, qty=field, width=width, cmap="RdBu_r", qtytitle=r"$\rho$$_{\mathrm{gas}}$",\
                         vector_resolution=20, vector_color=vc, key_x=0.35, key_y=0.815, key_color='yellow', key_length="250 km s**-1", units="Msol kpc**-2", subplot=ax)
            anno(ax, xy, rvir)
            anno(ax, xy, rvir/2., color="r")
    return s

def tmp2(h1, av_z=False, vc='k', rotate=False, out_units="Msol yr**-1 pc**-2", **kwargs):
    import matplotlib.pylab as plt
    from matplotlib.ticker import IndexLocator, FormatStrFormatter
    from matplotlib.colors import Colormap, LinearSegmentedColormap
    from matplotlib.patches import Circle
    import pynbody
    from pynbody.plot import sph

    def anno(ax, xy, rvir, color="lightsteelblue", facecolor="none", alpha=1):
            e = Circle(xy=xy, radius=rvir)

            ax.add_artist( e )
            e.set_clip_box( ax.bbox )
            e.set_edgecolor( color )
            e.set_facecolor( facecolor )  # "none" not None
            e.set_alpha( alpha )

    xy=(0,0)
    nrows=3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(6,10))

    # fig.subplots_adjust(hspace=0.001)
    # fig.subplots_adjust(wspace=0.9)

    def _load_s(h):
        s = h.pynbody_snapshot(filt=False)
        return s

    if "s" in kwargs:
        s = kwargs.pop("s")
    else:
        s = _load_s(h1)
        if rotate:
            # tx = pynbody.analysis.angmom.sideon(s[1].s[pynbody.filt.Sphere(radius='2 kpc')])
            # tx.apply_to(s[0])
            pynbody.analysis.angmom.sideon(s.s[pynbody.filt.Sphere(radius='2 kpc')])

    for ax, field in zip(axs.flatten(), ["rho", "mass_flux_radial", "rad_flux_radial"]):
                
        s.physical_units()
        rvir = h1.rvir.in_units(s.g['pos'].units)
        width = '%1.2f kpc' % (2.1*rvir)
        s.g['rad_flux_radial'].convert_units("s**-1 m**-2")
        s.g['rad_0_rho'].convert_units("s**-1 m**-2")

        print "Rvir = %1.2f kpc" % rvir.in_units("kpc")

        if field == "mass_flux_radial":
            sph.velocity_image(s.g, qty="outflow_rate", units=out_units,\
                         width=width, av_z=av_z, cmap="RdBu_r", qtytitle=r"$\vec{F}_{\mathrm{M}_{\mathrm{gas}}}$",\
                         subplot=ax, quiverkey=False, vector_color=vc, vector_resolution=20, key_length="1000 km s**-1", **kwargs)
        elif field == "rad_flux_radial":
            # sph.velocity_image(s.g, qty="rad_flux_radial", units="s**-1 m**-2",\
            #              width=width, cmap="RdBu_r", qtytitle=r"$\vec{F}_{\mathrm{ion}}$",\
            #              subplot=ax, quiverkey=False, vector_color=vc, vector_resolution=20, key_length="1000 km s**-1")

            sph.velocity_image(s.g, qty="rad_0_rho", units="s**-1 m**-2",\
                         width=width, cmap="RdBu_r", qtytitle=r"$\vec{F}_{\mathrm{ion}}$",\
                         subplot=ax, quiverkey=False, vector_color=vc, vector_resolution=20, key_length="1000 km s**-1", vmin=1e8, vmax=1e12)

            # sph.velocity_image(s.g, qty="temp", units="K",\
            #              width=width, av_z=av_z, cmap="jet", qtytitle=r"T",\
            #              subplot=ax, quiverkey=False, vector_color="k", vector_resolution=20, key_length="1000 km s**-1")
        else:
            sph.velocity_image(s.g, qty=field, width=width, cmap="RdBu_r", qtytitle=r"$\rho$$_{\mathrm{gas}}$",\
                     vector_resolution=20, vector_color=vc, key_x=0.35, key_y=0.815, key_color='yellow', key_length="250 km s**-1", units="Msol kpc**-2", subplot=ax)
        anno(ax, xy, rvir)
        anno(ax, xy, rvir/2., color="r")
    return s

def obs_errors(quantity, ax=None):
    '''
    @author Keri Dixon
    Plot observational constaints
    '''
    import matplotlib as mpl
    import matplotlib.pylab as plt
    import numpy as np

    if (ax is None):
        ax = plt.gca()

    # mpl.rcParams['axes.linewidth'] = 1.5
    # mpl.rcParams['xtick.labelsize'] = 18
    # mpl.rcParams['ytick.labelsize'] = 18
    # mpl.rcParams['axes.labelsize'] = 28
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.scatterpoints'] = 1
    mpl.rcParams['legend.fontsize'] = 8
    # mpl.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['ps.fonttype'] = 42

    if quantity == 'tau':
        z_p = range(5,22)
        tau_p = 0.058*np.ones(len(z_p)) # Planck (2016)
        upper_tau = (0.058+0.012)
        lower_tau = (0.058-0.012)

        ax.plot(z_p,tau_p,c='dimgray',lw=0.6)
        ax.axhspan(lower_tau, upper_tau, facecolor='gainsboro', edgecolor='none')

    elif quantity == 'Gamma':
        p1 = [0.18*1e-12] #Wyithe Bolton 2011 , lower then upper error
        # e1 = [[0.18*1e-12], [0.09e-12]]
        e1= [[0.3],[0.3]]
        z1 = [6.18]

        p2 = [(10**-12.84)] #Calverley 2011
        # e2 = [[10**(-12.84+0.18)-10**-12.84],[10**(-12.84+0.18)-10**-12.84]]
        e2= [[0.18],[0.18]]
        z2 = [5.9]

        ax.errorbar(z1, np.log10(p1), yerr=e1, c='royalblue',fmt='s',ls='-',lw=1,markeredgecolor='royalblue')
        ax.errorbar(z2, np.log10(p2), yerr=e2, c='forestgreen',fmt='^',ls='-',lw=1,markeredgecolor='forestgreen')
        ax.scatter(z1, np.log10(p1), c='royalblue',marker='s',s=70,lw=0,label='$\\rm Wyithe\;&\;Bolton\;(2011)$')
        ax.scatter(z2, np.log10(p2), c='forestgreen',marker='^',s=70,lw=0,label='$\\rm Calverley\;et\;al.\;(2011)$')

        # ax.errorbar(z1, p1, yerr=e1, c='royalblue',fmt='s',ls='-',lw=1,markeredgecolor='royalblue')
        # ax.errorbar(z2, p2, yerr=e2, c='forestgreen',fmt='^',ls='-',lw=1,markeredgecolor='forestgreen')
        # ax.scatter(z1, p1, c='royalblue',marker='s',s=70,lw=0,label='$\\rm Wyithe\;&\;Bolton\;(2011)$')
        # ax.scatter(z2, p2, c='forestgreen',marker='^',s=70,lw=0,label='$\\rm Calverley\;et\;al.\;(2011)$')

    elif quantity == 'xv':
        fan_z1 = 6.1       # Fan et al 2006 (lower limit)
        fan_x1 = 4.1e-4
        fan_l1 = 3.e-4
        fan_z2 = 5.85
        fan_x2 = 1.1e-4
        fan_l2 = 3.e-5
        McGr_z1 = 6.1      # McGreer et al 2011 (upper limit)
        McGr_x1 = 0.5
        McGr_z2 = 5.9      # McGreer et al 2015 (upper limit)
        McGr_x2 = 0.06
        McGr_u2 = 0.05
        Schr_z1 = 6.2      # Schroeder et al 2013 (lower limit)
        Schr_x1 = 0.1
        McQu_z1 = 6.3      # McQuinn et al 2008 (upper limit)
        McQu_x1 = 0.5
        Chor_z1 = 5.91     # Chornock et al 2013 (upper limit)
        Chor_x1 = 0.11
        Ota_z1 = 7.        # Ota et al 2008
        Ota_u1 = 0.26
        Ota_l1 = 0.26
        Ota_x1 = 0.38
        Ouch_z1 = 6.6      # Ouchi et al 2010 (upper limit)
        Ouch_x1 = 0.2
        Ouch_ul = 0.2
        Ouch_z2 = 6.6      # Ouchi et al 2010 (upper limit) clustering
        Ouch_x2 = 0.5

        ax.errorbar(fan_z1,fan_x1,yerr=[[fan_l1],[0.05]],lolims=True,marker='s', c='firebrick',mec='firebrick')
        ax.errorbar(fan_z2,fan_x2,yerr=[[fan_l2],[0.05]],lolims=True,marker='s', c='firebrick',mec='firebrick')
        ax.errorbar(McGr_z1,McGr_x1,0.05,uplims=True,marker='^', c='forestgreen',mec='forestgreen')
        ax.errorbar(McGr_z2,McGr_x2,yerr=[[0.05],[McGr_u2]],uplims=True,marker='^', c='forestgreen',mec='forestgreen')
        ax.errorbar(Schr_z1,Schr_x1,0.05,lolims=True,marker='o', c='royalblue',mec='royalblue')
        ax.errorbar(McQu_z1,McQu_x1,0.05,uplims=True,marker='D', c='purple',mec='purple')
        ax.errorbar(Chor_z1,Chor_x1,0.05,uplims=True,marker='D', c='purple',mec='purple')
        ax.errorbar(Ota_z1,Ota_x1,Ota_l1,marker='h', c='darkturquoise',mec='darkturquoise')
        ax.errorbar(Ouch_z1,Ouch_x1,0.05,uplims=True,marker='h', c='darkturquoise',mec='darkturquoise')
        ax.errorbar(Ouch_z2,Ouch_x2,0.05,uplims=True,marker='p', c='darkorange',mec='darkorange')

        ax.scatter(fan_z1,fan_x1,marker='s',c='firebrick',lw=0,s=40,label='$\\rm Ly\,\\alpha\;forest\;transmission$')
        ax.scatter(fan_z2,fan_x2,marker='s',c='firebrick',lw=0,s=40)
        ax.scatter(McGr_z1,McGr_x1,marker='^',c='forestgreen',lw=0,s=40,label='$\\rm Dark\;Ly\,\\alpha\;pixels$')
        ax.scatter(McGr_z2,McGr_x2,marker='^',c='forestgreen',lw=0,s=40)
        ax.scatter(Schr_z1,Schr_x1,marker='o',c='royalblue',lw=0,s=40,label='$\\rm Quasar\;near\;zones$')
        ax.scatter(McQu_z1,McQu_x1,marker='D',c='purple',lw=0,s=40,label='$\\rm GRB\;damping\;wing$')
        ax.scatter(Chor_z1,Chor_x1,marker='D',c='purple',lw=0,s=40)
        ax.scatter(Ota_z1,Ota_x1,marker='h',c='darkturquoise',lw=0,s=40,label='$\\rm Ly\,\\alpha\;emitters$')
        ax.scatter(Ouch_z1,Ouch_x1,marker='h',c='darkturquoise',lw=0,s=40)
        ax.scatter(Ouch_z2,Ouch_x2,marker='p',c='darkorange',s=40,lw=0,label='$\\rm Ly\,\\alpha\;clustering$')

    else:
        print 'ERROR: Invalid quantity flag'

def annotate_rvir(ax, rvir, color="lightsteelblue", facecolor="none", alpha=1):
    '''
    Annotates the virial radius on the axis.
    Assumes zero centred, rvir and extent must have the same units
    '''
    from matplotlib.patches import Circle
    
    xy = (0, 0)
    e = Circle(xy=xy, radius=rvir)

    ax.add_artist( e )
    e.set_clip_box( ax.bbox )
    e.set_edgecolor( color )
    e.set_facecolor( facecolor )  # "none" not None
    e.set_alpha( alpha )

def fit_scatter(x, y, ret_n=False, ret_sterr=False, nbins=10):
    '''
    Bins scattered points and fits with error bars
    '''
    import numpy as np

    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)

    bin_centres = (_[1:] + _[:-1])/2.

    if ret_sterr:
        stderr = std/np.sqrt(n)
        if ret_n:
            return bin_centres, mean, std, stderr, n
        return bin_centres, mean, std, stderr

    if ret_n:
        return bin_centres, mean, std, n
    return bin_centres, mean, std


def fit_median(X, Y, nbins=10):
    '''
    Fit the median line to a scatter
    '''

    import numpy as np
    bins = np.linspace(X.min(),X.max(), nbins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(X,bins)
    running_median = [np.median(Y[idx==k]) for k in range(nbins)]

    bc = bins-delta/2

    return bc, running_median


def grid(x, y, z, resX=100, resY=100):
    """
    Convert 3 column data to matplotlib grid.
    Can be used to convert x/y scatter with z value to correct format for plt.contour
    """
    import numpy as np
    from matplotlib.mlab import griddata

    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = np.meshgrid(xi, yi)
    return X, Y, Z

# def plot_RT2_self_shielding(snapshot, csvfilename, projections=None):
def plot_self_shielding(h, projections=None):
    import matplotlib

    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['axes.labelsize'] = 18

    import numpy as np
    import matplotlib.pylab as plt
    from matplotlib.ticker import IndexLocator, FormatStrFormatter
    from matplotlib.colors import Colormap, LinearSegmentedColormap
    from matplotlib.patches import Circle
    from seren3.analysis.visualization import engines, operators
    from seren3.utils import camera_utils, plot_utils

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    from matplotlib_scalebar.scalebar import ScaleBar
    from matplotlib_scalebar.dimension import SILengthDimension

    reload(engines)

    snapshot = h.base
    C = snapshot.C

    # Use snap 100 of RT2, shalos[1]

    # subsnap = camera_utils.legacy_cam_to_region(snapshot, csvfilename)
    # camera = subsnap.camera()
    subsnap = h.subsnap
    camera = subsnap.camera()
    camera_zoom = subsnap.camera()

    camera.map_max_size = 512
    camera_zoom.map_max_size = 1024
    camera_zoom.region_size *= .25
    camera_zoom.distance *= .25
    camera_zoom.far_cut_depth *= .25

    los = camera_utils.find_los(subsnap, camera=camera)
    camera.los_axis = los
    camera_zoom.los_axis = los

    length_fac = 3.
    camera.region_size *= length_fac
    camera.distance *= length_fac
    camera.far_cut_depth *= length_fac

    rvir = h["rvir"].in_units("kpc")
    dx = snapshot.array(camera.region_size[0], subsnap.info["unit_length"]).in_units("kpc")

    cm = plot_utils.load_custom_cmaps('blues_black_test')

    # fig = plt.figure(figsize=(6.5,12))

    eng_zoom = engines.SurfaceDensitySplatterEngine(subsnap.g)
    proj_zoom = eng_zoom.process(camera_zoom, random_shift=True)

    unit_sd = subsnap.info["unit_density"] * subsnap.info["unit_length"]
    unit_sd = unit_sd.express(C.Msun * C.kpc**-2)
    proj_zoom_map = proj_zoom.map * unit_sd

    if (projections is None):
        projections = []
        for i in range(4):  # 3 plots
            if i == 0:
                # Density
                eng = engines.SurfaceDensitySplatterEngine(subsnap.g)
                proj = eng.process(camera, random_shift=True)

                projections.append(proj)
            elif i == 1:
                # op = operators.MinTempOperator(C.K)
                # eng = engines.MassWeightedSplatterEngine(subsnap.g, "T")
                eng = engines.RayTraceMinTemperatureEngine(subsnap.g)
                proj = eng.process(camera)

                projections.append(proj)
            elif i == 2:
                op = operators.MinxHIIOperator(C.none)
                eng = engines.CustomRayTraceEngine(subsnap.g, "xHII", op)

                proj = eng.process(camera)

                projections.append(proj)

    # gs = gridspec.GridSpec(2, 4)
    # ax1 = plt.subplot(gs[0, 0:2])
    # ax2 = plt.subplot(gs[0,2:])
    # ax3 = plt.subplot(gs[1,1:3])

    # axs = [ax1, ax2, ax3]
    kpc_dim = SILengthDimension()
    kpc_dim.add_units("kpc", 3.08567758147e+19)

    def _annotate_rvir(ax, rvir, color="lightsteelblue", facecolor="none", alpha=1):
        xy = (0, 0)
        e = Circle(xy=xy, radius=rvir)

        ax.add_artist( e )
        e.set_clip_box( ax.bbox )
        e.set_edgecolor( color )
        e.set_facecolor( facecolor )  # "none" not None
        e.set_alpha( alpha )

    one_pixel_kpc = dx / float(camera.map_max_size)  # size of one pixel
    N_kpc_scale_bar = 20
    length_scale_bar = float(N_kpc_scale_bar) / one_pixel_kpc

    extent = [ -dx/2., dx/2, -dx/2., dx/2 ]

    # fig, axs = plt.subplots(3, 1, figsize=(5,12))
    fig, axs = plt.subplots(1, 3, figsize=(16,5))
    # fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.3)

    for i, proj in zip(range(3), projections):  # 3 plots
        ax = axs[i]
        sb = ScaleBar(dx / float(camera.map_max_size), "kpc", dimension=kpc_dim)
        sb.dimension._latexrepr['Zm'] = 'kpc'
        sb.dimension._latexrepr['Em'] = 'kpc'

        proj = projections[i]
        if i == 0:
            # Density
            proj_map = proj.map * unit_sd

            im = ax.imshow( np.log10(proj_map), cmap="RdBu_r", extent=extent, vmin=6. )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"log$_{10}$ $\rho$ [M$_{\odot}$ kpc$^{-2}$]")
        elif i == 1:
            proj_map = proj.map * subsnap.info["unit_temperature"].express(C.K)

            im = ax.imshow( np.log10(proj_map), cmap="jet_black", extent=extent, vmin=3. )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"log$_{10}$ min(T$_{2}$) [K/$\mu$]")
        elif i == 2:
            im = ax.imshow( np.log10(proj.map), vmin=-2, vmax=0, cmap="jet_black", extent=extent )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"log$_{10}$ min(x$_{\mathrm{HII}}$)")

        # h.annotate_rvir(proj, ax=ax)
        _annotate_rvir(ax, rvir)

        ax.set_xlabel(r"x [kpc]")
        # ax.set_ylabel(r"y [kpc]")

        # ax.add_artist(sb)
        # Scale bar
        # x_start = 300; y_start = 100
        # print length_scale_bar
        # ax.plot([x_start, x_start+length_scale_bar], [y_start, y_start], color="w", linestyle="-", linewidth=2.)
        # ax.set_axis_off()

    # plt.subplot(111)
    ax = axs[0]
    # ax = plt.gca()
    # im = ax.imshow( np.log10(proj_map.T), cmap="RdBu_r", extent=extent )
    # axins = zoomed_inset_axes(ax, 6, loc=1) # zoom = 6

    # sub region of the original image
    dx_zoom = snapshot.array(camera_zoom.region_size[0], subsnap.info["unit_length"]).in_units("kpc")
    extent_zoom = [ -dx_zoom/2., dx_zoom/2, -dx_zoom/2., dx_zoom/2 ]

    # Small offset
    extent_zoom[0] -= 0.1
    extent_zoom[1] -= 0.1
    extent_zoom[2] += 0.3
    extent_zoom[3] += 0.3

    axins = zoomed_inset_axes(ax, 4, loc=1) # zoom = 6
    axins.imshow(np.log10(proj_zoom_map.T), extent=extent_zoom, cmap="RdBu_r", vmin=6. )

    # x1, x2, y1, y2 = extent_zoom

    # Small offset
    x1, x2, y1, y2 = -1.97123744833-0.1, 1.97123744833-0.1, -1.97123744833+0.3, 1.97123744833+0.3

    print x1, x2, y1, y2
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    plt.setp(axins.get_xticklabels(), visible=False)
    plt.setp(axins.get_yticklabels(), visible=False)
    axins.xaxis.set_visible('False')
    axins.yaxis.set_visible('False')

    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    mark_inset(ax, axins, loc1=2, loc2=4, fc="w", ec="0.5", color="w")

    text_pos = (-20, 15)
    ax.text(text_pos[0], text_pos[1], "z = %1.2f" % snapshot.z, color="w", size="x-large")

    for ax in [axs[1], axs[2]]:
        plt.setp(ax.get_yticklabels(), visible=False)
        # plt.setp(ax.get_yticklabels(), visible=False)

    # axs[-1].set_xlabel(r"x [kpc]")
    axs[0].set_ylabel(r"y [kpc]")

    # plt.tight_layout()
    plt.draw()
    # plt.savefig("./self_shielding.pdf", format="pdf")
    plt.show()

    # plt.xticks(visible=False)
    # plt.yticks(visible=False)

    return projections, proj_zoom, extent, camera, camera_zoom




def baryfrac_xHII_panel_plot(sim, z, proj_list=None):
    import matplotlib as mpl
    from matplotlib import ticker
    import matplotlib.gridspec as gridspec
    import matplotlib.pylab as plt
    import numpy as np
    from seren3.analysis.visualization import engines, operators
    from seren3.utils import plot_utils

    op = operators.MassWeightedOperator("xHII", sim[sim.redshift(z[0])].C.none)

    if proj_list is None:
        proj_list = []
        for zi in z:
            snap = sim[sim.redshift(zi)]
            eng = engines.CustomSplatterEngine(snap.g, 'xHII', op, extra_fields=['rho'])
            cam = snap.camera()
            cam.map_max_size = 512
            proj = eng.process(cam)
            proj_list.append(proj)

    cm = plot_utils.load_custom_cmaps('blues_black_test')
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    for proj,ax in zip(proj_list, axs.flatten()):
        im = ax.imshow( np.log10(proj.map.T), vmin=-2, vmax=0, cmap=cm )
        ax.set_axis_off()

    text_pos = (14, 57)

    cols = ['white', 'white', 'white', 'k']
    for i,zi,ci in zip(range(4), z, cols):
        ax = axs.flatten()[i]
        ax.text(text_pos[0], text_pos[1], "z = %1.2f" % zi, color=ci)
    # axs.flatten()[-1].text(text_pos[0], text_pos[1], "z = %1.2f" % z[-1], color='k')

    plt.subplots_adjust(wspace=0.0, hspace=0.01)

    cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    cbar = plt.colorbar(im, cax=cax, **kw)
    cbar.set_label(r"log$_{10}$ x$_{\mathrm{HII}}$")

    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    # fig.savefig('/home/ds381/bc03_xHII_mw_proj.pdf', format='pdf')
    plt.show()

    return proj_list