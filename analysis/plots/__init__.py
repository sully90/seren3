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

def baryfrac_xHII_panel_plot(sim, z):
    import matplotlib as mpl
    from matplotlib import ticker
    import matplotlib.gridspec as gridspec
    import matplotlib.pylab as plt
    from seren3.analysis.visualization import engines, operators
    from seren3.utils import plot_utils

    op = operators.MassWeightedOperator("xHII", sim[sim.redshift(z[0])].C.none)

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

    for i,zi in zip(range(3), z):
        ax = axs.flatten()[i]
        ax.text(text_pos[0], text_pos[1], "z = %1.2f" % zi, color='white')
    axs.flatten()[-1].text(text_pos[0], text_pos[1], "z = %1.2f" % z[-1], color='k')

    plt.subplots_adjust(wspace=0.0, hspace=0.01)

    cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    cbar = plt.colorbar(im, cax=cax, **kw)
    cbar.set_label(r"log$_{10}$ x$_{\mathrm{HII}}$")

    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    # fig.savefig('/home/ds381/bc03_xHII_mw_proj.pdf', format='pdf')
    plt.show()