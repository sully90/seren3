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