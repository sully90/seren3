def plot_baryfrac_comparisson(snap1, snap2, ihalo1, ihalo2, PDF_CDF=None):
    import matplotlib.pylab as plt

    halos1 = snap1.halos(); shalos1 = halos1.sort("Mvir")
    halos2 = snap2.halos(); tree2 = halos2.kdtree()

    hs1_i1 = shalos1[ihalo1]; hs1_i2 = shalos1[ihalo2]

    d1, idx1 = tree2.query(hs1_i1.pos, 1)
    d2, idx2 = tree2.query(hs1_i2.pos, 1)
    hs2_i1 = halos2[idx1]
    hs2_i2 = halos2[idx2]

    if PDF_CDF is None:
        PDF_CDF = [ pdf_cdf(h.g, 'nH', cumulative=True) for h in [hs1_i1, hs1_i2, hs2_i1, hs2_i2] ]

    titles = [ "M$_{\mathrm{vir}}$ = %1.2e M$_{\odot}$/h \nT$_{\mathrm{vir}}$ = %1.2e K" % (h['mvir'], h.Tvir) for h in [hs1_i1, hs1_i2, hs2_i1, hs2_i2] ]

    fig, axs = plt.subplots(nrows=2, ncols=2)

    for snap,val,ax in zip([snap1, snap1, snap2, snap2], PDF_CDF, axs.flatten()):
        P,C,bc,dx = val
        plot_pdf_cdf(snap, P, bc, dx, True, r"n$_{\mathrm{H}}$ [m$^{-3}$]", ax1=ax, C=C, label_nstar=True)

    for ax, title in zip(axs.flatten(), titles):
        ax.set_title(title, fontsize=18)

    plt.tight_layout(pad=0.005)
    plt.show(block=False)

    return PDF_CDF

def rho_T_hist2d(snap, xo=None, yo=None, mass=None, den_field='nH', temp_field='T2', \
                nbins=500, plot=False, ax=None, title=None, show=False, **kwargs):
    ''' Produces a mass weighted, 2D histogram of density vs temperature '''
    from seren3.array import SimArray
    from seren3.utils.plot_utils import add_colorbar
    import numpy as np
    import matplotlib.pylab as plt

    if ax is None:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

    if xo is None or yo is None or mass is None:
        dset = snap.g[den_field, temp_field, 'mass'].flatten()
        mass = dset["mass"].in_units("Msol")

    totmass = mass.sum()
    xo, yo = (dset[den_field], dset[temp_field])

    h, xs, ys = hist2d(xo, yo, xlogrange=True, ylogrange=True, density=False, mass=mass, nbins=500)
    h /= totmass

    xs = SimArray(xs, dset[den_field].units)
    #xs.set_latex(dset[den_field].get_field_latex())

    ys = SimArray(ys, dset[temp_field].units)
    #ys.set_latex(dset[temp_field].get_field_latex())

    if plot:
        cmap = kwargs.get('cmap', 'coolwarm')
        p = plot_2dhist(h, xs, ys, den_field, temp_field, ax=ax, ret_im=True, cmap=cmap)
        p.set_clim(vmin=-8, vmax=-2)

    if title:
        plt.title(title)

    if show:
        plt.show()

    return h, xs, ys


def plot_2dhist(h, xs, ys, den_field, temp_field, \
            ax=None, cmap='RdBu_r', title=None, ret_im=False, show=False):
    import seren3
    from seren3.utils.plot_utils import add_colorbar
    import numpy as np
    import matplotlib.pylab as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
    from matplotlib.ticker import MultipleLocator

    temp_latex = ys.latex
    den_latex = xs.latex

    p = None
    if ax is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.ticker import MultipleLocator
        p = ax.imshow(np.log10(h), cmap=cmap, extent=[xs.min(), xs.max(), ys.min(), ys.max()])

        if title is not None:
            ax.set_title(title)
    else:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        p = ax.imshow(np.log10(h), cmap=cmap, \
                extent=[xs.min(), xs.max(), ys.min(), ys.max()], aspect='auto')

        if title is not None:
            plt.title(title)
    if ax is not None:
        ax.set_xlabel(r'log$_{10}$(%s)' % den_latex)
        ax.set_ylabel(r'log$_{10}$(%s)' % temp_latex)
    else:
        plt.xlabel(r'log$_{10}$(%s)' % den_latex)
        plt.ylabel(r'log$_{10}$(%s)' % temp_latex)

    ax = p.get_axes()
    cbar = add_colorbar(p)
    cbar.set_label('f(mass)')

    if show:
        plt.show(block=False)

    if ret_im:
        return p


def hist2d(xo, yo, weights=None, mass=None, gridsize=(100, 100), nbins = None, make_plot = True, density=True, **kwargs):
    '''
    Plots a 2D histogram of fields[0] against fields[1]
    '''
    import numpy as np
    from seren3.array import SimArray
    # process keywords
    x_range = kwargs.get('x_range', None)
    y_range = kwargs.get('y_range', None)
    xlogrange = kwargs.get('xlogrange', False)
    ylogrange = kwargs.get('ylogrange', False)
    ret_im = kwargs.get('ret_im', False)

    if y_range is not None:
        if len(y_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if ylogrange:
            y_range = [np.log10(np.min(yo)), np.log10(np.max(yo))]
        else:
            y_range = [np.min(yo), np.max(yo)]
        kwargs['y_range'] = y_range

    if x_range is not None:
        if len(x_range) != 2:
            raise RuntimeError("Range must be a length 2 list or array")
    else:
        if xlogrange:
            x_range = [np.log10(np.min(xo)), np.log10(np.max(xo))]
        else:
            x_range = [np.min(xo), np.max(xo)]
        kwargs['x_range'] = x_range

    if (xlogrange):
        x = np.log10(xo)
    else:
        x = xo

    if (ylogrange):
        y = np.log10(yo)
    else:
        y = yo

    if nbins is not None:
        gridsize = (nbins, nbins)

    if nbins is not None:
        gridsize = (nbins, nbins)

    ind = np.where((x > x_range[0]) & (x < x_range[1]) &
                   (y > y_range[0]) & (y < y_range[1]))

    x = x[ind[0]]
    y = y[ind[0]]

    draw_contours = False
    if weights is not None and mass is not None:
        draw_contours = True
        weights = weights[ind[0]]
        mass = mass[ind[0]]

        # produce a mass-weighted histogram of average weight values at each
        # bin
        hist, ys, xs = np.histogram2d(
            y, x, weights=weights * mass, bins=gridsize, range=[y_range, x_range])
        hist_mass, ys, xs = np.histogram2d(
            y, x, weights=mass, bins=gridsize, range=[y_range, x_range])
        good = np.where(hist_mass > 0)
        hist[good] = hist[good] / hist_mass[good]

    else:
        if weights is not None:
            # produce a weighted histogram
            weights = weights[ind[0]]
        elif mass is not None:
            # produce a mass histogram
            weights = mass[ind[0]]

        hist, ys, xs = np.histogram2d(
            y, x, weights=weights, bins=gridsize, range=[y_range, x_range])

        xs = .5 * (xs[:-1] + xs[1:])
        ys = .5 * (ys[:-1] + ys[1:])

    # if ret_im:
    #     return make_contour_plot(hist, xs, ys, **kwargs)

    # if make_plot:
    #     make_contour_plot(hist, xs, ys, **kwargs)
    #     if draw_contours:
    #         make_contour_plot(SimArray(density_mass, mass.units), xs, ys, filled=False,
    #                           clear=False, colorbar=False, colors='black', scalemin=nmin, nlevels=10)

    if isinstance(xo, SimArray):
        xs = SimArray(xs, xo.units)
        xs.set_field_latex(xo.get_field_latex())
    if isinstance(yo, SimArray):
        ys = SimArray(ys, yo.units)
        ys.set_field_latex(yo.get_field_latex())
    hist = np.flipud(hist)
    if density:
        return hist / len(xo), xs, ys
    return hist, xs, ys

def binned_by_density(density, field, xlogrange=True, ylogrange=True, thresh=None, den_field='nH', nbins=100):
    '''
    Bin a quantity by density
    thresh: lambda function to return indicies of cells to keep using np.where
    '''
    import numpy as np
    from seren3.analysis.plots import fit_scatter

    if thresh:
        idx = thresh(density, field)
        field = field[idx]
        density = density[idx]
    if xlogrange:
        density = np.log10(density)
    if ylogrange:
        field = np.log10(field)

    bin_centres, mean, std = fit_scatter(density, field, nbins=nbins)
    return bin_centres, mean, std

def nH_halo_cdf_fit(snapshot, the_mass_bins=[8., 9., 10.], nbins=10):
    import numpy as np
    from seren3.analysis.plots import fit_scatter
    from seren3.utils import flatten_nested_array

    # First, compute min/max nH in the snapshot
    print "Bracketing nH"
    min_nH = np.inf; max_nH = -np.inf
    for dset in snapshot.g["nH"]:
        nH = dset["nH"]
        if nH.min() < min_nH:
            min_nH = nH.min()
        if nH.max() > max_nH:
            max_nH = nH.max()

    nH_range = (min_nH, max_nH)

    halos = snapshot.halos(finder = 'ctrees')
    nhalos = len(halos)

    cdf_halos = []; bin_centre_halos = []; Mvir = []

    for i in range(nhalos):
        print '(%i / %i)' % (i+1, nhalos)
        h = halos[i]
        if len(h.g) > 0:
            P, C, bc, dx = pdf_cdf(h.g, "nH", cumulative=True, plot=False, bins=50, x_range=nH_range, density=True)
            cdf_halos.append(C)
            bin_centre_halos.append(bc)
            Mvir.append(h["Mvir"])

    cdf_halos = np.array(cdf_halos); bin_centre_halos = np.array(bin_centre_halos); Mvir = np.array(Mvir)

    # Bin idx
    mass_bins = np.digitize(np.log10(Mvir), the_mass_bins, right=True)
    binned_cdf = {}

    for i in range(len(the_mass_bins)):
        idx = np.where( mass_bins == i )
        x, y = ( flatten_nested_array(bin_centre_halos[idx]), flatten_nested_array(cdf_halos[idx]) )
        binned_cdf[i] = fit_scatter(x, y, ret_sterr=True)

    binned_cdf['the_mass_bins'] = the_mass_bins

    return binned_cdf

def pdf_cdf(snapshot, field, field_latex=None, bins=50, logscale=True, density=False, cumulative=False, \
         plot=False, show=False, color_ax2_y=True, P_y_range=None, x_range=None, **kwargs):
    '''
    Computes the PDF and (optional) CDF of a given field
    '''
    import seren3
    import numpy as np

    if not hasattr(snapshot, 'family'):
        raise Exception("Must supply family specific snapshot")

    # dset = snapshot[field].flatten()
    # data = dset[field]
    dset = snapshot[field].flatten()
    data = dset[field]

    if logscale:
        data = np.log10(data)
        if x_range is not None:
            x_range = np.log10(x_range)

    P, bin_edges = np.histogram(data, bins=bins, range=x_range, density=density)
    P = np.array( [float(i)/float(len(data)) for i in P] )
    bincenters = 0.5*(bin_edges[1:] + bin_edges[:-1])
    dx = (bincenters.max() - bincenters.min()) / bins

    C = None
    if cumulative:
        C = np.cumsum(P) * dx
        C = (C - C.min()) / C.ptp()

    if plot:
        if cumulative:
            plot_pdf_cdf(snapshot, P, bincenters, dx, logscale, field_latex, C=C, P_y_range=P_y_range, show=show, color_ax2_y=color_ax2_y, **kwargs)
        else:
            plot_pdf_cdf(snapshot, P, bincenters, dx, logscale, field_latex, P_y_range=P_y_range, show=show, color_ax2_y=color_ax2_y, **kwargs)

    if cumulative:
        return P, C, bincenters, dx
    return P, bincenters, dx

def plot_pdf_cdf(snapshot, P, bincenters, dx, logscale, field_latex, \
            ax1 = None, C=None, show=False, P_y_range=None, barcol='dimgrey', \
            cumul_col="b", label_nstar=False, label_T_star=False, color_ax2_y=True, label=True):

    import seren3
    import numpy as np
    import matplotlib.pylab as plt

    if ax1 is None:
        fig, ax1 = plt.subplots()
    ax2 = None
    if C is not None:
        ax2 = ax1.twinx()

    ax1.bar(bincenters, P, width=dx, color=barcol, edgecolor="w")

    ymin = ymax = None
    if P_y_range:
        ymin, ymax = P_y_range
    else:
        ymin, ymax = ax1.get_ylim()
    xmin, xmax = ax1.get_xlim()
    xr = (xmax - xmin)

    if C is not None:
        ax2.plot(bincenters, C, linewidth=3., color=cumul_col, linestyle="-")
        ax2.set_ylim(0., 1.)

    if label:
        ax1.set_ylabel("P")

    if C is not None and label:
        ax2.set_ylabel("Cumulative")
        #ax2.set_ylim([0.0, 1.0])

    if P_y_range:
        ax1.set_ylim(P_y_range)
    else:
        ax1.set_ylim([ymin, ymax])

    if C is not None and color_ax2_y:
        ax2.yaxis.label.set_color(cumul_col)
        ax2.spines["right"].set_color(cumul_col)
        ax2.tick_params(axis="y", colors=cumul_col)

    if label_nstar:
        fs = 15
        n_star = snapshot.array(snapshot.info_rt["n_star"].express(snapshot.C.H_cc), "cm**-3").in_units("m**-3")
        if logscale:
            ax1.vlines(np.log10(n_star), ymin, ymax, color='r', linestyle='--')
            # ax1.text(np.log10(snapshot.info_rt['n_star'].express(snapshot.C.H_cc)) + xr*0.01, ymax/2. + .01, r"n$_{*}$", color='r', fontsize=fs)
            # ax1.text(np.log10(n_star) + xr*0.01, 0.076, r"n$_{*}$", color='r', fontsize=fs)
        else:
            ax1.vlines(snapshot.info_rt['n_star'].express(snapshot.C.H_cc), ymin, ymax, color='r', linestyle='--')
            # ax1.text(n_star + xr*0.01, ymax/2. + .01, r"n$_{*}$", color='r', fontsize=fs)

    if label_T_star:
        print ymax/2. + .01
        T_th = 2.e4 - snapshot.info_rt['T2_star'].val
        fs = 15
        if logscale:
            ax1.vlines(np.log10(T_th), ymin, ymax, color='r', linestyle='--')
            ax1.text(np.log10(T_th) + xr*0.01, ymax/1.5 + .01, r"T$_{\mathrm{sf}}$", color='r', fontsize=fs)
        else:
            ax1.vlines(T_th, ymin, ymax, color='r', linestyle='--')
            ax1.text(T_th + xr*0.01, ymax/1.5 + .01, r"T$_{\mathrm{sf}}$", color='r', fontsize=fs)


    if logscale and label:
        ax1.set_xlabel(r"log$_{10}$(%s)" % field_latex)
    elif label:
        ax1.set_xlabel(field_latex)

    if show:
        plt.show()

    return ax1, ax2
