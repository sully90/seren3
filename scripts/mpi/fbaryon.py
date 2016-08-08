import numpy as np

def unpack(iout, path='./'):
    import pickle
    fname = "%s/pickle/fbaryon_%05i.p" % (path, iout)
    data = pickle.load( open(fname, "rb") )

    def _unpack(data):
        tot_mass, fb = (np.zeros(len(data)), np.zeros(len(data)))
        for i in range(len(data)):
            tot_mass[i] = data[i].result["tot_mass"]
            fb[i] = data[i].result["fb"]
        return tot_mass, fb

    return _unpack(data)    

#  fbaryon.plot(paths, iouts, labels, cols=cols, plot_scatter=False, label_z=True, cmap='jet_black', nbins=10); plt.title(r"$\langle x_{\mathrm{HII}} \rangle _{\mathrm{v}}$ = 0.75"); plt.show()
def plot(paths, ioutputs, labels, cols=None, nbins=10, plot_scatter=True, label_z=True, alpha=0.25, cmap='jet'):
    '''
    Loads pickled data and plots log_{10}(Mhalo) vs fb in units of the mean
    '''
    import seren3, pickle, matplotlib.pylab as plt, numpy as np
    #from seren3.analysis.plots import fit_scatter
    from seren3.analysis import plots
    reload(plots)
    from seren3.utils.plot_utils import ncols

    def _unpack(data):
        tot_mass, fb = (np.zeros(len(data)), np.zeros(len(data)))
        for i in range(len(data)):
            tot_mass[i] = data[i].result["tot_mass"]
            fb[i] = data[i].result["fb"]
        return tot_mass, fb

    if cols == None:
        cols = ncols(len(paths), cmap=cmap)
    for path, iout, label, c in zip(paths, ioutputs, labels, cols):
        sim = seren3.init(path)
        snap = sim[iout]
        cosmo = snap.cosmo
        fname = "%s/pickle/fbaryon_%05i.p" % (path, iout)
        data = pickle.load( open(fname, "rb") )
        tot_mass, fb = _unpack(data)
        # Put tot_mass in units of Msol/h
        h = cosmo["h"]
        tot_mass *= h
        log_tot_mass = np.log10(tot_mass)

        # put fb in units of the cosmic mean baryon fraction
        cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
        fb_cosmic_mean = fb/cosmic_mean

        if label_z:
            label = "%s z=%1.2f" % (label, snap.z)

        # plot the scatter and fit
        if plot_scatter:
            plt.scatter(log_tot_mass, fb_cosmic_mean, color=c, alpha=alpha)
        bin_centres, mean, std, n = plots.fit_scatter(log_tot_mass, fb_cosmic_mean, ret_n=True, nbins=nbins)

        sterr = std/np.sqrt(n)
        plt.errorbar(bin_centres, mean, yerr=sterr, label=label, color=c, linewidth=1.5)

    plt.xlabel(r"log$_{10}$(M$_{\mathrm{h}}$[M$_{\odot}$/h])")
    plt.ylabel(r"f$_{\mathrm{gas}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
    plt.legend(loc='lower right')


def gnedin_fitting_func(Mh, Mc, alpha, **cosmo):
    f_bar = cosmo["omega_b_0"] /  cosmo["omega_M_0"]
    return f_bar * (1 + (2**(alpha/3.) - 1) * (Mh/Mc)**(-alpha))**(-3./alpha)


def fit(snapshot):
    '''
    Fit Mc and alpha
    '''
    import numpy as np
    import scipy.optimize as optimization

    def _gnedin_fitting_func(snapshot):
        def wrapped(Mh, Mc, alpha):
            cosmo = snapshot.cosmo
            f_bar = cosmo["omega_b_0"] /  cosmo["omega_M_0"]
            return f_bar * (1 + (2**(alpha/3.) - 1) * (Mh/Mc)**(-alpha))**(-3./alpha)
        return wrapped

    tot_mass, fb = unpack(snapshot.ioutput, path=snapshot.path)
    log_tot_mass = np.log10(tot_mass)

    # Initial guess at fit
    cosmo = snapshot.cosmo
    cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    fb_cosmic_mean = fb/cosmic_mean

    idx_Mc = np.abs(fb_cosmic_mean - 0.5).argmin()
    Mc = tot_mass[idx_Mc]
    alpha = 1.5
    p0 = [Mc, alpha]  # the initial guess at Mc/alpha

    # Curve fit
    popt, pcov = optimization.curve_fit(_gnedin_fitting_func(snapshot), tot_mass, fb, p0=p0, maxfev=1000)

    # Errors
    sigma_Mc = np.sqrt(pcov[0,0])
    sigma_alpha = np.sqrt(pcov[1,1])

    # Correlation
    corr = pcov[0,1] / (sigma_Mc * sigma_alpha)

    Mc_fit = popt[0]
    alpha_fit = popt[1]

    return (Mc_fit, sigma_Mc), (alpha_fit, sigma_alpha), corr
    # return popt, pcov


def main(path, iout):
    import seren3
    import pickle, os
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading data")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)  # disbale multiprocessing

    halos = snap.halos()
    mpi_halos = halos.mpi_spheres()

    dest = {}
    for h, sto in mpi.piter(mpi_halos, storage=dest):
        sphere = h["reg"]
        subsnap = snap[sphere]

        # part_dset = subsnap.p[["mass", "epoch", "id"]].f
        part_mass = subsnap.p["mass"].f
        gas_mass = subsnap.g["mass"].f

        part_mass_tot = part_mass.in_units("Msol").sum()
        gas_mass_tot = gas_mass.in_units("Msol").sum()
        tot_mass = part_mass_tot + gas_mass_tot

        fb = gas_mass_tot/tot_mass
        sto.idx = h["id"]
        sto.result = {"fb" : fb, "tot_mass" : tot_mass}

    if mpi.host:
        pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        pickle.dump( mpi.unpack(dest), open( "%s/fbaryon_%05i.p" % (pickle_path, iout), "wb" ) )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    try:
        main(path, iout)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        mpi.terminate(500, e=e)