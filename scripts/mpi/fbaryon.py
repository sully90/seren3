import numpy as np

_DEFAULT_ALPHA = 2.
_MASS_UNIT = "Msol h**-1"


def _gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def _bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return _gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2)


def tidal_force_pdf(snapshot, fname, plot=True, **kwargs):
    '''
    Computes and (optional) plots the PDF of the tidal force for each halo, averaged
    over a dynamical time
    '''
    import pickle
    import peakutils
    from peakutils.peak import centroid
    from seren3.analysis.plots import fit_scatter
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit

    nbins = kwargs.pop("nbins", 50)
    def pdf(arr, bins):
        log_arr = np.log10(arr)
        idx = np.where(np.isinf(log_arr))
        log_arr = np.delete(log_arr, idx)

        P, bin_edges = np.histogram(log_arr, bins=bins,  density=False)
        P = np.array( [float(i)/float(len(data)) for i in P] )
        bincenters = 0.5*(bin_edges[1:] + bin_edges[:-1])
        dx = (bincenters.max() - bincenters.min()) / bins
        C = np.cumsum(P) * dx
        C = (C - C.min()) / C.ptp()
        return P,C,bincenters,dx

    data = pickle.load(open(fname, 'rb'))
    tidal_force_tdyn = np.zeros(len(data))

    for i in range(len(data)):
        res = data[i].result
        tidal_force_tdyn[i] = res["hprops"]["tidal_force_tdyn"]

    P,C,bincenters,dx = pdf(tidal_force_tdyn, nbins)
    n = int(np.round(nbins/5.))
    bc, mean, std = fit_scatter(bincenters, P, nbins=n)

    # Interpolate the PDF
    fn = interp1d(bincenters, P)
    x = np.linspace(bincenters.min(), bincenters.max(), 1000)
    y = fn(x)
    
    # Fit peaks to get initial estimate of guassian properties
    indexes = peakutils.indexes(y, thres=0.02, min_dist=250)
    peaks_x = peakutils.interpolate(x, y, ind=indexes)

    # Do the bimodal fit
    expected = (peaks_x[0], 0.2, 1.0, peaks_x[1], 0.2, 1.0)
    params,cov=curve_fit(_bimodal,x,y,expected)
    sigma=np.sqrt(np.diag(cov))

    # Refit the peaks to improve accuracy
    y_2 = _bimodal(x,*params)
    fn = interp1d(x, y_2)
    indexes = peakutils.indexes(y_2, thres=0.02, min_dist=250)
    peaks_x = peakutils.interpolate(x, y_2, ind=indexes)

    if plot:
        import matplotlib.pylab as plt

        p = plt.plot(bincenters, P * (1. + snapshot.z)**3, linewidth=1.5, label="z=%1.2f" % snapshot.z)
        col = p[0].get_color()
        plt.plot(x, y_2 * (1. + snapshot.z)**3, color=col, lw=3)#, label='model')

        plt.scatter(peaks_x, fn(peaks_x) * (1. + snapshot.z)**3, color='r', s=250, marker='o')

        plt.xlabel(r"log$_{10}$ $\langle F_{\mathrm{Tidal}} \rangle_{t_{\mathrm{dyn}}}$")
        plt.ylabel(r"P (1 + z)$^{3}$")
        plt.legend()
        # plt.show()

    return P,C,bincenters,dx,(bc,mean,std,indexes,peaks_x)


def Okamoto_Mc_fn():
    from seren3 import config
    from scipy import interpolate
    # from seren3.analysis.interpolate import extrap1d

    fname = "%s/Mc_Okamoto08.txt" % config.get('data', 'data_dir')
    data = np.loadtxt(fname)
    ok_a, ok_z, ok_Mc = data.T

    fn = interpolate.interp1d(ok_z, np.log10(ok_Mc), fill_value="extrapolate")
    return lambda z: 10**fn(z)


def interp_Okamoto_Mc(z):
    fn = Okamoto_Mc_fn()
    return fn(z)
    # fn = interpolate.InterpolatedUnivariateSpline(ok_z, np.log10(ok_Mc), k=1)  # interpolate on log mass
    # return 10**fn(z)


def plot_fits(Mc_amr, Mc_cudaton, **cosmo):
    import numpy as np
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec

    Mc_okamoto = interp_Okamoto_Mc(cosmo["z"])
    print "Okamoto 08 Mc(z=%1.1f) = %e" % (cosmo["z"], Mc_okamoto)

    fig = plt.figure()
    gs = gridspec.GridSpec(3,3,wspace=0.,hspace=0.)

    ax1 = fig.add_subplot(gs[1:,:])
    ax2 = fig.add_subplot(gs[:1,:], sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)

    cosmic_mean = cosmo['omega_b_0']/cosmo['omega_M_0']

    mass = np.linspace(1e6, 1e10, 1000)
    fb_amr = fbaryon.gnedin_fitting_func(mass, Mc_amr, 2., **cosmo)/cosmic_mean
    fb_cudaton = fbaryon.gnedin_fitting_func(mass, Mc_cudaton, 2., **cosmo)/cosmic_mean
    fb_okamoto = fbaryon.gnedin_fitting_func(mass, Mc_okamoto, 2., **cosmo)/cosmic_mean

    diff = fb_amr - fb_cudaton

    ax1.plot(mass, fb_amr, linewidth=2., color='r', label='AMR')
    ax1.plot(mass, fb_cudaton, linewidth=2., color='b', label='CUDATON')
    ax1.plot(mass, fb_okamoto, linewidth=2., color='k', label='Okamoto 08', linestyle='-.')
    ax1.hlines(0.5, mass.min(), mass.max(), linestyle='--', linewidth=2.,\
             label=r"$\frac{1}{2}$ $\frac{\Omega_{\mathrm{b}}}{\Omega_{\mathrm{M}}}$")
    ax1.legend(loc="upper left")

    ax2.plot(mass, diff, linewidth=2., color='k')

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    ax1.set_title("z = %1.1f" % cosmo["z"])
    ax1.set_xlabel(r"log$_{10}$(M$_{\mathrm{h}}$[M$_{\odot}$/h])")
    ax1.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
    ax2.set_ylabel("Difference")

    plt.show()

    return fb_amr, fb_cudaton


def load_data(fname):
    import pickle

    f = open(fname, "rb")
    data = pickle.load(f)

    # Load the required arrays. We use dm particle and cell count to filter poorly resolved halos
    # pid==-1 gives distinct halos only
    nrecords = len(data)
    mass = np.zeros(nrecords); fb = np.zeros(nrecords)
    np_dm = np.zeros(nrecords); ncell = np.zeros(nrecords)

    # Use the tidal force, in dimensionless units of Rvir/Rhill (Hill = Hill sphere, or sphere of influence over satellites)
    # averaged over a single dynamical time.

    tidal_force_tdyn = np.zeros(nrecords)
    pid = np.zeros(nrecords)

    for i in range(nrecords):
        res = data[i].result
        mass[i] = res["tot_mass"]; fb[i] = res["fb"]
        np_dm[i] = res["np_dm"]; ncell[i] = res["ncell"]

        pid[i] = res["pid"]
        tidal_force_tdyn[i] = res["hprops"]["tidal_force_tdyn"]
        del res

    f.close()
    return mass, fb, np_dm, ncell, tidal_force_tdyn, pid


def filter_and_load_data(fname, tidal_force_cutoff=np.inf, dm_particle_cutoff=100., ncell_cutoff=1.,):
    _MASS_CUTOFF = 1e5  # Msun/h
    print tidal_force_cutoff, dm_particle_cutoff, ncell_cutoff

    mass, fb, np_dm, ncell, tidal_force_tdyn, pid = load_data(fname)

    # Apply resolution cutoff
    idx = np.where( np.logical_and(np_dm >= dm_particle_cutoff, ncell >= ncell_cutoff) )
    mass = mass[idx]; fb = fb[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

    # Keep distinct halos only
    idx = np.where(pid[idx]==-1)
    mass = mass[idx]; fb = fb[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

    # Apply mass cutoff
    idx = np.where(mass >= _MASS_CUTOFF)
    mass = mass[idx]; fb = fb[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

    # Apply tidal force cutoff
    idx = np.where(tidal_force_tdyn <= tidal_force_cutoff)
    mass = mass[idx]; fb = fb[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

    return mass, fb, np_dm, ncell, tidal_force_tdyn, pid


def plot(snapshot, fname, **kwargs):
    '''
    Plot and fit the halo baryon fraction
    '''
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec
    from seren3.analysis.plots import fit_scatter, grid

    fig = plt.figure(figsize=(9, 8))

    # Set some global variables
    _SIZE = 12

    fix_alpha = kwargs.get("fix_alpha", True)
    use_lmfit = kwargs.get("use_lmfit", True)

    # Cosmological params
    cosmo = snapshot.cosmo
    cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]

    # Open the file and load the pickle dictionary  # Filter
    mass, fb, np_dm, ncell, tidal_force_tdyn, pid = filter_and_load_data(fname, **kwargs)
    print len(mass)

    # Compute the characteristic mass scale, Mc
    fit_dict = fit(mass, fb, fix_alpha, use_lmfit=use_lmfit, **cosmo)
    Mc_fit, Mc_stderr = ( fit_dict["Mc"]["fit"], fit_dict["Mc"]["stderr"] )
    alpha_fit, alpha_stderr = ( fit_dict["alpha"]["fit"], fit_dict["alpha"]["stderr"] )

    # Arrays for generating the fit
    x = np.linspace(mass.min(), mass.max(), 10000)
    y = gnedin_fitting_func(x, Mc_fit, alpha_fit, **cosmo)

    print 'Mc(z=%1.1f) = %e +/- %e, alpha = %f' % (snapshot.z, Mc_fit, Mc_stderr, alpha_fit)

    # Plot in units of the cosmic mean
    fb_cosmic_mean = fb/cosmic_mean_b
    y_cosmic_mean = y/cosmic_mean_b

    ax1 = plt.gca()

    # carr = np.log10(1.+tidal_force_tdyn)
    carr = tidal_force_tdyn
    p = ax1.scatter(mass, fb_cosmic_mean, s=_SIZE, alpha=1., c=carr, cmap="jet_black")
    cbar = plt.colorbar(p, ax=ax1)
    # cbar.set_label(r"Time Since Last MM [Gyr]")
    cbar.set_label(r"Tidal Force (Av. over dyn time)")
    # ax1.plot(x, y_cosmic_mean, color='k', linewidth=2.)

    Mc_okamoto = interp_Okamoto_Mc(cosmo["z"])
    ax1.plot(x, gnedin_fitting_func(x, Mc_okamoto, 2., **cosmo)/cosmic_mean_b,\
                 color='k', linewidth=2., label=r"Okamoto08 Mc(z=%1.1f) = %1.2e M$_{\odot}$/h" % (cosmo['z'], Mc_okamoto))

    # ax1.plot(x, gnedin_fitting_func(x, Mc_fit, 2., **cosmo)/cosmic_mean_b,\
    #              color='r', linewidth=2., label=r"Fit Mc(z=%1.1f) = %1.2e +/- %1.2e M$_{\odot}$/h" % (cosmo['z'], Mc_fit, Mc_stderr))
    y_upper = gnedin_fitting_func(x, Mc_fit + Mc_stderr, alpha_fit, **cosmo)/cosmic_mean_b
    y_lower = gnedin_fitting_func(x, Mc_fit - Mc_stderr, alpha_fit, **cosmo)/cosmic_mean_b
    ax1.fill_between(x, y1=y_lower, y2=y_upper, color='r', label=r"Fit Mc(z=%1.1f) = %1.2e +/- %1.2e M$_{\odot}$/h" % (cosmo['z'], Mc_fit, Mc_stderr))

    ax1.set_xscale("log")
    ax1.set_title("z = %1.2f" % snapshot.z)
    ax1.set_xlabel(r"log$_{10}$(M$_{\mathrm{h}}$[M$_{\odot}$/h])")
    ax1.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
    ax1.hlines(0.5, mass.min(), mass.max(), linestyle='--', linewidth=2.,\
             label=r"$\frac{1}{2}$ $\frac{\Omega_{\mathrm{b}}}{\Omega_{\mathrm{M}}}$")
    plt.legend(loc='upper left')
    # plt.ylim(0., 1.5)

    # plt.show()


def fit_sim_iouts(iouts, pickle_dir, fix_alpha=True, use_lmfit=True, *args, **cosmo):
    # Compute Mc/alpha fit for each output

    output = {}
    for iout in iouts:
        fname = "%s/ConsistentTrees/fbaryon_tdyn_%05i.p" % (pickle_dir, iout)
        mass, fb, np_dm, ncell, tidal_force_tdyn, pid = filter_and_load_data(fname, *args)

        # Compute the characteristic mass scale, Mc
        fit_dict = fit(mass, fb, fix_alpha, use_lmfit=use_lmfit, **cosmo)
        # Mc_fit, Mc_stderr = ( fit_dict["Mc"]["fit"], fit_dict["Mc"]["stderr"] )
        # alpha_fit, alpha_stderr = ( fit_dict["alpha"]["fit"], fit_dict["alpha"]["stderr"] )

        output[iout] = fit_dict

    return output


def tmp_plot_bc03_bpass(fix_alpha=False):
    import matplotlib.pylab as plt
    from seren3.core.simulation import Simulation

    paths = ['/research/prace/david/bpass/bc03/', \
            '/lustre/scratch/astro/ds381/simulations/bpass/bc03_fesc5']#, \
            # '/lustre/scratch/astro/ds381/simulations/bpass/cosma/bin_sed/']#, \
            # '/research/prace/david/aton/256/']

    ppaths = ['/lustre/scratch/astro/ds381/simulations/bpass/bc03/pickle/', \
            '/lustre/scratch/astro/ds381/simulations/bpass/bc03_fesc5/pickle/']#, \
            # '/lustre/scratch/astro/ds381/simulations/bpass/cosma/bin_sed/pickle/']#, \
            # '/lustre/scratch/astro/ds381/simulations/aton/256/pickle/']

    sims = [Simulation(p) for p in paths]
    z=[12, 9.520844871763828,9.1183961742776667,8.870301819250594,8.637584547306572,8.478813937808214,7.970466391120848,\
        7.649564270488902,7.243141404330228,7.020725505646379,6.6678946695209875,\
        6.435186929659703,6.152974042392178]

    sim_iouts = []
    for sim in sims:
        iouts = [sim.redshift(i) for i in z]
        print iouts
        sim_iouts.append(iouts)
    # iouts = [60, 66, 70, 76, 80, 86, 90, 96, 100, 106]
    labels = ["BC03", "BC03_FESC5"]#, "ATON"]
    cols = ['r', 'b']#, 'm']

    plot_Mc_var(sims, sim_iouts, ppaths, labels, cols, fix_alpha)
    plt.show()


def plot_Mc_var(sims, sim_iouts, pickle_paths, labels, cols, fix_alpha=True):
    import pickle
    import matplotlib.pylab as plt
    from seren3.analysis.plots import fit_scatter
    from seren3.utils import tau as tau_mod
    from scipy.interpolate import interp1d

    fig, axs = plt.subplots(2, 2, figsize=(11,10))

    plot_PLANCK=True
    for sim, iouts, ppath, label, c in zip(sims, sim_iouts, pickle_paths, labels, cols):
        print ppath
        cosmo = sim[iouts[0]].cosmo
        data = pickle.load( open("%s/T_time_averaged.p" % ppath, "rb") )
        # data = pickle.load( open("%s/xHII_reion_history.p" % ppath, "rb") )
        z = np.zeros(len(data))
        var = np.zeros(len(data))

        for i in range(len(data)):
            res = data[i].result
            z[i] = res['z']
            var[i] = res['mw']
            # var[i] = res["volume_weighted"]

        tau, redshifts = tau_mod.interp_xHe(var, z, sim)
        tau_mod.plot(tau, redshifts, ax=axs[1,1], plot_PLANCK=plot_PLANCK, label=label, color=c)
        plot_PLANCK=False

        idx = np.where(z <= 18.)
        z = z[idx]
        var = var[idx]

        bc, mean, std, stderr = fit_scatter(z, var, ret_sterr=True)
        # fn = interp1d(bc, mean, fill_value="extrapolate")
        fn = interp1d(z, var, fill_value="extrapolate")

        # Compute Mc
        output = fit_sim_iouts(iouts, ppath, fix_alpha, True,\
                0.5, 20., 1., **cosmo)
                # 0.15, 50., 1., **cosmo)
                # 0.3, 20., 1., **cosmo)
        z_sim = np.array([sim[i].z for i in iouts])
        Mc = np.zeros(len(output))
        Mc_stderr = np.zeros(len(output))

        for i in range(len(output)):
            iout = iouts[i]
            Mc[i] = output[iout]["Mc"]["fit"]
            Mc_stderr[i] = output[iout]["Mc"]["stderr"]

        # Plot Mc
        axs[0,0].errorbar(z_sim, Mc, yerr=Mc_stderr, label=label, linewidth=2., color=c)
        axs[0,1].errorbar(fn(z_sim), Mc, yerr=Mc_stderr, label=label, linewidth=2., color=c)

        axs[1,0].plot(z, 1.-var, label=label, color=c)
        #p = axs[1,0].errorbar(bc, mean, yerr=stderr, linewidth=1, label=label)
        #children = p.get_children()
        #l = children[0]
        #axs[1,0].plot(bc, fn(bc), linewidth=4., linestyle='--', color=l.get_color(), zorder=10)

    Ok_z = np.linspace(6, 10, 100)
    Ok_fn = Okamoto_Mc_fn()
    Ok_Mc = np.array([Ok_fn(i) for i in Ok_z])

    axs[0,0].plot(Ok_z, Ok_Mc, color='g', label="Okamoto et al. 08", linestyle='-.')

    axs[0,0].set_xlabel(r"$z$")
    axs[0,0].set_ylabel(r"$M_{\mathrm{c}}$ [M$_{\odot}$/h]")
    axs[0,0].set_xlim(6, 14)
    axs[0,0].set_ylim(1e7, 2e8)

    axs[0,1].set_xlabel(r"$\langle x_{\mathrm{HII}} \rangle_{V}$")
    axs[0,1].set_ylabel(r"$M_{\mathrm{c}}$ [M$_{\odot}$/h]")
    # axs[0,1].set_xscale("log")
    axs[0,1].set_xlim(0.2, 1.05)
    axs[0,1].set_ylim(1e7, 2e8)

    axs[1,0].set_xlabel(r"$z$")
    axs[1,0].set_ylabel(r"$\langle x_{\mathrm{HI}} \rangle_{V}$")
    # axs[1,0].set_xlim(6, 18)
    axs[1,0].set_xlim(6, 14)

    for ax in axs.flatten()[:-1]:
        ax.set_yscale("log")
    for ax in axs.flatten():
        ax.legend()

    plt.tight_layout()
    # fig.savefig('./Mc_panel_plot.pdf', format='pdf', dpi=10000)

def gnedin_fitting_func(Mh, Mc, alpha=_DEFAULT_ALPHA, **cosmo):
    f_bar = cosmo["omega_b_0"] /  cosmo["omega_M_0"]
    return f_bar * (1 + (2**(alpha/3.) - 1) * (Mh/Mc)**(-alpha))**(-3./alpha)


def lmfit_gnedin_fitting_func(params, mass, data, **cosmo):
    # For use with the lmfit module

    Mc = params["Mc"].value
    alpha = params["alpha"].value
    f_bar = cosmo["omega_b_0"] /  cosmo["omega_M_0"]
    model = f_bar * (1 + (2**(alpha/3.) - 1) * (mass/Mc)**(-alpha))**(-3./alpha)
    return model - data  # what we want to minimise


def fit(mass, fb, fix_alpha, use_lmfit=True, **cosmo):
    import scipy.optimize as optimization

    # Make an initial guess at Mc
    cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    fb_cosmic_mean = fb/cosmic_mean_b

    idx_Mc_guess = np.abs( fb_cosmic_mean - 0.5 ).argmin()
    Mc_guess = mass[idx_Mc_guess]

    p0 = [Mc_guess]
    if fix_alpha is False:
        alpha_guess = _DEFAULT_ALPHA
        p0.append(alpha_guess)

    if use_lmfit:
        # Alternative least-squares fitting routine

        from lmfit import minimize, Parameters
        fit_params = Parameters()
        fit_params.add("Mc", value=p0[0], min=0.)
        fit_params.add("alpha", value=_DEFAULT_ALPHA, vary=np.logical_not(fix_alpha), min=0.)
        # print fit_params
        result = minimize( lmfit_gnedin_fitting_func, fit_params, args=(mass, fb), kws=cosmo)
        if result.success:
            Mc_res = result.params['Mc']
            alpha_res = result.params['alpha']
            return {"Mc" : {"fit" : Mc_res.value, "stderr" : Mc_res.stderr},\
                    "alpha" : {"fit" : alpha_res.value, "stderr" : alpha_res.stderr}}
        else:
            raise Exception("Could not fit params: %s" % result.message)
    else:
        # Curve fit
        fn = lambda *args: gnedin_fitting_func(*args, **cosmo)
        popt, pcov = optimization.curve_fit( fn, mass, fb, p0=p0, maxfev=1000 )

        # Fit
        Mc_fit = popt[0]

        # Errors
        sigma_Mc = np.sqrt(pcov[0,0])

        if fix_alpha:
            return {"Mc" : {"fit" : Mc_fit, "stderr" : sigma_Mc},\
                    "alpha" : {"fit" : _DEFAULT_ALPHA, "stderr" : None}}
        else:
            alpha_fit = popt[1]
            sigma_alpha = np.sqrt(pcov[1,1])

            # correlation between Mc and alpha
            corr = pcov[0,1] / (sigma_Mc * sigma_alpha)
            print 'corr = ', corr
            return {"Mc" : {"fit" : Mc_fit, "stderr" : sigma_Mc},\
                    "alpha" : {"fit" : alpha_fit, "stderr" : sigma_alpha},\
                    "corr" : corr}


def estimate_unresolved(snapshot, halo):
    '''
    For halos whos rvir is unresolved, we reset the radius to have the minimum cell width
    and estimate the enclosed mass.
    Warning: these halos are completely unresolved objects, and this function is used at the
    users discretion.
    '''
    dx = snapshot.array(snapshot.info["boxlen"]/2**snapshot.info["levelmax"],\
                     snapshot.info["unit_length"])
    sphere = halo.sphere
    rvir = sphere.radius  # code units
    sphere.radius = dx

    subsnap = snapshot[sphere]
    gas_dset = subsnap.g["mass"].flatten()
    # print len(gas_dset['mass']), gas_dset['mass'].units

    ncell = len(gas_dset["mass"])
    print ncell
    fact = rvir.in_units(dx.units)/sphere.radius
    # print rvir, fact

    gas_mass_tot = gas_dset["mass"].in_units(_MASS_UNIT).sum()
    return gas_mass_tot * fact


def main(path, iout, pickle_path, allow_estimation=False):
    import seren3
    import pickle, os
    from seren3.analysis.parallel import mpi
    import random

    mpi.msg("Loading data")
    sim = seren3.init(path)
    # snap = seren3.load_snapshot(path, iout)
    snap = sim.snapshot(iout)

    # Age function and age now to compute time since last MM
    age_fn = sim.age_func()
    # age_now = snap.age
    age_now = age_fn(snap.z)

    snap.set_nproc(1)  # disbale multiprocessing/threading

    # Use consistent trees halo catalogues to compute time since last major merger
    # finder = "ahf"
    finder = "ctrees"
    # halos = snap.halos(finder="ctrees")
    halos = snap.halos(finder=finder)

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        mpi.msg("Working on halo %i \t %i" % (i, h.hid))

        part_dset = h.p[["id", "mass", "epoch"]].flatten()
        ix_dm = np.where(np.logical_and( part_dset["id"] > 0., part_dset["epoch"] == 0 ))  # index of dm particles
        ix_stars = np.where( np.logical_and( part_dset["id"] > 0., part_dset["epoch"] != 0 ) )  # index of star particles
        np_dm = len(ix_dm[0])

        # if np_dm > 50.:
        gas_dset = h.g["mass"].flatten()
        ncell = len(gas_dset["mass"])
        gas_mass_tot = 0.
        if ncell == 0 and allow_estimation:
            mpi.msg("Estimating gas mass for halo %s" % h["id"])
            gas_mass_tot = estimate_unresolved(snap, h)
            ncell = 1
        else:
            gas_mass_tot = gas_dset["mass"].in_units(_MASS_UNIT).sum()

        part_mass_tot = part_dset["mass"].in_units(_MASS_UNIT).sum()
        star_mass_tot = part_dset["mass"].in_units(_MASS_UNIT)[ix_stars].sum()
        # gas_mass_tot = gas_dset["mass"].in_units(_MASS_UNIT).sum()

        tot_mass = part_mass_tot + gas_mass_tot

        fb = (gas_mass_tot + star_mass_tot)/tot_mass
        sto.idx = h["id"]

        if finder == "ctrees":
            # Compute time since last major merger and store
            pid = h["pid"]  # -1 if distinct halo
            scale_of_last_MM = h["scale_of_last_mm"]  # aexp of last major merger
            z_of_last_MM = (1./scale_of_last_MM)-1.
            age_of_last_MM = age_fn(z_of_last_MM)

            time_since_last_MM = age_now - age_of_last_MM
            sto.result = {"fb" : fb, "tot_mass" : tot_mass,\
                     "pid" : pid, "np_dm" : np_dm, "ncell" : ncell,\
                     "time_since_last_MM" : time_since_last_MM, "hprops" : h.properties}
            # else:
            #     mpi.msg("Skipping halo with %i dm particles" % np_dm)
        else:
            sto.result = {"fb" : fb, "tot_mass" : tot_mass,\
                     "np_dm" : np_dm, "ncell" : ncell,\
                     "hprops" : h.properties}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/fbaryon_tdyn_%05i.p" % (pickle_path, iout)
        if allow_estimation:
            fname = "%s/fbaryon_tdyn_gdset_est_%05i.p" % (pickle_path, iout)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]
    try:
        main(path, iout, pickle_path)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        mpi.terminate(500, e=e)

