import numpy as np

_MASS_UNIT = "Msol h**-1"

def unpack(fname):
    import pickle
    # fname = "%s/pickle/fbaryon_%05i.p" % (path, iout)
    data = pickle.load( open(fname, "rb") )

    def _unpack(data):
        tot_mass, fb, pid = (np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data)))
        for i in range(len(data)):
            tot_mass[i] = data[i].result["tot_mass"]
            fb[i] = data[i].result["fb"]
            pid = data[i].result["pid"]
        return tot_mass, fb, pid

    return _unpack(data)

def plot_compare_distinct(path, ioutput, fname, last_MM_limit, colors, linestyles, np_dm_thresh, ncell_thresh):
    import pickle
    import seren3
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec

    #Plot alpha and Mc against redshift
    fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax2 = ax1.twinx()
    gs = gridspec.GridSpec(3,3,wspace=0.,hspace=0.)

    ax1 = fig.add_subplot(gs[1:,:])
    ax2 = fig.add_subplot(gs[:1,:], sharex=ax1)
    plt.setp(ax2.get_xticklabels(), visible=False)

    _SIZE = 12

    def _filter_npart_dm(mass, fb, pid, time_since_last_MM, np_dm, ncell, dm_thresh, cell_thresh):
        good = np.where(np.logical_and(np_dm >= dm_thresh, ncell >= cell_thresh))
        return mass[good], fb[good], pid[good], time_since_last_MM[good]

    snap = seren3.load_snapshot(path, ioutput)
    with open(fname, 'rb') as f:
        data = pickle.load(f)

        nrecords = len(data)
        mass = np.zeros(nrecords)
        fb = np.zeros(nrecords)
        pid = np.zeros(nrecords)
        time_since_last_MM = np.zeros(nrecords)
        np_dm = np.zeros(nrecords)
        ncell = np.zeros(nrecords)

        for i in range(nrecords):
            res = data[i].result
            mass[i] = res["tot_mass"]
            fb[i] = res["fb"]
            pid[i] = res["pid"]
            time_since_last_MM[i] = res["time_since_last_MM"]
            np_dm[i] = res["np_dm"]
            ncell[i] = res["ncell"]
            del res

        mass, fb, pid, time_since_last_MM = _filter_npart_dm(mass, fb, pid, time_since_last_MM, np_dm, ncell, np_dm_thresh, ncell_thresh)


        log_mass = np.log10(mass)
        ix_pid = np.where(pid==-1)

        cosmo = snap.cosmo
        cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
        fb_cosmic_mean = fb/cosmic_mean

        (Mc_fit_all, sigma_Mc_all), (alpha_fit_all, sigma_alpha_all), corr_all = fit(mass, fb, **cosmo)
        print Mc_fit_all, alpha_fit_all
        x = np.linspace(mass.min(), mass.max(), 1000)
        y = gnedin_fitting_func(x, Mc_fit_all, alpha_fit_all, **cosmo)
        ax2.plot(np.log10(x), y, linewidth=2., color='k', label='All', linestyle='-')

        ax1.scatter(log_mass, fb_cosmic_mean, color='b', s=_SIZE, label='All')
        ax1.scatter(log_mass[ix_pid], fb_cosmic_mean[ix_pid], color='r', s=_SIZE, label='Distinct')

        for last_MM, c, ls in zip(last_MM_limit, colors, linestyles):
            # alpha=1. - last_MM
            alpha = 1.
            ix_last_MM = np.where( np.logical_and(pid == -1, time_since_last_MM >= last_MM) )  # Last MM >= last_MM_limit*1000. Myr ago
            (Mc_fit, sigma_Mc), (alpha_fit, sigma_alpha), corr = fit(mass[ix_last_MM], fb[ix_last_MM], **cosmo)
            y = gnedin_fitting_func(mass[ix_last_MM], Mc_fit, alpha_fit, **cosmo)
            # ax2.plot(log_mass[ix_last_MM], y, linewidth=2., color='k', linestyle=ls)

            ax1.scatter(log_mass[ix_last_MM], fb_cosmic_mean[ix_last_MM], alpha=alpha, color=c, s=_SIZE, label='Last MM > %f Myr' % (last_MM*1000.))

        ax1.set_xlabel(r"log$_{10}$(M$_{\mathrm{h}}$[M$_{\odot}$/h])")
        ax1.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
        # plt.legend(loc='upper right')
        plt.xlim(7., 10.5)

        plt.show()


def interp_Okamoto_Mc(z):
    from seren3 import config
    from scipy import interpolate

    fname = "%s/Mc_Okamoto08.txt" % config.get('data', 'data_dir')
    data = np.loadtxt(fname)
    ok_a, ok_z, ok_Mc = data.T

    fn = interpolate.interp1d(ok_z, ok_Mc)
    return fn(z)

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


def plot(snapshot, fname, tidal_force_cutoff=np.inf, dm_particle_cutoff=100, ncell_cutoff=1, nbins=12):
    '''
    Plot and fit the halo baryon fraction
    '''
    import pickle
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec
    from seren3.analysis.plots import fit_scatter, grid

    _SIZE = 12  # point size
    # _MASS_CUTOFF = 2e7  # Msun/h
    _MASS_CUTOFF = 1e6  # Msun/h

    # Prepare the figure
    # fig = plt.figure()
    # gs = gridspec.GridSpec(3,3,wspace=0.,hspace=0.)

    # ax1 = fig.add_subplot(gs[1:,:])
    # ax2 = fig.add_subplot(gs[:1,:], sharex=ax1)
    # plt.setp(ax2.get_xticklabels(), visible=False)
    ax1 = plt.gca()

    cosmo = snapshot.cosmo
    cosmic_mean = cosmo["omega_b_0"]/cosmo["omega_M_0"]
    with open(fname, "rb") as f:
        data = pickle.load(f)

        # Load the required arrays. We use dm particle and cell count to filter poorly resolved halos
        # pid==-1 gives distinct halos only
        nrecords = len(data)
        mass = np.zeros(nrecords); fb = np.zeros(nrecords); pid = np.zeros(nrecords)
        time_since_last_MM = np.zeros(nrecords); np_dm = np.zeros(nrecords); ncell = np.zeros(nrecords)
        tidal_force_tdyn = np.zeros(nrecords)

        for i in range(nrecords):
            res = data[i].result
            mass[i] = res["hprops"]["mvir"]; fb[i] = res["fb"]; pid[i] = res["pid"]
            time_since_last_MM[i] = res["time_since_last_MM"]; np_dm[i] = res["np_dm"]
            ncell[i] = res["ncell"]
            tidal_force_tdyn[i] = res["hprops"]["tidal_force_tdyn"]
            del res

        # Apply resolution cutoff
        idx = np.where( np.logical_and(np_dm >= dm_particle_cutoff, ncell >= ncell_cutoff) )
        mass = mass[idx]; fb = fb[idx]; time_since_last_MM = time_since_last_MM[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

        # Keep distinct halos only
        idx = np.where(pid[idx]==-1)
        mass = mass[idx]; fb = fb[idx]; time_since_last_MM = time_since_last_MM[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

        # Apply mass cutoff
        idx = np.where(mass >= _MASS_CUTOFF)
        mass = mass[idx]; fb = fb[idx]; time_since_last_MM = time_since_last_MM[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

        # Apply time since last merger cutoff
        # idx = np.where(time_since_last_MM >= time_since_last_MM_cutoff)
        # mass = mass[idx]; fb = fb[idx]; time_since_last_MM = time_since_last_MM[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

        # Apply tidal force limit
        idx = np.where(tidal_force_tdyn <= tidal_force_cutoff)
        mass = mass[idx]; fb = fb[idx]; time_since_last_MM = time_since_last_MM[idx]; tidal_force_tdyn = tidal_force_tdyn[idx]

        bc, mean, std = fit_scatter(np.log10(mass), fb, nbins=nbins)
        bc = 10**bc
        # Fit the data
        (Mc, Mc_sigma), (alpha, alpha_sigma), corr = fit(mass, fb, **cosmo)
        # (Mc, Mc_sigma), (alpha, alpha_sigma), corr = fit(bc, mean, **cosmo)
        x = np.linspace(mass.min(), mass.max(), 1000)
        y = gnedin_fitting_func(x, Mc, alpha, **cosmo)

        print 'Mc(z=%1.1f) = %e, alpha = %f' % (snapshot.z, Mc, alpha)

        # Plot in units of the cosmic mean
        fb_cosmic_mean = fb/cosmic_mean
        y_cosmic_mean = y/cosmic_mean

        # X,Y,Z = grid(mass, fb_cosmic_mean, np.log10(1. + tidal_force_tdyn))
        # ax1.contour(X, Y, Z)

        p = ax1.scatter(mass, fb_cosmic_mean, s=_SIZE, alpha=1., c=np.log10(1.+tidal_force_tdyn), cmap="jet_black")
        cbar = plt.colorbar(p, ax=ax1)
        # cbar.set_label(r"Time Since Last MM [Gyr]")
        cbar.set_label(r"Tidal Force (Av. over dyn time)")
        # ax1.plot(x, y_cosmic_mean, color='k', linewidth=2.)

        Mc_okamoto = interp_Okamoto_Mc(cosmo["z"])
        ax1.plot(x, gnedin_fitting_func(x, Mc_okamoto, 2., **cosmo)/cosmic_mean, color='k', linewidth=2., label=r"Okamoto08 Mc(z=%1.1f) = %1.2e M$_{\odot}$/h" % (cosmo['z'], Mc_okamoto))
        ax1.plot(x, gnedin_fitting_func(x, Mc, 2., **cosmo)/cosmic_mean, color='r', linewidth=2., label=r"Fit Mc(z=%1.1f) = %1.2e M$_{\odot}$/h" % (cosmo['z'], Mc))
        # ax1.plot(x, gnedin_fitting_func(x, Mc, 1., **cosmo)/cosmic_mean, color='b', linewidth=2.)
        # ax1.plot(bc, mean/cosmic_mean, linewidth=3., color='b', linestyle='-')
        ax1.set_xscale("log")

        ax1.set_title("z = %1.1f" % snapshot.z)
        ax1.set_xlabel(r"log$_{10}$(M$_{\mathrm{h}}$[M$_{\odot}$/h])")
        ax1.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
        ax1.hlines(0.5, mass.min(), mass.max(), linestyle='--', linewidth=2.,\
                 label=r"$\frac{1}{2}$ $\frac{\Omega_{\mathrm{b}}}{\Omega_{\mathrm{M}}}$")
        plt.legend(loc='upper left')
        # plt.ylim(0., 1.5)

        plt.show()

def gnedin_fitting_func(Mh, Mc, alpha, **cosmo):
    f_bar = cosmo["omega_b_0"] /  cosmo["omega_M_0"]
    return f_bar * (1 + (2**(alpha/3.) - 1) * (Mh/Mc)**(-alpha))**(-3./alpha)


def fit(tot_mass, fb, **cosmo):
    '''
    Fit Mc and alpha
    '''
    import numpy as np
    import scipy.optimize as optimization

    log_tot_mass = np.log10(tot_mass)
    cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    fb_cosmic_mean = fb/cosmic_mean

    idx_Mc = np.abs(fb_cosmic_mean - 0.5).argmin()
    Mc = tot_mass[idx_Mc]
    alpha = 1.5
    p0 = [Mc, alpha]  # the initial guess at Mc/alpha

    def _gnedin_fitting_func(Mh, Mc, alpha):
        return gnedin_fitting_func(Mh, Mc, alpha, **cosmo)

    # Curve fit
    popt, pcov = optimization.curve_fit(_gnedin_fitting_func, tot_mass, fb, p0=p0, maxfev=1000)

    # Errors
    sigma_Mc = np.sqrt(pcov[0,0])
    sigma_alpha = np.sqrt(pcov[1,1])

    # Correlation
    corr = pcov[0,1] / (sigma_Mc * sigma_alpha)

    Mc_fit = popt[0]
    alpha_fit = popt[1]

    return (Mc_fit, sigma_Mc), (alpha_fit, sigma_alpha), corr
    # return popt, pcov


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
    halos = snap.halos(finder="ctrees")

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        mpi.msg("Working on halo %i \t %i" % (i, h.hid))

        part_dset = h.p[["id", "mass", "epoch"]].flatten()
        ix_dm = np.where(np.logical_and( part_dset["id"] >= 0., part_dset["epoch"] == 0 ))
        ix_stars = np.where( np.logical_and( part_dset["id"] >= 0., part_dset["epoch"] != 0 ) )
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

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/fbaryon_mm_%05i.p" % (pickle_path, iout)
        if allow_estimation:
            fname = "%s/fbaryon_mm_gdset_est_%05i.p" % (pickle_path, iout)
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