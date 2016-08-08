import numpy as np


def interp_xHe(xion, z, sim):
    """ Interpolate for xHe fractions based on Iliev, Scannapieco & Shapiro 2005
    Input -----------------------------------
    xion - Hydrogen ionization fractions
    z - redshifts corresponding to xion

    Returns ---------------------------------
    tau - the optical depth to reionization
    """
    from scipy import interpolate

    cosmo = sim[1].cosmo
    del cosmo['z'], cosmo['aexp']
    # agefunc, zfunc = sim.z_to_age(zmax=100., zmin=0., return_inverse=True)

    redshifts = np.linspace(99., 0., num=150)
    xion_func = interpolate.interp1d(z, xion)

    idx = np.where(np.logical_and(redshifts >= z.min(), redshifts <= z.max()))
    xion_H_interp = np.zeros_like(redshifts)
    xion_H_interp[idx] = xion_func(redshifts[idx])
    idx = np.where(np.logical_and(xion_H_interp == 0., redshifts <= z.max()))
    xion_H_interp[idx] = 1.

    idx = np.where(np.logical_and(
        np.logical_and(redshifts <= 6., redshifts > 3.), xion_H_interp > 0.95))
    xion_He_interp = np.zeros_like(xion_H_interp)
    xion_He_interp[idx] = 1.
    idx = np.where(np.logical_and(redshifts <= 3., xion_H_interp > 0.95))
    xion_He_interp[idx] = 2.

    xion_H_interp = xion_H_interp[::-1]
    xion_He_interp = xion_He_interp[::-1]
    redshifts = redshifts[::-1]

    cosmo['X_H'] = 0.75
    cosmo['Y_He'] = 0.25
    tau = integrate_optical_depth(
        xion_H_interp, xion_He_interp, redshifts, **cosmo)

    return tau, redshifts


def plot_from_sims(sims, labels, cols=None, show=True):
    '''
    Plots optical depth to reionization for each sim, if reion_history table exists
    '''
    from seren3.utils import lookahead
    from seren3.scripts.mpi import reion_history
    import matplotlib.pylab as plt
    import numpy as np

    VW = "volume_weighted"
    fig, ax = plt.subplots()

    if cols == None:
        from seren3.utils.plot_utils import ncols
        cols = ncols(len(sims))

    count = 0
    for sim, has_more in lookahead(sims):
        table = reion_history.load_xHII_table(path=sim.path)
        vw = np.zeros(len(table))
        z = np.zeros(len(table))

        # Unpack the table
        for i in range(len(table)):
            vw[i] = table[i+1][VW]
            z[i] = table[i+1]["z"]

        # plot
        label = labels[count]
        c = cols[count]
        count += 1
        tau, redshifts = interp_xHe(vw, z, sim)
        if has_more:
            plot(tau, redshifts, ax=ax, label=label, color=c)
        else:
            # last element, draw PLANCK constraints
            plot(tau, redshifts, ax=ax, label=label, color=c, plot_PLANCK=True)
    if show:
        plt.legend()
        plt.show(block=False)
    return fig, ax


def plot(tau, z, ax=None, color='k', label=None, plot_WMAP7=False, plot_PLANCK=False, show=False):
    import matplotlib.pylab as plt

    if ax is None:
        ax = plt.gca()
    p = ax.plot(z, tau, color=color, label=label, linewidth=1.5)
    tau_planck, err = (0.066, 0.016)  # http://arxiv.org/pdf/1502.01589v2.pdf
    tau_WMAP7, err = (0.088, 0.015)  # http://arxiv.org/pdf/1502.01589v2.pdf

    if plot_PLANCK:    
        ax.hlines(
            tau_planck, 0., 99., label='Planck 2015 TT+lowP+lensing', linestyle='--')
        ax.fill_between(
            z, tau_planck - err, tau_planck + err, color='k', alpha=0.3)

    if plot_WMAP7:
        ax.hlines(
            tau_WMAP7, 0., 99., label='WMAP7 (Larson et al 2011)', linestyle='--')
        ax.fill_between(
            z, tau_WMAP7 - err, tau_WMAP7 + err, color='k', alpha=0.3)

    ax.set_ylabel(r'$\tau_{reion}$')
    ax.set_xlabel(r'$z$')
    ax.set_xlim(0., 14.5)
    ax.set_ylim(0., 0.12)
    
    if show:
        plt.legend()
        plt.show(block=False)
    return p


def integrate_optical_depth(x_ionH, x_ionHe, z, **cosmo):
    import seren3.cosmology as C
    import scipy.integrate as si
    H0 = cosmo['h'] * 100.
    sigma_T = 6.65e-29
    c = 3e8
    mH = 1.67372353855e-27
    mHe = 6.64647616211e-27

    # Compute densities today
    # m^-3
    n_H_0 = (C.rho_crit_now(**cosmo) * cosmo['omega_b_0'] * cosmo['X_H']) / mH
    n_He_0 = (C.rho_crit_now(**cosmo) * cosmo['omega_b_0'] * cosmo['Y_He']) / mHe
    n_p = n_H_0 + 2. * n_He_0
    n_e = n_H_0 * x_ionH + n_He_0 * x_ionHe
    x = n_e / n_p

    H0 *= (1000. / 3.08e22)

    Hz = lambda z: H0 * np.sqrt((cosmo['omega_M_0']
                                 * (1. + z) ** 3.) + cosmo['omega_lambda_0'])

    tau_star = c * sigma_T * n_p  # terms outside intergral s^-1

    integrand = -1 * tau_star * x * ((1. + z) ** 2) / Hz(z)

    integral = np.empty(integrand.shape)
    integral[..., 1:] = si.cumtrapz(integrand, z)
    integral[..., 0] = 0.0
    return np.abs(integral)
