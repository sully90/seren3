import numpy as np

# From Fan et al. -> compute neutral fraction from Gunn-Peterson optical depth and redshift
xHI_fn = lambda tau, z, cosmo: tau / ( 1.8e5 * cosmo['h']**-1 * cosmo['omega_M_0']**1./2. * ( (cosmo['omega_b_0'] * cosmo['h']**2) / 0.02 ) * ( (1. + z) / 7. ) )

def solve_photoionization_rate(redshift, transmitted_flux_ratio):
    '''
    Here we integrate eqn. 8 from Fan et al. 2006 to solve for the
    (uniform) photoionization background
    NB: tau_eff = -ln(transmitted_flux_ration)
    '''
    # First, get out IGM distribution function
    from scipy import integrate
    from scipy.optimize import fsolve

    p = IGM_distribution_function(redshift)
    # vol_p = lambda Del: Del * p(Del)

    # tau as a function of overdensity
    def tau_fn_Del(Del, Gamma):
        tau0 = 82  # From McDonald & Miralda-Escude 2001
        return tau0 * ((1. + redshift)/7.)**(4.5) * (0.05/Gamma) * Del**2.

    def integrand(Gamma):
        fn = lambda Del: np.exp(-1. * tau_fn_Del(Del, Gamma)) * p(Del)
        T = integrate.quad(fn, 0., np.inf)
        print T, transmitted_flux_ratio
        return transmitted_flux_ratio - T

    init_Gamma = 0.1
    Gamma_sol = fsolve(integrand, init_Gamma)
    return Gamma_sol

# IGM distribution function
def IGM_distribution_function(redshift):
    '''
    Computes IGM distribution function using in Fan et al. 2006, with constants interpolated from
    Miralda-Escude et al. 2000
    '''
    from scipy import interpolate
    # from seren3.analysis.plots import extrap1d
    from seren3.analysis import plots
    reload(plots)

    Escude_z = np.array([2., 3., 4., 6.])
    A = np.array([0.406, 0.558, 0.711, 0.864])
    delta0 = lambda z: 7.61/(1. + z)  # rough fit
    beta = np.array([2.23, 2.35, 2.48, 2.50])
    C0 = np.array([0.558, 0.599, 0.611, 0.880])

    def interp(z, const):
        '''
        Performs linear 1D interpolation with redshift
        '''
        fn = interpolate.InterpolatedUnivariateSpline(z, const)
        return fn

    Escude_constants = {}
    for key,arr in zip(["A", "beta", "C0"],[A,beta,C0]):
        fn = interp(Escude_z, arr)
        Escude_constants[key] = fn(redshift)

    p = lambda Del: Escude_constants["A"]\
         * np.exp( -1 * ( ( Del**(-2./3.) - Escude_constants["C0"] )**2.\
            /( 2.*( (2.*delta0(redshift)/3.)**2. ) ) ) ) * Del**(-1. * Escude_constants["beta"])

    return p  # Del * p(Del) / Del**2 * p(Del) are volume/mass weighted respectively

def plot_IGM_distribution_function(**kwargs):
    '''
    For comparrison with Miralda-Escude et al. 2000
    '''
    import matplotlib.pylab as plt

    Del = np.logspace(-2, 2, 1000)

    for z,ls in zip([2, 3, 4, 7], ['-', '--', '-.', ':']):
        p = IGM_distribution_function(z)
        plt.semilogx(Del, Del * p(Del), color='k', linewidth=2., linestyle=ls, label='z=%i' % z)
        plt.semilogx(Del, Del**2 * p(Del), color='k', linewidth=1., linestyle=ls)

    plt.ylabel(r"P(log $\Delta$)")
    plt.xlabel(r"log $\Delta$")
    plt.xlim(1e-2, 1e2)
    plt.legend()
    plt.show(block=kwargs.pop("block", True))


def fan_et_al_xHI(**cosmo):
    '''
    Returns Fan et al. 2006 observations from Quasar Gunn-Peterson troughs for the neutral
    fraction (inc. error bars)
    '''
    zGP = np.array([5.025, 5.25, 5.45, 5.65, 5.85, 6.1])  # redshift bin centres
    tauGP = np.array([2.1, 2.5, 2.6, 3.2, 4.0, 7.1])  # last two are upper limits
    sigmaGP = np.array([0.3, 0.5, 0.6, 0.8, 0.8, 2.1])

    xHI = np.array( [xHI_fn(t, z, cosmo) for t,z in zip(tauGP, zGP)] )
    xHI_upper = np.array( [xHI_fn(t+s, z, cosmo) for t,s,z in zip(tauGP, sigmaGP, zGP)] )
    xHI_lower = np.array( [xHI_fn(t-s, z, cosmo) for t,s,z in zip(tauGP, sigmaGP, zGP)] )
    xHI_err = xHI_upper - xHI_lower

    return zGP, xHI, xHI_err

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


def plot_from_sims(sims, labels, cols=None, show=True, **kwargs):
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
            plot(tau, redshifts, ax=ax, label=label, color=c, **kwargs)
        else:
            # last element, draw PLANCK constraints
            plot(tau, redshifts, ax=ax, label=label, color=c, plot_PLANCK=True, **kwargs)
    if show:
        plt.legend()
        plt.show(block=False)
    return fig, ax


def plot(tau, z, ax=None, color='k', label=None, plot_WMAP7=False, plot_PLANCK=False, show=False, **kwargs):
    import matplotlib.pylab as plt

    if ax is None:
        ax = plt.gca()
    p = ax.plot(z, tau, color=color, label=label, **kwargs)
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

    ax.set_ylabel(r'$\tau_{\mathrm{e}}$')
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

    # NB: nH = [(1-Y)rho_{cr}/mH](1+z)**3 which gives (1+z)**2 in below ((1+z)**-1 in tau integral)
    integrand = -1 * tau_star * x * ((1. + z) ** 2) / Hz(z)

    integral = np.empty(integrand.shape)
    integral[..., 1:] = si.cumtrapz(integrand, z)
    integral[..., 0] = 0.0
    return np.abs(integral)
