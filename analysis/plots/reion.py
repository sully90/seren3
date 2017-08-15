import numpy as np

def interp_reion_z(z, arr, zmax=14):
    '''
    Return an interpolation function for reionization history as a
    function of redshift
    '''
    from scipy.interpolate import interp1d

    ix = np.where(z <= zmax)
    return interp1d(z[ix], arr[ix])


def dxHII_dz(dz, z, arr, **kwargs):
    '''
    Computes gradient of xHII with respect to redshift
    '''
    fn = interp_reion_z(z, arr, **kwargs)

    z_start = 6
    z_end = kwargs.pop("zmax", 14)
    z_deriv = np.arange(z_start, z_end, dz)[::-1]

    func = fn(z_deriv)
    d_func = np.gradient(func, dz)
    return z_deriv, d_func


def halo_self_shielding(h, projections=None):
    import numpy as np
    import matplotlib as mpl
    from matplotlib import ticker
    import matplotlib.pylab as plt
    from seren3.analysis.visualization import engines, operators
    from seren3.utils import plot_utils

    C = h.base.C

    fields = ["rho", "xHII", "xHII"]
    field_labels = ["\rho", "x_{\mathrm{HII}}", "x_{\mathrm{HII,min}}"]

    if projections is None:
        op1 = operators.DensityWeightedOperator("rho", h.info["unit_density"])
        op2 = operators.DensityWeightedOperator("xHII", C.none)
        op3 = operators.MinxHIIOperator(C.none)

        projections = []

        ops = [op1, op2, op3]
        camera = h.camera()
        camera.map_max_size = 4096

        for field, op in zip(fields, ops):
            eng = engines.CustomRayTraceEngine(h.g, field, op, extra_fields=["rho"])
            proj = eng.process(camera)

            projections.append(proj)

    # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5,12))
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,5))
    cm_rho = "jet_black"
    cm_xHII = "jet_black"
    # cm_xHII = plot_utils.load_custom_cmaps('blues_black_test')

    count = 0
    for proj,ax,field,fl in zip(projections, axs.flatten(), fields, field_labels):
        im = None
        cm = cm_rho
        if field == "xHII":
            cm = cm_xHII
            im = ax.imshow( np.log10(proj.map.T), vmin=-2, vmax=0, cmap=cm )
        else:
            unit = h.info["unit_density"].express(C.Msun * C.kpc**-3)
            im = ax.imshow( np.log10(proj.map.T * unit), cmap=cm )
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax)

        if count == 0:
            cbar.set_label(r"log$_{10}$ $\rho$ [M$_{\odot}$ kpc$^{-3}$]")
        elif count == 1:
            cbar.set_label(r"log$_{10}$ x$_{\mathrm{HII}}$")
        elif count == 2:
            cbar.set_label(r"log$_{10}$ x$_{\mathrm{HII,min}}$")
        count += 1
            

    plt.tight_layout()
    return projections

def load_reionization_history(simulation, pickle_path=None):
    import pickle

    if pickle_path is None:
        pickle_path = "%s/pickle/" % simulation.path

    fname = "%s/xHII_reion_history.p" % pickle_path
    xHII_data = pickle.load( open(fname, "rb") )

    z_xHII = np.zeros(len(xHII_data))
    xHII_vw = np.zeros(len(xHII_data))
    xHII_mw = np.zeros(len(xHII_data))

    for i in range(len(xHII_data)):
        res = xHII_data[i].result
        z_xHII[i] = res['z']
        xHII_vw[i] = res["volume_weighted"]
        xHII_mw[i] = res["mass_weighted"]

    return z_xHII, xHII_vw, xHII_mw

def load_Gamma_history(simulation, pickle_path=None):
    import pickle

    if pickle_path is None:
        pickle_path = "%s/pickle/" % simulation.path

    fname = "%s/Gamma_time_averaged.p" % pickle_path
    Gamma_data = pickle.load( open(fname, "rb") )

    z_Gamma = np.zeros(len(Gamma_data))
    Gamma_vw = np.zeros(len(Gamma_data))
    Gamma_mw = np.zeros(len(Gamma_data))

    for i in range(len(Gamma_data)):
        res = Gamma_data[i].result
        z_Gamma[i] = res['z']
        Gamma_vw[i] = res["vw"]
        Gamma_mw[i] = res["mw"]

    return z_Gamma, Gamma_vw, Gamma_mw

def plot(sims, labels, cols, pickle_paths=None, mode="landscape", **kwargs):
    '''
    Plots neutral fraction, tau and Gamma with observations
    '''
    import matplotlib as mpl
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.scatterpoints'] = 1
    mpl.rcParams['legend.fontsize'] = 18
    
    import numpy as np
    import matplotlib.pylab as plt
    import pickle
    from . import obs_errors
    from seren3.utils import tau as tau_mod

    reload(tau_mod)

    if (pickle_paths is None):
        pickle_paths = ["%s/pickle/" % sim.path for sim in sims]

    fig, axs = (None, None)
    if mode == "landscape":
        fig, axs = plt.subplots(1, 3, figsize=(144,4))
    elif mode == "portrait":
        fig, axs = plt.subplots(3, 1, figsize=(6,12))
    else:
        raise Exception("Unknown mode: %s. Please use 'landscape' or 'portrait'")

    plot_PLANCK=True
    plot_obs=True

    for sim, ppath, label, c in zip(sims, pickle_paths, labels, cols):
        cosmo = sim[sim.numbered_outputs[0]].cosmo

        z_xHII, xHII_vw, xHII_mw = load_reionization_history(sim, pickle_path=ppath)

        z_Gamma, Gamma_vw, Gamma_mw = load_Gamma_history(sim, pickle_path=ppath)

        # Plot neutral fraction
        axs[0].plot(z_xHII, 1. - xHII_vw, color=c, linestyle="-", linewidth=2., label=label)
        axs[0].plot(z_xHII, 1. - xHII_mw, color=c, linestyle="--", linewidth=2.)
        if (plot_obs): obs_errors("xv", ax=axs[0])

        # Plot Gamma
        axs[1].plot(z_Gamma, np.log10(Gamma_vw), color=c, linestyle="-", linewidth=2., label=label)
        axs[1].plot(z_Gamma, np.log10(Gamma_mw), color=c, linestyle="--", linewidth=2.)
        if (plot_obs): obs_errors("Gamma", ax=axs[1])
        plot_obs = False

        tau, redshifts = tau_mod.interp_xHe(xHII_vw, z_xHII, sim)
        tau_mod.plot(tau, redshifts, ax=axs[2], plot_PLANCK=plot_PLANCK, label=label, color=c)
        plot_PLANCK=False

    for ax in axs.flatten():
        ax.set_xlim(5.8, 16)

    axs[0].legend()
    axs[1].legend(loc="lower left")
    axs[2].legend()

    axs[0].set_xlabel(r"$z$")
    axs[0].set_ylabel(r"$\langle x_{\mathrm{HI}} \rangle_{V,M}$")

    axs[1].set_xlabel(r"$z$")
    axs[1].set_ylabel(r"log $_{10}$ $\langle \Gamma \rangle_{V,M}$ [$s^{-1}$]")
    axs[1].set_ylim(-18, -8)

    axs[2].set_ylim(0.0, 0.122)

    fig.tight_layout()
    # plt.show()
    fig.savefig("./reion_hist_%s.pdf" % mode, format="pdf")

