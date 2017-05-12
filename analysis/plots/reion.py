def plot(sims, pickle_paths, labels, cols, **kwargs):
    '''
    Plots neutral fraction, tau and Gamma with observations
    '''
    import matplotlib as mpl
    import numpy as np
    import matplotlib.pylab as plt
    import pickle
    from . import obs_errors
    from seren3.utils import tau as tau_mod

    reload(mpl)

    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.scatterpoints'] = 1
    mpl.rcParams['legend.fontsize'] = 10

    fig, axs = plt.subplots(3, 1, figsize=(6,12))

    plot_PLANCK=True
    plot_obs=True

    for sim, ppath, label, c in zip(sims, pickle_paths, labels, cols):
        cosmo = sim[sim.numbered_outputs[0]].cosmo
        xHII_data = pickle.load( open("%s/xHII_reion_history.p" % ppath, "rb") )
        z_xHII = np.zeros(len(xHII_data))
        xHII_vw = np.zeros(len(xHII_data))
        xHII_mw = np.zeros(len(xHII_data))

        for i in range(len(xHII_data)):
            res = xHII_data[i].result
            z_xHII[i] = res['z']
            xHII_vw[i] = res["volume_weighted"]
            xHII_mw[i] = res["mass_weighted"]

        Gamma_data = pickle.load( open("%s/Gamma_time_averaged.p" % ppath, "rb") )
        z_Gamma = np.zeros(len(Gamma_data))
        Gamma_vw = np.zeros(len(Gamma_data))
        Gamma_mw = np.zeros(len(Gamma_data))

        for i in range(len(xHII_data)):
            res = Gamma_data[i].result
            z_Gamma[i] = res['z']
            Gamma_vw[i] = res["vw"]
            Gamma_mw[i] = res["mw"]

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
        ax.set_xlim(5.5, 16)

    axs[0].legend()
    axs[1].legend(loc="lower left")
    axs[2].legend()

    axs[0].set_xlabel(r"$z$")
    axs[0].set_ylabel(r"$\langle x_{\mathrm{HI}} \rangle_{V}$")

    axs[1].set_xlabel(r"$z$")
    axs[1].set_ylabel(r"$\langle \Gamma \rangle_{V}$ [$s^{-1}$]")
    axs[1].set_ylim(-18, -8)

    fig.tight_layout()
    # plt.show()
    fig.savefig("./reion_tmp.pdf", format="pdf")