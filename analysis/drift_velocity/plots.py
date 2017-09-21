def plot_power_spectra(kbins, deltab_2, deltac_2, deltac_2_nodeconv, tf, ax=None):
    '''
    Plot density and velocity power spectra and compare with CAMB
    '''
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.cosmology.transfer_function import TF

    if ax is None:
        ax = plt.gca()

    k, pkb = tf.TF_Pk(TF.B)
    k, pkc = tf.TF_Pk(TF.C)

    ax.loglog(kbins, deltac_2, label="CDM", color="royalblue", linewidth=2.)
    ax.loglog(kbins, deltac_2_nodeconv, color="navy", linestyle='--')
    ax.loglog(kbins, deltab_2, label="Baryons", color="darkorange", linewidth=2.)

    # CAMB
    deltab_2_CAMB = pkb * (k ** 3.) / (2. * np.pi ** 2.)
    deltac_2_CAMB = pkc * (k ** 3.) / (2. * np.pi ** 2.)

    # direc = '/lustre/scratch/astro/ds381/simulations/baryon_drift/100Mpc/z200/zoom/lvl14/'
    # fname = "%s/input_powerspec_baryon.txt" % direc
    # ps_data = np.loadtxt(fname, unpack=True)
    # k = ps_data[0]
    # P_bar = ps_data[1] * (2 * np.pi) ** 3 * tf._norm

    # fname = "%s/input_powerspec_cdm.txt" % direc
    # ps_data = np.loadtxt(fname, unpack=True)
    # P_cdm = ps_data[1] * (2 * np.pi) ** 3 * tf._norm
    # deltac_2_CAMB = P_cdm * (k ** 3.)
    # deltab_2_CAMB = P_bar * (k ** 3.)

    ax.loglog(k, deltac_2_CAMB, color="royalblue", linestyle=":")
    ax.loglog(k, deltab_2_CAMB, color="darkorange", linestyle=":")

    ax.set_xlabel(r"k [Mpc$^{-1}$ h a$^{-1}$]", fontsize=20)
    # ax.set_ylabel(r"$\mathcal{P}(k)$", fontsize=20)
    ax.legend(loc="upper left", frameon=False, prop={"size" : 18})
    # plt.xlim(0.001, 100)
    ax.set_xlim(0.01, 1e4)
    ax.set_ylim(1e-12, 2)

def plot_velocity_power_spectra(kbins, vdeltab_2, vdeltac_2, tf, ax=None):
    '''
    Plot density and velocity power spectra and compare with CAMB
    '''
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.cosmology import linear_velocity_ps
    from seren3.cosmology.transfer_function import TF

    if ax is None:
        ax = plt.gca()

    k, pkb = tf.TF_Pk(TF.B)
    k, pkc = tf.TF_Pk(TF.C)

    ix = np.where(~np.isnan(vdeltab_2))

    ax.loglog(kbins[ix][3:], vdeltac_2[ix][3:], label="CDM", color="royalblue", linewidth=2.)
    ax.loglog(kbins[ix][3:], vdeltab_2[ix][3:], label="Baryons", color="darkorange", linewidth=2.)

    # CAMB
    deltab_2_CAMB = pkb * (k ** 3.) / (2. * np.pi ** 2.)
    deltac_2_CAMB = pkc * (k ** 3.) / (2. * np.pi ** 2.)
    cosmo = tf.cosmo

    vdeltab_2_CAMB = linear_velocity_ps(k, np.sqrt(deltab_2_CAMB), **cosmo)**2
    vdeltac_2_CAMB = linear_velocity_ps(k, np.sqrt(deltac_2_CAMB), **cosmo)**2

    vnorm = vdeltab_2_CAMB/deltab_2_CAMB
    k, pkb = tf.TF_Pk(TF.VBARYON)
    k, pkc = tf.TF_Pk(TF.VCDM)
    vdeltab_2_CAMB = pkb * (k ** 3.) / (2. * np.pi ** 2.) * vnorm * 0.702
    vdeltac_2_CAMB = pkc * (k ** 3.) / (2. * np.pi ** 2.) * vnorm * 0.702

    # direc = '/lustre/scratch/astro/ds381/simulations/baryon_drift/100Mpc/z200/zoom/lvl14/'
    # fname = "%s/input_powerspec_baryon.txt" % direc
    # ps_data = np.loadtxt(fname, unpack=True)
    # k = ps_data[0]
    # P_bar = ps_data[1] * (2 * np.pi) ** 3 * tf._norm

    # fname = "%s/input_powerspec_cdm.txt" % direc
    # ps_data = np.loadtxt(fname, unpack=True)
    # P_cdm = ps_data[1] * (2 * np.pi) ** 3 * tf._norm
    # deltac_2_CAMB = P_cdm * (k ** 3.)
    # deltab_2_CAMB = P_bar * (k ** 3.)

    ax.loglog(k, vdeltac_2_CAMB, color="royalblue", linestyle=":")
    ax.loglog(k, vdeltab_2_CAMB, color="darkorange", linestyle=":")

    ax.set_xlabel(r"k [Mpc$^{-1}$ h a$^{-1}$]", fontsize=20)
    ax.set_ylabel(r"$\mathcal{P}_{v}(k)$ [km s$^{-1}$]", fontsize=20)
    ax.legend(loc="lower left", frameon=False, prop={"size" : 18})
    # plt.xlim(0.001, 100)
    ax.set_xlim(0.01, 1e4)
    # ax.set_ylim(1e-12, 2)


def plot_velocity(data_9, data_14):
    import matplotlib.pylab as plt

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for ax, data in zip(axs.flatten(), [data_9, data_14]):
        kbins, deltab_2, deltac_2, deltac_2_nodeconv, tf = data
        kbins = kbins[3:]
        deltab_2 = deltab_2[3:]
        deltac_2 = deltac_2[3:]
        deltac_2_nodeconv = deltac_2_nodeconv[3:]
        plot_velocity_power_spectra(kbins, deltab_2, deltac_2, tf, ax=ax)

    # axs[0].set_ylabel(r"$\mathcal{P}(k)$", fontsize=20)
    axs[0].set_ylabel(r"$\mathcal{P}_{v}(k)$ [km s$^{-1}$]", fontsize=20)
    fig.tight_layout()
    plt.show()


def plot(data_9, data_14):
    import matplotlib.pylab as plt

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for ax, data in zip(axs.flatten(), [data_9, data_14]):
        kbins, deltab_2, deltac_2, deltac_2_nodeconv, tf = data
        kbins = kbins[ix][3:]
        deltab_2 = deltab_2[3:]
        deltac_2 = deltac_2[3:]
        deltac_2_nodeconv = deltac_2_nodeconv[3:]
        plot_power_spectra(kbins, deltab_2, deltac_2, deltac_2_nodeconv, tf, ax=ax)
        # kbins, vdeltab_2, vdeltac_2, tf = data
        # plot_velocity_power_spectra(kbins, deltab_2, deltac_2, tf, ax=ax)

    axs[0].set_ylabel(r"$\mathcal{P}(k)$", fontsize=20)
    # axs[0].set_ylabel(r"$\mathcal{P}_{v}(k)$ [km s$^{-1}$]", fontsize=20)
    fig.tight_layout()
    plt.show()


def plot_power_spectra_bias(kbins_bias, deltab_2_bias, deltac_2_bias, kbins, deltab_2, deltac_2,  tf, ax=None):
    '''
    Plot density and velocity power spectra and compare with CAMB
    '''
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.cosmology.transfer_function import TF
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(5,4,wspace=0.,hspace=0.)

    ax = fig.add_subplot(gs[2:,:])
    ax2 = fig.add_subplot(gs[:2,:], sharex=ax)

    k, pkb = tf.TF_Pk(TF.B)
    k, pkc = tf.TF_Pk(TF.C)

    ix = np.where(~np.isnan(deltab_2_bias))

    ax.loglog(kbins_bias[ix], deltac_2_bias[ix], label="CDM", color="royalblue", linewidth=2.)
    ax.loglog(kbins_bias[ix], deltab_2_bias[ix], label="Baryons", color="darkorange", linewidth=2.)

    ix = np.where(~np.isnan(deltab_2))

    ax.loglog(kbins[ix], deltac_2[ix], color="royalblue", linewidth=2., linestyle="--")
    ax.loglog(kbins[ix], deltab_2[ix], color="darkorange", linewidth=2., linestyle="--")

    ax.loglog([0.0001, 0.0001], [100, 100], color="k", linewidth=2., linestyle="-", label="Biased")
    ax.loglog([0.0001, 0.0001], [100, 100], color="k", linewidth=2., linestyle="--", label="Unbiased")

    ax2.plot(kbins_bias[ix], deltac_2_bias[ix]/deltac_2[ix], color="royalblue", linewidth=2.)
    ax2.plot(kbins_bias[ix], deltab_2_bias[ix]/deltab_2[ix], color="darkorange", linewidth=2.)
    ax2.plot(np.linspace(0.1, 3000), np.ones(50), linestyle=":", color="k", label="Unity")

    # CAMB
    deltab_2_CAMB = pkb * (k ** 3.) / (2. * np.pi ** 2.)
    deltac_2_CAMB = pkc * (k ** 3.) / (2. * np.pi ** 2.)

    # direc = '/lustre/scratch/astro/ds381/simulations/baryon_drift/100Mpc/z200/zoom/lvl14/'
    # fname = "%s/input_powerspec_baryon.txt" % direc
    # ps_data = np.loadtxt(fname, unpack=True)
    # k = ps_data[0]
    # P_bar = ps_data[1] * (2 * np.pi) ** 3 * tf._norm

    # fname = "%s/input_powerspec_cdm.txt" % direc
    # ps_data = np.loadtxt(fname, unpack=True)
    # P_cdm = ps_data[1] * (2 * np.pi) ** 3 * tf._norm
    # deltac_2_CAMB = P_cdm * (k ** 3.)
    # deltab_2_CAMB = P_bar * (k ** 3.)

    ax.loglog(k, deltac_2_CAMB, color="royalblue", linestyle=":", alpha=0.5)
    ax.loglog(k, deltab_2_CAMB, color="darkorange", linestyle=":", alpha=0.5)

    ax.set_xlabel(r"k [Mpc$^{-1}$ h a$^{-1}$]", fontsize=20)
    ax.set_ylabel(r"$\mathcal{P}(k)$", fontsize=20)
    ax.legend(loc="lower left", ncol=2, frameon=False, prop={"size" : 18})
    # plt.xlim(0.001, 100)
    ax.set_xlim(1, 2000)
    ax.set_ylim(1e-8, 2)
    ax2.set_ylim(-0.2, 1.2)

    ax2.set_ylabel(r"$b(k,v_{bc})$", fontsize=20)
    ax2.set_title(r"$|v_{bc,\mathrm{rec}}|$ = 19.06 km s$^{-1}$", fontsize=20)
    ax2.legend(loc="lower left", frameon=False, prop={"size" : 20})
    plt.setp(ax2.get_xticklabels(), visible=False)

