import numpy as np
from seren3.analysis import baryon_fraction
from seren3.analysis.baryon_fraction import neural_net2

from seren3.analysis.baryon_fraction.neural_net2 import _X_IDX, _Y_IDX, _Y_IDX_RES

def _get_dir_name(out_dir, ioutput):
    print "%s/neural-net2/%i_final/" % (out_dir, ioutput)
    return "%s/neural-net2/%i_final/" % (out_dir, ioutput)

def _get_input_fname(out_dir, ioutput, weight, pdf_sampling):
    if pdf_sampling:
        return "%s/fb_%05i_%s_ftidal_pdf_sampling.input" % (out_dir, ioutput, weight)
    return "%s/fb_%05i_%s.input" % (out_dir, ioutput, weight)

def _get_results_fname(out_dir, ioutput, weight, NN, pdf_sampling):
    if pdf_sampling:
        return "%s/fb_%05i_NN%i_%s_ftidal_pdf_sampling.results" % (out_dir, ioutput, NN, weight)    
    return "%s/fb_%05i_NN%i_%s.results" % (out_dir, ioutput, NN, weight)

def load_data(snapshot, pickle_path=None):
    import pickle

    if (pickle_path is None):
        pickle_path = "%s/pickle/" % snapshot.path

    fname = "%s/ConsistentTrees/fbaryon_tdyn_%05i.p" % (pickle_path, snapshot.ioutput)

    fb_data = pickle.load( open(fname, "rb") )

    nrecords = len(fb_data)
    hids = np.zeros(nrecords)
    mvir = np.zeros(nrecords); fb = np.zeros(nrecords)
    np_dm = np.zeros(nrecords); ncell = np.zeros(nrecords)

    # Use the tidal force, in dimensionless units of Rvir/Rhill (Hill = Hill sphere, or sphere of influence over satellites)
    # averaged over a single dynamical time.

    tidal_force_tdyn = np.zeros(nrecords)
    pid = np.zeros(nrecords)

    for i in range(nrecords):
        res = fb_data[i].result
        hids[i] = int(fb_data[i].idx)
        mvir[i] = res["tot_mass"]; fb[i] = res["fb"]
        np_dm[i] = res["np_dm"]; ncell[i] = res["ncell"]
        pid[i] = res["pid"]
        tidal_force_tdyn[i] = res["hprops"]["tidal_force_tdyn"]

    return hids, mvir, fb, tidal_force_tdyn, pid, np_dm, ncell


# def filter_and_load_data(snapshot, idx, pickle_path=None):
#     hids, mvir, fb, tidal_force_tdyn, pid, np_dm, ncell = load_data(snapshot, pickle_path=pickle_path)

#     # idx = np.where(np.logical_and(pid == -1, \
#     #     np.logical_and(np_dm >= 20, \
#     #         np.logical_and(ncell >= 20, tidal_force_tdyn <= 0.3))))

#     # idx = np.where(np.logical_and(pid == -1, \
#     #     np.logical_and(np_dm >= 20, \
#     #         np.logical_and(ncell >= 20, tidal_force_tdyn <= np.inf))))

#     # _MASS_CUTOFF = 5e6

#     # idx = np.where(np.logical_and(np.logical_and(pid == -1, mvir >= _MASS_CUTOFF), np.logical_and(np_dm >= 50, ncell >= 50)))

#     return hids[idx], mvir[idx], fb[idx], tidal_force_tdyn[idx], pid[idx], np_dm[idx], ncell[idx]


def ANN_RMS(snapshot, pickle_path):
    '''
    Compute RMS error of the ANN for this output
    '''
    log_mvir, fb, ftidal, xHII, T, T_U, pid = neural_net2.load_training_arrays(snapshot, pickle_path=pickle_path, weight="mw")

    input_fname = _get_input_fname(dir_name, ioutput, weight, pdf_sampling)
    input_data = np.loadtxt(input_fname, unpack=True)
    log_mvir_unscaled, fb_unscaled = reverse_scaling(input_data, _Y_IDX)
    mvir_unscaled = 10**log_mvir_unscaled

    log_ftidal = input_data[1]
    log_xHII = input_data[2]
    # xHII = _reverse_scaling_xHII(input_data[2])
    log_xHII = np.log10(xHII)
    T = input_data[3]
    T_U = input_data[4]

    results_fname = _get_results_fname(dir_name, ioutput, weight, NN, pdf_sampling)
    results_data = np.loadtxt(results_fname, unpack=True)

    tmp, fb_unscaled = reverse_scaling(results_data, _Y_IDX_RES)

    assert len(fb_unscaled) == len(fb)

    delta = np.zeros(len(fb))
    for i in range(len(fb)):
        delta[i] = fb[i] - fb_unscaled[i]

    rms = np.sqrt(delta.mean()**2)
    return rms

def plot_fb_panels(sim_name, ioutputs, sim_label, weight="mw", weight_label="M", **kwargs):
    import numpy as np
    import seren3
    import matplotlib.pylab as plt
    from seren3.analysis.plots import fit_scatter

    fig, axes = plt.subplots(nrows=4, ncols=len(ioutputs), sharex=True, sharey=True, figsize=(10,10))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.05)

    sim = seren3.load(sim_name)

    for i in range(len(ioutputs)):
        # ioutput = sim.redshift(zi)
        ioutput = ioutputs[i]
        snapshot = sim[ioutput]

        pickle_path = "%s/pickle/" % snapshot.path

        axs = axes[:,i]

        axs[0].set_title("z = %1.2f" % (snapshot.z))

        log_mvir, fb, ftidal, xHII, T, T_U, pid = neural_net2.load_training_arrays(snapshot, pickle_path=pickle_path, weight=weight)
        cosmo = snapshot.cosmo
        cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]

        y_min = 0.
        y_max = (fb/cosmic_mean_b).max()

        fb_cosmic_mean = fb / cosmic_mean_b

        mvir = 10**log_mvir

        log_xHI = np.log10(1. - xHII)
        log_ftidal = np.log10(ftidal)

        bc, mean, std, sterr = fit_scatter(log_mvir, fb_cosmic_mean, nbins=10, ret_sterr=True)

        def _plot(mvir, fb, carr, ax, **kwargs):
            ax.errorbar(10**bc, mean, yerr=std, linewidth=3., color="k", linestyle="--")
            return ax.scatter(mvir, fb, c=carr, **kwargs)

        labels = [r"log$_{10} \langle F_{\mathrm{tidal}} \rangle_{t_{\mathrm{dyn}}}$",\
                 r"$\langle x_{\mathrm{HII}} \rangle_{\mathrm{%s}}$" % weight_label,\
                 # r"log$_{10} \langle x_{\mathrm{HII}} \rangle_{\mathrm{%s}}$" % weight_label,\
                 r"log$_{10} \langle T \rangle_{\mathrm{%s}}$/$T_{\mathrm{vir}}$" % weight_label ,\
                 r"log$_{10}$ T/|U|"]

        count = 0
        cbar = None
        for ax, carr, lab in zip(axs.flatten(), [np.log10(ftidal), xHII, T, T_U], labels):
            ax.errorbar(10**bc, mean, yerr=std, linewidth=1.5, color="k", linestyle="--")
            plt.xlim(5e6, 1.5e10)
            sp = None
            if (count == 0):
                sp = ax.scatter(mvir, fb_cosmic_mean, c=carr, vmin=-1, vmax=4, **kwargs)
            if (count == 2):
                sp = ax.scatter(mvir, fb_cosmic_mean, c=carr, vmin=-0.5, vmax=1.5, **kwargs)
            else:
                sp = ax.scatter(mvir, fb_cosmic_mean, c=carr, **kwargs)

            if (count == 1):
                cbar = fig.colorbar(sp, ax=ax, ticks=[1., 0.75, 0.5, 0.25, 0.])
            else:    
                cbar = fig.colorbar(sp, ax=ax)
            if (i == len(ioutputs) - 1):
                cbar.set_label(lab)

            # ax.set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")

            if (i == 0):
                ax.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")

            ax.set_xscale("log")
            count += 1

            ax.set_ylim(y_min, y_max)

        axs[-1].set_xlabel(r"M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")

    for ax in axes.flatten():
        ax.set_xlim(5e6, 1.5e10)
        
    fig.tight_layout()
    # plt.show()


def plot_fb_panels_ANN(snapshot, sim_name, pickle_path, out_dir, NN, weight="mw", weight_label="M", **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.analysis.plots import fit_scatter
    from seren3.analysis.baryon_fraction.neural_net2 import _MVIR_MIN, _MVIR_MAX, _FB_MIN, _FB_MAX

    reload(neural_net2)

    log_mvir, fb, ftidal, xHII, T, T_U, pid_scaled = neural_net2.load_training_arrays(snapshot, pickle_path=pickle_path, weight=weight)
    
    cosmo = snapshot.cosmo
    cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]

    mvir = 10**log_mvir
    fb_cosmic_mean = fb / cosmic_mean_b
    bc, mean, std, sterr = fit_scatter(log_mvir, fb_cosmic_mean, nbins=10, ret_sterr=True)

    y_min = 0.
    y_max = (fb/cosmic_mean_b).max()

    pdf_sampling = kwargs.pop("pdf_sampling", False)

    def _reverse_scaling_xHII(arr):
        unscaled = arr/2.
        unscaled += 0.5
        return unscaled

    def reverse_scaling(data, y_idx):
        x,y = (data[_X_IDX], data[y_idx])

        def _reverse_scaling_mvir(arr):
            unscaled = arr/2.
            unscaled += 0.5
            unscaled *= (_MVIR_MAX - _MVIR_MIN)
            unscaled += _MVIR_MIN
            return unscaled

        def _reverse_scaling_fb(arr):
            unscaled = arr/2.
            unscaled += 0.5
            unscaled *= (_FB_MAX - _FB_MIN)
            unscaled += _FB_MIN
            return unscaled

        # return _reverse_scaling(x, x_orig), _reverse_scaling(y, y_orig)
        # return x, _reverse_scaling2(y, y_orig)
        return _reverse_scaling_mvir(x), _reverse_scaling_fb(y)

    fig, axes = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(16,12))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.25)

    def _plot(mvir, fb, carr, ax, **kwargs):
        ax.errorbar(10**bc, mean, yerr=std, linewidth=2., color="k", linestyle="--")
        return ax.scatter(mvir, fb, c=carr, **kwargs)

    labels = [r"log$_{10} \langle F_{\mathrm{tidal}} \rangle_{t_{\mathrm{dyn}}}$",\
             r"$\langle x_{\mathrm{HII}} \rangle_{\mathrm{%s}}$" % weight_label,\
             # r"log$_{10} \langle x_{\mathrm{HII}} \rangle_{\mathrm{%s}}$" % weight_label,\
             r"log$_{10} \langle T \rangle_{\mathrm{%s}}$/$T_{\mathrm{vir}}$" % weight_label,\
             r"log$_{10}$ T/|U|", r"pid"]

    axs = axes[:,0]
    axs[0].set_title("%s z = %1.2f" % (sim_name, snapshot.z))

    count = 0
    cbar = None
    for ax, carr, lab in zip(axs.flatten(), [np.log10(ftidal), xHII, T, T_U], labels):
    # for ax, carr, lab in zip(axs.flatten(), [np.log10(ftidal), np.log10(xHII), T], labels):

        sp = None
        if (count == 0):
            sp = _plot(mvir, fb_cosmic_mean, carr, ax, vmin=-2, vmax=0.5, **kwargs)
        if (count == 2):
            sp = _plot(mvir, fb_cosmic_mean, carr, ax, vmin=-0.5, vmax=1.5, **kwargs)
        else:
            sp = _plot(mvir, fb_cosmic_mean, carr, ax, **kwargs)

        if (count == 1):
            cbar = fig.colorbar(sp, ax=ax, ticks=[1., 0.75, 0.5, 0.25, 0.])
        else:
            cbar = fig.colorbar(sp, ax=ax)
        cbar.set_label(lab)

        if (count == 0):
            cbar.set_clim(-2,0.5)

        ax.set_xlim(5e6, 1.5e10)

        # ax.set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
        ax.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
        ax.set_xscale("log")
        count += 1

        ax.set_ylim(y_min, y_max)

    axs[-1].set_xlabel(r"M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")

    axs = axes[:,1]
    axs[0].set_title("ANN z = %1.2f" % (snapshot.z))

    ioutput = snapshot.ioutput
    dir_name = _get_dir_name(out_dir, ioutput)

    input_fname = _get_input_fname(dir_name, ioutput, weight, pdf_sampling)
    input_data = np.loadtxt(input_fname, unpack=True)
    log_mvir_unscaled, fb_unscaled = reverse_scaling(input_data, _Y_IDX)
    mvir_unscaled = 10**log_mvir_unscaled

    # log_ftidal = np.log10(input_data[1])
    # log_xHII = np.log10(input_data[2])
    log_ftidal = input_data[1]
    log_xHII = input_data[2]
    # xHII = _reverse_scaling_xHII(input_data[2])
    T = input_data[3]
    T_U = input_data[4]

    results_fname = _get_results_fname(dir_name, ioutput, weight, NN, pdf_sampling)
    results_data = np.loadtxt(results_fname, unpack=True)

    tmp, fb_unscaled = reverse_scaling(results_data, _Y_IDX_RES)

    fb_cosmic_mean_unscaled = fb_unscaled / cosmic_mean_b
    # fig.suptitle('ANN z = %1.2f' % snapshot.z, fontsize=16)

    bc, mean, std, sterr = fit_scatter(log_mvir_unscaled, fb_cosmic_mean_unscaled, nbins=15, ret_sterr=True)

    count = 0
    for ax, carr, lab in zip(axs.flatten(), [log_ftidal, 10**log_xHII, T, T_U], labels):
    # for ax, carr, lab in zip(axs.flatten(), [log_ftidal, log_xHII, T], labels):

        sp = None
        if (count == 0):
            sp = _plot(mvir_unscaled, fb_cosmic_mean_unscaled, carr, ax, vmin=-2, vmax=0.5, **kwargs)
        if (count == 2):
            sp = _plot(mvir_unscaled, fb_cosmic_mean_unscaled, carr, ax, vmin=-0.5, vmax=1.5, **kwargs)
        else:
            sp = _plot(mvir_unscaled, fb_cosmic_mean_unscaled, carr, ax, **kwargs)
            
        if (count == 1):
            cbar = fig.colorbar(sp, ax=ax, ticks=[1., 0.75, 0.5, 0.25, 0.])
        else:
            cbar = fig.colorbar(sp, ax=ax)
        cbar.set_label(lab)            

        if (count == 0):
            cbar.set_clim(-2,0.5)

        ax.set_xlim(5e6, 1.5e10)

        # ax.set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
        ax.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
        ax.set_xscale("log")
        count += 1

        ax.set_ylim(y_min, y_max)

    axs[-1].set_xlabel(r"M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
    
    # plt.show()


def plot_Mc_z_xHII_by_names(simnames, labels, colours, NN, fix_alpha=False, **kwargs):
    import seren3
    sims = [seren3.load(name) for name in simnames]

    ioutputs = [[42, 48, 60, 70, 80, 90, 100, 106] for n in simnames]

    pickle_paths = ["%s/pickle/" % sim.path for sim in sims]
    ann_paths = [sim.path for sim in sims]

    plot_Mc_z_xHII(sims, ioutputs, pickle_paths, ann_paths, labels, \
            colours, NN, fix_alpha=False, **kwargs)


def plot_Mc_z_xHII(simulations, simulation_ioutputs, pickle_paths, ann_out_paths, labels, \
            colours, NN, fix_alpha=False, use_lmfit=True, **kwargs):
    '''
    Plot the evolution of Mc against z and xHII
    '''
    import matplotlib.pylab as plt
    from seren3.cosmology import hoeft_Mc
    from seren3.analysis import plots
    from seren3.analysis.plots import reion
    from seren3.analysis.baryon_fraction import tidal_force
    from scipy.interpolate import interp1d
    import matplotlib.patheffects as path_effects

    reload(baryon_fraction)
    reload(neural_net2)
    reload(tidal_force)

    _Y_LIM = (5e6, 2e8)

    ANN_iouts = [106, 100, 90, 80, 70, 60, 48, 42]

    weight = kwargs.pop("weight", "mw")
    # pdf_sampling = kwargs.pop("pdf_sampling", False)
    # filter_ftidal = kwargs.pop("filter_ftidal", False)
    filter_ftidal = False
    pdf_sampling = filter_ftidal

    print "filter tidal force? ", filter_ftidal

    fig, axes = plt.subplots(2, 2, figsize=(10,10))

    axs = axes[0,:]

    err_str = "stderr" if use_lmfit else "sigma"

    for simulation, ioutputs, ppath, ann_out_path, label, color in zip(simulations, \
                simulation_ioutputs, pickle_paths, ann_out_paths, labels, colours):
        print simulation
        z_xHII, xHII_vw, xHII_mw = reion.load_reionization_history(simulation, pickle_path=ppath)

        # Keep only z<18
        idx = np.where(z_xHII <= 18.)
        z_xHII = z_xHII[idx]
        xHII_vw = xHII_vw[idx]
        xHII_mw = xHII_mw[idx]

        # Fit and interpolate reion. history
        bc, mean, std, stderr = plots.fit_scatter(z_xHII, xHII_mw, nbins=25, ret_sterr=True)
        fn_xHII = interp1d(z_xHII, xHII_mw, fill_value="extrapolate")
        # fn_xHII = interp1d(bc, mean, fill_value="extrapolate")
        fn_xHII_std = interp1d(bc, std, fill_value="extrapolate")

        # Compute Mc at each output
        z = np.zeros(len(ioutputs))
        Mc = np.zeros(len(ioutputs))
        Mc_err = np.zeros(len(ioutputs))
        # Mc_ann = np.zeros(len(ioutputs))
        # Mc_ann_err = np.zeros(len(ioutputs))
        z_ann = np.zeros(len(ANN_iouts))
        Mc_ann = np.zeros(len(ANN_iouts))
        Mc_ann_err = np.zeros(len(ANN_iouts))

        z_ann_ftidal_pdf = np.zeros(len(ANN_iouts))
        Mc_ann_ftidal_pdf = np.zeros(len(ANN_iouts))
        Mc_ann_err_ftidal_pdf = np.zeros(len(ANN_iouts))
        
        ANN_count = 0
        for i in range(len(ioutputs)):
            ioutput = ioutputs[i]
            snapshot = simulation[ioutput]

            cosmo = snapshot.cosmo
            cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]

            # hids, mvir, fb, tidal_force_tdyn, pid, np_dm, ncell = filter_and_load_data(snapshot, pickle_path=ppath)
            log_mvir, fb, ftidal, xHII, T, T_U, pid = neural_net2.load_training_arrays(snapshot, pickle_path=ppath, weight="mw")

            # Filter ftidal?
            if filter_ftidal:
                P,C,bincenters,dx,x_pdf,y_2_pdf,(ftidal_pdf,indexes,peaks_x,params,sigma) = tidal_force.tidal_force_pdf(snapshot)
                idx = np.where(ftidal <= 10**peaks_x[-1])
                # idx = np.where(np.logical_and(ftidal < 0.25, pid == -1))
                log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
                xHII = xHII[idx]; T = T[idx]; T_U = T_U[idx]; pid = pid[idx]

            mvir = 10**log_mvir
            Mc_fit_dict = baryon_fraction.fit(mvir, fb, fix_alpha, use_lmfit=use_lmfit, **cosmo)
            z[i] = snapshot.z
            Mc[i] = Mc_fit_dict["Mc"]["fit"]
            Mc_err[i] = Mc_fit_dict["Mc"][err_str] * 5  # 3 sigma

            # alpha = Mc_fit_dict["alpha"]["fit"]

            # Neural net
            if ioutput in ANN_iouts:
                print simulation, ioutput, ann_out_path, NN
                ann_Mc_fit_dict = neural_net2.compute_ANN_Mc(simulation, ioutput, ann_out_path, NN, use_lmfit=use_lmfit, fix_alpha=fix_alpha, pdf_sampling=False)
                # ann_Mc_fit_dict = neural_net2.compute_ANN_Mc(simulation, ioutput, ann_out_path, NN, alpha=alpha, use_lmfit=use_lmfit, fix_alpha=True)
                z_ann[ANN_count] = snapshot.z
                Mc_ann[ANN_count] = ann_Mc_fit_dict["Mc"]["fit"]
                Mc_ann_err[ANN_count] = ann_Mc_fit_dict["Mc"][err_str] * 5  # 3 sigma

                ann_Mc_fit_dict = neural_net2.compute_ANN_Mc(simulation, ioutput, ann_out_path, NN, use_lmfit=use_lmfit, fix_alpha=fix_alpha, pdf_sampling=True)
                z_ann_ftidal_pdf[ANN_count] = snapshot.z
                Mc_ann_ftidal_pdf[ANN_count] = ann_Mc_fit_dict["Mc"]["fit"]
                Mc_ann_err_ftidal_pdf[ANN_count] = ann_Mc_fit_dict["Mc"][err_str] * 5  # 3 sigma
                ANN_count += 1

        # Plot
        # axs[0].errorbar(z, Mc, yerr=Mc_err, linewidth=2., color=color)
        axs[0].fill_between(z, Mc - Mc_err, Mc + Mc_err, facecolor=color, alpha=0.35, interpolate=True, label=label)#, transform=trans)

        if not pdf_sampling:
            e = axs[0].errorbar(z_ann, Mc_ann, yerr=Mc_ann_err, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='None')
        else:
            e = axs[0].errorbar(z_ann, Mc_ann_ftidal_pdf, yerr=Mc_ann_err_ftidal_pdf, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='None')
        # e[1][0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
        #                           path_effects.Normal()])
        # e[1][1].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
        #                           path_effects.Normal()])
        # e[2][0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
        #                           path_effects.Normal()])

        axs[1].fill_between(fn_xHII(z), Mc - Mc_err, Mc + Mc_err, facecolor=color, alpha=0.35, interpolate=True, label=label)#, transform=trans)
        # axs[1].errorbar(fn_xHII(z), Mc, xerr=fn_xHII_std(z), yerr=Mc_err, label=label, linewidth=2., color=color)

        if not pdf_sampling:
            e = axs[1].errorbar(fn_xHII(z_ann), Mc_ann, xerr=fn_xHII_std(z_ann), yerr=Mc_ann_err, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-')
        else:
            e = axs[1].errorbar(fn_xHII(z_ann), Mc_ann_ftidal_pdf, xerr=fn_xHII_std(z_ann), yerr=Mc_ann_err_ftidal_pdf, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-')

        # c2 = None
        # if color == "b":
        #     c2 = "c"
        # elif color == "g":
        #     c2 = "k"
        # else:
        #     c2 = "m"

        # e = axs[1].errorbar(fn_xHII(z_ann_ftidal_pdf), Mc_ann_ftidal_pdf, xerr=fn_xHII_std(z_ann), yerr=Mc_ann_err_ftidal_pdf, color=c2, label="%s ANN PDF SAMPLED" % label,\
        #      fmt="o", markerfacecolor=c2, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-.', linewidth=3.)

    # Plot the Ok08 fit
    Ok_z = np.linspace(6, 10, 100)
    Ok_fn = baryon_fraction.Okamoto_Mc_fn()
    Ok_Mc = np.array([Ok_fn(i) for i in Ok_z])
    axs[0].plot(Ok_z, Ok_Mc, color='k', label="Okamoto et al. 08", linestyle='-.')
    
    Hoeft_Mc = [hoeft_Mc(zi, omega_m = cosmo["omega_M_0"]) for zi in Ok_z ]
    axs[0].plot(Ok_z, Hoeft_Mc, color='m', label="Hoeft et al. 06", linestyle='-.')

    # Axis labels and limits
    axs[0].set_xlabel(r"$z$")
    axs[0].set_ylabel(r"$M_{\mathrm{c}}$ [M$_{\odot}$/h]")
    axs[0].set_xlim(5.5, 12.5)
    axs[0].set_ylim(_Y_LIM[0], _Y_LIM[1])

    axs[1].set_xlabel(r"$\langle x_{\mathrm{HII}} \rangle_{\mathrm{M}}$")
    axs[1].set_ylabel(r"$M_{\mathrm{c}}$ [M$_{\odot}$/h]")
    # axs[1].set_xlim(0.0, 1.05)
    axs[0].set_ylim(_Y_LIM[0], _Y_LIM[1])

    for ax in axs.flatten():
        ax.set_yscale("log")
        ax.legend(prop={'size':10})

    # Tidal cutoff
    axs = axes[1,:]
    filter_ftidal = True
    pdf_sampling = filter_ftidal

    for simulation, ioutputs, ppath, ann_out_path, label, color in zip(simulations, \
                simulation_ioutputs, pickle_paths, ann_out_paths, labels, colours):
        print simulation
        z_xHII, xHII_vw, xHII_mw = reion.load_reionization_history(simulation, pickle_path=ppath)

        # Keep only z<18
        idx = np.where(z_xHII <= 18.)
        z_xHII = z_xHII[idx]
        xHII_vw = xHII_vw[idx]
        xHII_mw = xHII_mw[idx]

        # Fit and interpolate reion. history
        bc, mean, std, stderr = plots.fit_scatter(z_xHII, xHII_mw, nbins=25, ret_sterr=True)
        fn_xHII = interp1d(z_xHII, xHII_mw, fill_value="extrapolate")
        # fn_xHII = interp1d(bc, mean, fill_value="extrapolate")
        fn_xHII_std = interp1d(bc, std, fill_value="extrapolate")

        # Compute Mc at each output
        z = np.zeros(len(ioutputs))
        Mc = np.zeros(len(ioutputs))
        Mc_err = np.zeros(len(ioutputs))
        # Mc_ann = np.zeros(len(ioutputs))
        # Mc_ann_err = np.zeros(len(ioutputs))
        z_ann = np.zeros(len(ANN_iouts))
        Mc_ann = np.zeros(len(ANN_iouts))
        Mc_ann_err = np.zeros(len(ANN_iouts))

        z_ann_ftidal_pdf = np.zeros(len(ANN_iouts))
        Mc_ann_ftidal_pdf = np.zeros(len(ANN_iouts))
        Mc_ann_err_ftidal_pdf = np.zeros(len(ANN_iouts))
        
        ANN_count = 0
        for i in range(len(ioutputs)):
            ioutput = ioutputs[i]
            snapshot = simulation[ioutput]

            cosmo = snapshot.cosmo
            cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]

            # hids, mvir, fb, tidal_force_tdyn, pid, np_dm, ncell = filter_and_load_data(snapshot, pickle_path=ppath)
            log_mvir, fb, ftidal, xHII, T, T_U, pid = neural_net2.load_training_arrays(snapshot, pickle_path=ppath, weight="mw")

            # Filter ftidal?
            if filter_ftidal:
                P,C,bincenters,dx,x_pdf,y_2_pdf,(ftidal_pdf,indexes,peaks_x,params,sigma) = tidal_force.tidal_force_pdf(snapshot)
                idx = np.where(ftidal <= 10**peaks_x[-1])
                # idx = np.where(np.logical_and(ftidal < 0.25, pid == -1))
                log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
                xHII = xHII[idx]; T = T[idx]; T_U = T_U[idx]; pid = pid[idx]

            mvir = 10**log_mvir
            Mc_fit_dict = baryon_fraction.fit(mvir, fb, fix_alpha, use_lmfit=use_lmfit, **cosmo)
            z[i] = snapshot.z
            Mc[i] = Mc_fit_dict["Mc"]["fit"]
            Mc_err[i] = Mc_fit_dict["Mc"][err_str] * 5  # 3 sigma

            # alpha = Mc_fit_dict["alpha"]["fit"]

            # Neural net
            if ioutput in ANN_iouts:
                print simulation, ioutput, ann_out_path, NN
                ann_Mc_fit_dict = neural_net2.compute_ANN_Mc(simulation, ioutput, ann_out_path, NN, use_lmfit=use_lmfit, fix_alpha=fix_alpha, pdf_sampling=False)
                # ann_Mc_fit_dict = neural_net2.compute_ANN_Mc(simulation, ioutput, ann_out_path, NN, alpha=alpha, use_lmfit=use_lmfit, fix_alpha=True)
                z_ann[ANN_count] = snapshot.z
                Mc_ann[ANN_count] = ann_Mc_fit_dict["Mc"]["fit"]
                Mc_ann_err[ANN_count] = ann_Mc_fit_dict["Mc"][err_str] * 5  # 3 sigma

                ann_Mc_fit_dict = neural_net2.compute_ANN_Mc(simulation, ioutput, ann_out_path, NN, use_lmfit=use_lmfit, fix_alpha=fix_alpha, pdf_sampling=True)
                z_ann_ftidal_pdf[ANN_count] = snapshot.z
                Mc_ann_ftidal_pdf[ANN_count] = ann_Mc_fit_dict["Mc"]["fit"]
                Mc_ann_err_ftidal_pdf[ANN_count] = ann_Mc_fit_dict["Mc"][err_str] * 5  # 3 sigma
                ANN_count += 1

        # Plot
        # axs[0].errorbar(z, Mc, yerr=Mc_err, linewidth=2., color=color)
        axs[0].fill_between(z, Mc - Mc_err, Mc + Mc_err, facecolor=color, alpha=0.35, interpolate=True, label=label)#, transform=trans)

        if not pdf_sampling:
            e = axs[0].errorbar(z_ann, Mc_ann, yerr=Mc_ann_err, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='None')
        else:
            e = axs[0].errorbar(z_ann, Mc_ann_ftidal_pdf, yerr=Mc_ann_err_ftidal_pdf, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='None')
        # e[1][0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
        #                           path_effects.Normal()])
        # e[1][1].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
        #                           path_effects.Normal()])
        # e[2][0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
        #                           path_effects.Normal()])

        axs[1].fill_between(fn_xHII(z), Mc - Mc_err, Mc + Mc_err, facecolor=color, alpha=0.35, interpolate=True, label=label)#, transform=trans)
        # axs[1].errorbar(fn_xHII(z), Mc, xerr=fn_xHII_std(z), yerr=Mc_err, label=label, linewidth=2., color=color)

        if not pdf_sampling:
            e = axs[1].errorbar(fn_xHII(z_ann), Mc_ann, xerr=fn_xHII_std(z_ann), yerr=Mc_ann_err, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-')
        else:
            e = axs[1].errorbar(fn_xHII(z_ann), Mc_ann_ftidal_pdf, xerr=fn_xHII_std(z_ann), yerr=Mc_ann_err_ftidal_pdf, color=color, label="%s ANN" % label,\
                 fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-')

        # c2 = None
        # if color == "b":
        #     c2 = "c"
        # elif color == "g":
        #     c2 = "k"
        # else:
        #     c2 = "m"

        # e = axs[1].errorbar(fn_xHII(z_ann_ftidal_pdf), Mc_ann_ftidal_pdf, xerr=fn_xHII_std(z_ann), yerr=Mc_ann_err_ftidal_pdf, color=c2, label="%s ANN PDF SAMPLED" % label,\
        #      fmt="o", markerfacecolor=c2, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-.', linewidth=3.)

    # Plot the Ok08 fit
    Ok_z = np.linspace(6, 10, 100)
    Ok_fn = baryon_fraction.Okamoto_Mc_fn()
    Ok_Mc = np.array([Ok_fn(i) for i in Ok_z])
    axs[0].plot(Ok_z, Ok_Mc, color='k', label="Okamoto et al. 08", linestyle='-.')

    Hoeft_Mc = [hoeft_Mc(zi, omega_m = cosmo["omega_M_0"]) for zi in Ok_z ]
    axs[0].plot(Ok_z, Hoeft_Mc, color='m', label="Hoeft et al. 06", linestyle='-.')

    # Axis labels and limits
    axs[0].set_xlabel(r"$z$")
    axs[0].set_ylabel(r"$M_{\mathrm{c}}$ [M$_{\odot}$/h]")
    axs[0].set_xlim(5.5, 12.5)
    axs[0].set_ylim(_Y_LIM[0], _Y_LIM[1])

    axs[1].set_xlabel(r"$\langle x_{\mathrm{HII}} \rangle_{\mathrm{M}}$")
    axs[1].set_ylabel(r"$M_{\mathrm{c}}$ [M$_{\odot}$/h]")
    # axs[1].set_xlim(0.0, 1.05)
    axs[0].set_ylim(_Y_LIM[0], _Y_LIM[1])

    for ax in axs.flatten():
        ax.set_yscale("log")
        ax.legend(prop={'size':10})

    fig.tight_layout()
    #plt.show()

def main(path, iout, pickle_path, allow_estimation=False):
    import seren3
    import pickle, os
    from seren3.analysis.parallel import mpi
    import random

    def _mass_weighted_average(halo, mass_units="Msol h**-1"):
        dset = halo.g[["xHII", "T", "mass"]].flatten()

        cell_mass = dset["mass"].in_units(mass_units)

        return np.sum(dset["xHII"]*cell_mass)/cell_mass.sum(), np.sum(dset["T"]*cell_mass)/cell_mass.sum()

    mpi.msg("Loading data")
    sim = seren3.init(path)
    # snap = seren3.load_snapshot(path, iout)
    snap = sim.snapshot(iout)

    # Age function and age now to compute time since last MM
    age_fn = sim.age_func()
    # age_now = age_fn(snap.z)

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
        fb, tot_mass, ncell, np_dm = baryon_fraction.compute_fb(h, return_stats=True)

        sto.idx = h["id"]

        if finder == "ctrees":
            # Compute time since last major merger and store
            pid = h["pid"]  # -1 if distinct halo

            scale_now = h["aexp"]
            z_now = (1./scale_now)-1.
            age_now = age_fn(z_now)

            scale_of_last_MM = h["scale_of_last_mm"]  # aexp of last major merger
            z_of_last_MM = (1./scale_of_last_MM)-1.
            age_of_last_MM = age_fn(z_of_last_MM)

            xHII_mw, T_mw = _mass_weighted_average(h)

            time_since_last_MM = age_now - age_of_last_MM
            sto.result = {"fb" : fb, "tot_mass" : tot_mass,\
                     "pid" : pid, "np_dm" : np_dm, "ncell" : ncell,\
                     "time_since_last_MM" : time_since_last_MM, "hprops" : h.properties, \
                     "xHII_mw" : xHII_mw, "T_mw" : T_mw}
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
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]
    # try:
    main(path, iout, pickle_path)
    # except Exception as e:
    #     from seren3.analysis.parallel import mpi
    #     mpi.terminate(500, e=e)
