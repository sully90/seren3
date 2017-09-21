import numpy as np

def load_dbs(halo):
    from seren3.scripts.mpi import time_int_fesc_all_halos, history_mass_flux_all_halos

    fesc_res = time_int_fesc_all_halos.load_halo(halo)
    mass_flux_res = history_mass_flux_all_halos.load_halo(halo)

    return fesc_res, mass_flux_res

def compute_integrated_nion_esc(snapshot):
    from scipy.integrate import trapz
    from seren3.array import SimArray
    from seren3.scripts.mpi import time_int_fesc_all_halos

    def _integrate_halo(fesc_res):
        photons_escaped = SimArray(fesc_res["I1"], "s**-1").in_units("yr**-1")
        cum_photons_escaped = trapz(photons_escaped, fesc_res["lbtime"].in_units("yr"))

        return cum_photons_escaped

    fesc_db = time_int_fesc_all_halos.load(snapshot)
    nphotons_escaped = np.zeros(len(fesc_db))
    mvir = np.zeros(len(fesc_db))

    for i in range(len(fesc_db)):
        hid = int(fesc_db[i].idx)
        fesc_res = fesc_db[i].result

        nphotons_escaped[i] = _integrate_halo(fesc_res)
        mvir[i] = fesc_res["Mvir"]

    ix = np.where( ~np.isnan(nphotons_escaped) )

    nphotons_escaped = nphotons_escaped[ix]
    mvir = mvir[ix]

    return mvir, nphotons_escaped

def plot_integrated_nion_esc(snapshot, ax=None, **kwargs):
    '''
    Plot cumulative photons escaped as a function of halo mass
    '''
    import matplotlib.pylab as plt
    from seren3.analysis import plots

    if (ax is None):
        ax = plt.gca()

    color = kwargs.pop("color", "k")
    nbins = kwargs.pop("nbins", 5)
    label = kwargs.pop("label", None)
    ls = kwargs.pop("ls", "-")
    lw = kwargs.pop("lw", 1.)

    mvir, nphotons_escaped = compute_integrated_nion_esc(snapshot)

    print "%e" % nphotons_escaped.sum()
    log_mvir = np.log10(mvir)
    log_nphotons_escaped = np.log10(nphotons_escaped)

    ix = np.where( np.logical_and(np.isfinite(log_nphotons_escaped), log_mvir >= 7.5) )
    log_nphotons_escaped = log_nphotons_escaped[ix]
    log_mvir = log_mvir[ix]

    bc, mean, std, sterr = plots.fit_scatter(log_mvir, log_nphotons_escaped, nbins=nbins, ret_sterr=True)

    ax.scatter(log_mvir, log_nphotons_escaped, s=10, color=color, alpha=0.1)
    e = ax.errorbar(bc, mean, yerr=std, color=color, label=label,\
         fmt="o", markerfacecolor=color, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=lw)

    if (kwargs.pop("legend", False)):
        ax.legend(loc="lower right", frameon=False, prop={"size":16})

    ax.set_xlabel(r'log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]', fontsize=20)
    ax.set_ylabel(r'log$_{10}$ $\int_{0}^{t_{\mathrm{H}}}$ $\dot{\mathrm{N}}_{\mathrm{ion}}(t)$ f$_{\mathrm{esc}}$ ($t$) $dt$ [#]', fontsize=20)



def outflow_fesc(sim, halo, ax1=None, legend=False):
    '''
    Plot outflow rate (dm/dt) and fesc for this
    halo
    '''
    import random
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.analysis.stars import sfr as sfr_fn

    from matplotlib import rcParams
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    rcParams['axes.labelsize'] = 18
    rcParams['xtick.major.pad'] = 10
    rcParams['ytick.major.pad'] = 10

    # def _compute_rho_sfr(halo, halo_catalogue, back_to_aexp):
    #     from seren3.analysis import stars
    #     from seren3.array import SimArray
    #     rho_sfr = []
    #     age = []
    #     age_now = halo.base.age

    #     sfr, vw, mw = stars.gas_SFR_density(halo, return_averages=True)
    #     rho_sfr.append(mw)
    #     age.append(halo.base.age)

    #     for prog in halo_catalogue.iterate_progenitors(halo, back_to_aexp=back_to_aexp):
    #         sfr, vw, mw = stars.gas_SFR_density(prog, return_averages=True)
    #         rho_sfr.append(mw)
    #         age.append(prog.base.age)

    #     lbtime = SimArray(age_now - np.array(age), "Gyr")
    #     return SimArray(rho_sfr, mw.units), lbtime

    # if (rho_sfr is None) or (rho_sfr_lbtime is None):
    #     rho_sfr, rho_sfr_lbtime = _compute_rho_sfr(halo, halo_catalogue, back_to_aexp)

    # ax2_col = '#CB4335'
    ax2_col = 'r'
    # sSFR_col = '#0099cc'
    sSFR_col = 'dodgerblue'

    age_now = halo.base.age

    if ax1 is None:
        fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    fesc_res, mass_flux_res = load_dbs(halo)

    tint_fesc = fesc_res['tint_fesc_hist']
    fesc = fesc_res["fesc"]
    fesc_lbtime = fesc_res["lbtime"]

    F, F_plus, F_minus = mass_flux_res["F"].T
    mass_flux_lbtime = mass_flux_res["lbtime"]

    ax1.set_xlim(0., fesc_lbtime.in_units("Myr").max())

    nbins=150
    agerange = [0, fesc_lbtime.max()]
    sSFR, SFR, sSFR_lookback_time, sSFR_binsize = sfr_fn(halo, ret_sSFR=True, nbins=nbins, agerange=agerange)

    ix = np.where(fesc > 1.)[0]
    for ii in ix:
        fesc[ii] = random.uniform(0.9, 1.0)

    fesc_time = fesc_lbtime.in_units("Myr")[::-1]
    mass_flux_time = mass_flux_lbtime.in_units("Myr")[::-1]
    sSFR_time = sSFR_lookback_time.in_units("Myr")[::-1]

    ax2.plot(fesc_time, fesc, color="r", linewidth=2.)
    ax2.plot(fesc_time, tint_fesc, color="m", linewidth=2., linestyle="--")

    ax1.fill_between(sSFR_time, SFR.in_units("Msol yr**-1"), color=sSFR_col, alpha=0.2)
    ax1.plot(sSFR_time, SFR.in_units("Msol yr**-1"), color=sSFR_col, linewidth=3.5, label="Star Formation Rate")
    ax1.plot(mass_flux_time, F_plus, color="darkorange", linewidth=3.5, linestyle='-.', label="Outflow rate")
    ax1.plot(mass_flux_time, np.abs(F_minus), color="g", linewidth=1., linestyle=':', label="Inflow rate")
    # ax2.fill_between(rho_sfr_lbtime.in_units("Myr"), rho_sfr, color=sSFR_col, alpha=0.2)

    #Dummy
    ax1.plot([-100, -90], [-100, -90], color='r', linewidth=2., label=r"f$_{\mathrm{esc}}$($t$)")
    ax1.plot([-100, -90], [-100, -90], color="m", linewidth=.5, linestyle="--", label=r"$\langle$f$\rangle$$_{\mathrm{esc}}$($\leq t_{\mathrm{H}}$)")

    if legend:
        ax1.set_xlabel(r"$t$ [Myr]")
    ax1.set_ylabel(r"dM/d$t$ [M$_{\odot}$ yr$^{-1}$]")
    ax2.set_ylabel(r"f$_{\mathrm{esc}}$")

    for tl in ax2.get_yticklabels():
        tl.set_color(ax2_col)
    ax2.yaxis.label.set_color(ax2_col)

    # Put redshift on upper x axis
    xticks = halo.base.array(ax1.get_xticks(), "Myr")[:-1]
    ax_z = ax1.twiny()
    age_array = (age_now - xticks[::-1]).in_units("Gyr")
    z_fn = sim.redshift_func()

    ax_z_xticks = z_fn(age_array)
    ax_z.set_xlim(ax1.get_xlim())
    # ax2.set_xlim(xticks.min(), xticks.max())
    ax_z.set_xticks(xticks)
    # ax_z.set_xticks(np.linspace(0, 400, 9))

    ax_z_xticks = ["%1.1f" % zi for zi in ax_z_xticks]
    ax_z.set_xticklabels(ax_z_xticks)

    if not legend:
        ax_z.set_xlabel(r"$z$")

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    yticks = ax1.get_yticks()
    # ax1.set_ylim(yticks.min(), yticks.max()*10.)
    # ax1.set_ylim(SFR.in_units("Msol yr**-1").min(), SFR.in_units("Msol yr**-1").min() + 5.)
    max_y = max(SFR.in_units("Msol yr**-1").max(), F_plus.max(), np.abs(F_minus).max())
    ax1.set_ylim(SFR.in_units("Msol yr**-1").min(), max_y + 2.)
    # ax1.set_ylim(0.04, max_y + 2.)

    # ax1.set_ylim(1e-3, 1e2)
    # ax1.legend(frameon=True, loc="upper right", prop={"size" : 16})

    # Shrink current axis's height by 10% on the bottom
    for ax in [ax1, ax2, ax_z]:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    # Put a legend below current axis
    ax1.grid(True)

    if legend:
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                  frameon=False, ncol=5, prop={"size" : 16})

    # ax2.set_ylim(ax2.get_ylim()[0], 1.1)
    # ax2.set_ylim(0.0, 1.)
    ax2.set_ylim(0.01, 1.)
    # plt.show()
    return fesc_res, mass_flux_res, (fesc_time, mass_flux_time)
    # return age_array, z_fn, xticks