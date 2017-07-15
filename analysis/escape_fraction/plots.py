def load_dbs(halo):
    from seren3.scripts.mpi import time_int_fesc_all_halos, history_mass_flux_all_halos

    fesc_res = time_int_fesc_all_halos.load_halo(halo)
    mass_flux_res = history_mass_flux_all_halos.load_halo(halo)

    return fesc_res, mass_flux_res


def integrated_nescape_outflow(snapshot, halos):
    '''
    Integrate the total mass ourflowed and photons escaped for all haloes
    '''
    import numpy as np
    from scipy.integrate import trapz
    from seren3.scripts.mpi import time_int_fesc_all_halos, history_mass_flux_all_halos

    fesc_db = time_int_fesc_all_halos.load(snapshot)

    def _int_I2_weighted(I1, I2, lbtime):
        return trapz(I1, lbtime) / trapz(I2, lbtime)

    def _integrate_halo(fesc_res, mass_flux_res):
        fesc_lbtime = fesc_res["lbtime"].in_units("Myr")
        mass_flux_lbtime = mass_flux_res['lbtime'].in_units("Myr")

        tint_fesc_hist = fesc_res["tint_fesc_hist"]
        I2 = fesc_res["I2"]

        F, F_plus, F_minus = mass_flux_res["F"].transpose()

        if (len(F_plus) != len(I2)):
            return None, None

        mass_flux_I1 = F_plus * I2
        massescape_dt = np.zeros(len(mass_flux_lbtime))
        for i in xrange(len(massescape_dt)):        
            massescape_dt[i] = _int_I2_weighted(mass_flux_I1[i:], I2[i:], mass_flux_lbtime[i:])
        massescape_dt = snapshot.array(massescape_dt, "Msol yr**-1")
        # if (hasattr(F_plus, "units") is False):
        #     F_plus = snapshot.array(F_plus, "Msol yr**-1")

        idx = np.where(~np.isnan(tint_fesc_hist))
        tint_fesc_hist = tint_fesc_hist[idx]; fesc_lbtime = fesc_lbtime[idx]

        idx = np.where(~np.isnan(massescape_dt))
        massescape_dt = massescape_dt[idx]; mass_flux_lbtime = mass_flux_lbtime[idx]

        total_outflowed_mass = trapz(massescape_dt, mass_flux_lbtime.in_units("yr"))

        if len(tint_fesc_hist) == 0:
            return None, None
        return tint_fesc_hist[0], total_outflowed_mass

    tint_fesc_arr = []
    outflowed_mass_arr = []
    for i in range(len(fesc_db)):
        hid = int(fesc_db[i].idx)
        h = halos.with_id(hid)

        fesc_res = fesc_db[i].result
        mass_flux_res = history_mass_flux_all_halos.load_halo(h)
        tint_fesc, total_outflowed_mass = _integrate_halo(fesc_res, mass_flux_res)

        if (tint_fesc is None or total_outflowed_mass is None):
            continue
        elif (tint_fesc == 0 or total_outflowed_mass == 0):
            continue
        tint_fesc_arr.append(tint_fesc)
        outflowed_mass_arr.append(total_outflowed_mass)

    tint_fesc_arr = np.array(tint_fesc_arr)
    outflowed_mass_arr = np.array(outflowed_mass_arr)

    return tint_fesc_arr, outflowed_mass_arr


def outflow_fesc(sim, halo):
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

    rcParams['axes.labelsize'] = 20
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

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    fesc_res, mass_flux_res = load_dbs(halo)

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

    ax1.fill_between(sSFR_time, SFR.in_units("Msol yr**-1"), color=sSFR_col, alpha=0.2)
    ax1.plot(sSFR_time, SFR.in_units("Msol yr**-1"), color=sSFR_col, linewidth=3., label="Star Formation Rate")
    ax1.plot(mass_flux_time, F_plus, color="darkorange", linewidth=2., linestyle='-.', label="Outflow rate")
    ax1.plot(mass_flux_time, np.abs(F_minus)/np.abs(F_minus).max(), color="g", linewidth=2., linestyle=':', label="Inflow rate")
    # ax2.fill_between(rho_sfr_lbtime.in_units("Myr"), rho_sfr, color=sSFR_col, alpha=0.2)

    #Dummy
    ax1.plot([-100, -90], [-100, -90], color='r', linewidth=2., label=r"f$_{\mathrm{esc}}$")

    ax1.set_xlabel(r"t [Myr]")
    ax1.set_ylabel(r"d$M$/d$t$ [M$_{\odot}$ yr$^{-1}$]")
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
    ax_z.set_xlabel(r"$z$")

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    yticks = ax1.get_yticks()
    # ax1.set_ylim(yticks.min(), yticks.max()*10.)
    # ax1.set_ylim(SFR.in_units("Msol yr**-1").min(), SFR.in_units("Msol yr**-1").min() + 5.)
    max_y = max(SFR.in_units("Msol yr**-1").max(), F_plus.max(), F_minus.max())
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
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              frameon=False, ncol=4, prop={"size" : 16})

    # ax2.set_ylim(ax2.get_ylim()[0], 1.1)
    # ax2.set_ylim(0.0, 1.)
    ax2.set_ylim(0.01, 1.)
    plt.show()
    return fesc_res, mass_flux_res, (fesc_time, mass_flux_time)
    # return age_array, z_fn, xticks