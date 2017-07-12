def plot_outflow_fesc(sim, halo):
    '''
    Plot outflow rate (dm/dt) and fesc for this
    halo
    '''
    import random
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.scripts.mpi import history_mass_flux_all_halos
    from seren3.analysis.stars import sfr as sfr_fn

    reload(history_mass_flux_all_halos)

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

    fesc_res = load_halo(halo)
    mass_flux_res = history_mass_flux_all_halos.load_halo(halo)

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
    ax1.plot(mass_flux_time, F_plus, color="darkorange", linewidth=4., linestyle='-.', label="Outflow rate")
    ax1.plot(mass_flux_time, np.abs(F_minus)/np.abs(F_minus).max(), color="g", linewidth=4., linestyle=':', label="Inflow rate")
    # ax2.fill_between(rho_sfr_lbtime.in_units("Myr"), rho_sfr, color=sSFR_col, alpha=0.2)

    ax1.set_xlabel(r"t [Myr]")
    ax1.set_ylabel(r"$dM/dt$ [M$_{\odot}$ yr$^{-1}$]")
    ax2.set_ylabel(r"f$_{\mathrm{esc}}$")

    for tl in ax2.get_yticklabels():
        tl.set_color(ax2_col)
    ax2.yaxis.label.set_color(ax2_col)

    # Put redshift on upper x axis
    xticks = halo.base.array(ax1.get_xticks(), "Myr")
    ax_z = ax1.twiny()
    age_array = (age_now - xticks[::-1]).in_units("Gyr")
    z_fn = sim.redshift_func()

    ax_z_xticks = z_fn(age_array)
    ax2.set_xlim(0., fesc_lbtime.in_units("Myr").max())
    ax_z.set_xticks(xticks)

    ax_z_xticks = ["%1.1f" % zi for zi in ax_z_xticks]
    ax_z.set_xticklabels(ax_z_xticks)
    ax_z.set_xlabel("z")

    ax1.set_yscale("log")
    yticks = ax1.get_yticks()
    # ax1.set_ylim(yticks.min(), yticks.max()*10.)
    ax1.set_ylim(SFR.in_units("Msol yr**-1").min()/10., yticks.max()*10.)
    ax1.grid(True)
    ax1.legend(frameon=True, loc="upper right", prop={"size" : 16})
    plt.show()
    return fesc_res, mass_flux_res
    # return age_array, z_fn, xticks


def load(snap):
    import pickle

    data = pickle.load( open("%s/pickle/ConsistentTrees/time_int_fesc_all_halos_%05i.p" % (snap.path, snap.ioutput), "rb") )
    return data


def load_halo(halo):
    data = load(halo.base)

    for i in range(len(data)):
        if (int(data[i].idx) == int(halo["id"])):
            return data[i].result


def plot(snap, idata, data=None):
    from matplotlib import rcParams
    rcParams['figure.figsize'] = 16, 6
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    rcParams['axes.labelsize'] = 20
    rcParams['xtick.major.pad'] = 10
    rcParams['ytick.major.pad'] = 10

    import matplotlib.pylab as plt

    if (data is None):
        data = load(snap)
    item = data[idata]

    halos = snap.halos(finder="ctrees")
    hid = int(item.idx)
    h = halos.with_id(hid)

    print "Mvir = %1.2e [Msol/h]" % h["Mvir"]

    tint_fesc = item.result["tint_fesc_hist"]
    I1 = item.result["I1"]
    I2 = item.result["I2"]
    lbtime = item.result["lbtime"]

    fesc = I1/I2

    print (fesc*100.).min(), (fesc*100.).max()

    from seren3.analysis.stars import sfr as sfr_fn

    nbins=75
    agerange = [0, lbtime.max()]
    sSFR, SFR, sSFR_lookback_time, sSFR_binsize = sfr_fn(h, ret_sSFR=True, nbins=nbins, agerange=agerange)

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    lbtime_units = "Myr"
    ax1.semilogy(lbtime.in_units(lbtime_units), tint_fesc*100., color='k', linewidth=2.5, linestyle='--')
    ax1.semilogy(lbtime.in_units(lbtime_units), fesc*100., color='#ff3300', linewidth=2.5)

    # sSFR
    sSFR_col = '#0099cc'
    ax2.fill_between(sSFR_lookback_time.in_units(lbtime_units), sSFR, color=sSFR_col, alpha=0.2)
    ax2.set_ylim(1e-1, 1.1e2)
    ax2.set_yscale("log")

    for tl in ax2.get_yticklabels():
        tl.set_color(sSFR_col)
    ax2.set_ylabel(r"sSFR [Gyr$^{-1}$]")
    ax2.yaxis.label.set_color(sSFR_col)

    for ax in [ax1, ax2]:
        ax.tick_params('both', length=20, width=2, which='major')
        ax.tick_params('both', length=10, width=1, which='minor')

    ax1.set_ylabel(r"f$_{\mathrm{esc}}$ [%]")
    ax1.set_xlabel(r"Lookback-time [%s]" % lbtime_units)

    ax1.set_ylim(1e-1, 1.1e2)
    # ax1.set_ylim((fesc*100.).min() - 10., (fesc*100.).max() + 10.)

    # plt.xlim(0., 200.)
    plt.tight_layout()
    plt.show()
    # plt.savefig("/home/ds381/bpass_%05d_halo%i_timeint_fesc_sSFR.pdf" % (snap.ioutput, hnum), format='pdf', dpi=2000)


def main(path, pickle_path):
    import random
    import numpy as np
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.exceptions import NoParticlesException
    from seren3.analysis.escape_fraction import time_integrated_fesc
    from seren3.scripts.mpi import write_fesc_hid_dict

    mpi.msg("Loading simulation...")
    sim = seren3.init(path)

    iout_start = max(sim.numbered_outputs[0], 60)
    back_to_aexp = sim[60].info["aexp"]
    # iouts = range(iout_start, max(sim.numbered_outputs)+1)
    print "IOUT RANGE HARD CODED"
    iouts = range(iout_start, 110)
    # iouts = [109]

    for iout in iouts[::-1]:
        snap = sim[iout]
        mpi.msg("Working on snapshot %05i" % snap.ioutput)
        snap.set_nproc(1)
        halos = snap.halos(finder="ctrees")

        halo_ids = None
        if mpi.host:
            db = write_fesc_hid_dict.load_db(path, iout)
            halo_ids = db.keys()
            random.shuffle(halo_ids)

        dest = {}
        for i, sto in mpi.piter(halo_ids, storage=dest, print_stats=True):
            h = halos.with_id(i)
            res = time_integrated_fesc(h, back_to_aexp, return_data=True)
            if (res is not None):
                mpi.msg("%05i \t %i \t %i" % (snap.ioutput, h.hid, i))
                tint_fesc_hist, I1, I2, lbtime, hids = res

                fesc = I1/I2
                sto.idx = h.hid
                sto.result = {'tint_fesc_hist' : tint_fesc_hist, 'fesc' : fesc, 'I1' : I1, \
                        'I2' : I2, 'lbtime' : lbtime, 'Mvir' : h["Mvir"], 'hids' : hids}
        if mpi.host:
            import pickle, os
            # pickle_path = "%s/pickle/%s/" % (snap.path, halos.finder)
            if not os.path.isdir(pickle_path):
                os.mkdir(pickle_path)
            pickle.dump( mpi.unpack(dest), open("%s/time_int_fesc_all_halos_%05i.p" % (pickle_path, snap.ioutput), "wb") )

        mpi.msg("Waiting...")
        mpi.comm.Barrier()

if __name__ == "__main__":
    import sys, warnings
    path = sys.argv[1]
    pickle_path = sys.argv[2]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # main(path)
        # try:
        main(path, pickle_path)
        # except Exception as e:
        #    from seren3.analysis.parallel import mpi
        #    mpi.msg("Caught exception - terminating")
        #    mpi.terminate(500, e=e)

