def integrate_surface_flux(flux_map, r):
    '''
    Integrates a healpix surface flux to compute the total
    net flux out of the sphere.
    r is the radius of the sphere in meters
    '''
    import numpy as np
    import healpy as hp
    from scipy.integrate import trapz
    from seren3.array import SimArray

    if not ((isinstance(flux_map, SimArray) or isinstance(r, SimArray))):
        raise Exception("Must pass SimArrays")

    # Compute theta/phi
    npix = len(flux_map)
    nside = hp.npix2nside(npix)
    # theta, phi = hp.pix2ang(nside, range(npix))
    theta, phi = hp.pix2ang(nside, range(npix))
    r = r.in_units("kpc")  # make sure r is in meters

    # Compute the integral
    # integrand = np.zeros(len(theta))
    ix = theta.argsort()
    integrand = r**2 * np.sin(theta[ix]) * flux_map[ix]

    # for i in range(len(theta)):
    #     th, ph = (theta[i], phi[i])
    #     integrand[i] = r**2 * np.sin(th) * flux_map[i]  # mass_flux_radial function already deals with unit vev

    # integrand = integrand[:, None] + np.zeros(len(phi))  # 2D over theta and phi
    # I = trapz(trapz(integrand, phi), theta)
    I = trapz(integrand, theta[ix]) * 2.*np.pi
    return SimArray(I, "Msol yr**-1")


def dm_by_dt(subsnap, filt=False, **kwargs):
    '''
    Compute mass flux at the virial sphere
    '''
    import numpy as np
    from seren3.array import SimArray
    from seren3.analysis.render import render_spherical

    reload(render_spherical)

    rvir = SimArray(subsnap.region.radius, subsnap.info["unit_length"])
    to_distance = rvir/4.
    # to_distance = rvir
    in_units = "kg s**-1 m**-2"
    s = kwargs.pop("s", subsnap.pynbody_snapshot(filt=filt))

    if "nside" not in kwargs:
        kwargs["nside"] = 2**3
    kwargs["radius"] = to_distance
    kwargs["denoise"] = True
    im = render_spherical.render_quantity(subsnap.g, "mass_flux_radial", s=s, in_units=in_units, out_units=in_units, **kwargs)
    im.convert_units("Msol yr**-1 kpc**-2")

    def _compute_flux(im, to_distance, direction=None):
        im_tmp = im.copy()
        ix = None
        if ("out" == direction):
            ix = np.where(im_tmp < 0)
            im_tmp[ix] = 0
        elif ("in" == direction):
            ix = np.where(im_tmp > 0)
            im_tmp[ix] = 0
        else:
            return integrate_surface_flux(im, to_distance)    

        return integrate_surface_flux(im_tmp, to_distance)

    F = _compute_flux(im, to_distance)
    F_plus = _compute_flux(im, to_distance, direction="out")
    F_minus = _compute_flux(im, to_distance, direction="in")
    return (F, F_plus, F_minus), im


def integrate_dm_by_dt(I1, I2, lbtime):
    from scipy.integrate import trapz
    return trapz(I1, lbtime) / trapz(I2, lbtime)


def mass_flux_hist(halo, back_to_aexp, return_data=True, **kwargs):
    '''
    Compute history of in/outflows
    '''
    import numpy as np
    from seren3.scripts.mpi import write_mass_flux_hid_dict

    db = kwargs.pop("db", write_mass_flux_hid_dict.load_db(halo.base.path, halo.base.ioutput))
    if (int(halo["id"]) in db.keys()):
        catalogue = halo.base.halos(finder="ctrees")

        F = []
        age_arr = []
        hids = []
        iouts = []

        def _compute(h, db):
            hid = int(h["id"])
            res = db[hid]

            F.append(res["F"])
            age_arr.append(h.base.age)
            hids.append(hid)
            iouts.append(h.base.ioutput)

        _compute(halo, db)

        for prog in catalogue.iterate_progenitors(halo, back_to_aexp=back_to_aexp):
            prog_db = write_mass_flux_hid_dict.load_db(prog.base.path, prog.base.ioutput)

            if (int(prog["id"]) in prog_db.keys()):
                _compute(prog, prog_db)
            else:
                break
        F = np.array(F)
        age_arr = np.array(age_arr)
        hids = np.array(hids, dtype=np.int64)
        iouts = np.array(iouts)

        lbtime = halo.base.age - age_arr

        if return_data:    
            return F, age_arr, lbtime, hids, iouts
        return F
    else:
        return None

def fesc_tot_outflow(snapshot):
    '''
    Integrate the total mass ourflowed and photons escaped for all haloes
    '''
    import numpy as np
    from scipy.integrate import trapz
    from seren3.array import SimArray
    from seren3.scripts.mpi import time_int_fesc_all_halos, history_mass_flux_all_halos

    fesc_db = time_int_fesc_all_halos.load(snapshot)
    mass_flux_db = history_mass_flux_all_halos.load(snapshot)
    mass_flux_hids = np.array( [int(res.idx) for res in mass_flux_db] )

    def _integrate_halo(fesc_res, mass_flux_res):
        photons_escaped = SimArray(fesc_res["I1"], "s**-1").in_units("yr**-1")
        cum_photons_escaped = trapz(photons_escaped, fesc_res["lbtime"].in_units("yr"))

        F, F_plus, F_minus = mass_flux_res["F"].transpose()
        F_plus = SimArray(F_plus, "Msol yr**-1")
        F_minus = SimArray(F_minus, "Msol yr**-1")

        if (len(F_plus) != len(photons_escaped)):
            return np.nan, np.nan

        cum_outflowed_mass = trapz(F_plus, mass_flux_res["lbtime"].in_units("yr"))
        cum_inflowed_mass = np.abs(trapz(F_minus, mass_flux_res["lbtime"].in_units("yr")))

        # return cum_photons_escaped, cum_outflowed_mass - cum_inflowed_mass
        return cum_photons_escaped, cum_outflowed_mass


    nphotons_escaped = np.zeros(len(fesc_db))
    tot_mass_outflowed = np.zeros(len(fesc_db))
    mvir = np.zeros(len(fesc_db))

    for i in range(len(fesc_db)):
        hid = int(fesc_db[i].idx)
        fesc_res = fesc_db[i].result
        mass_flux_res_ix = np.abs(mass_flux_hids - hid).argmin()
        mass_flux_res = mass_flux_db[mass_flux_res_ix].result

        nphotons_escaped[i], tot_mass_outflowed[i] = _integrate_halo(fesc_res, mass_flux_res)
        mvir[i] = fesc_res["Mvir"]

    ix = np.where( np.logical_and( ~np.isnan(nphotons_escaped), ~np.isnan(tot_mass_outflowed)) )

    nphotons_escaped = nphotons_escaped[ix]
    tot_mass_outflowed = tot_mass_outflowed[ix]
    mvir = mvir[ix]

    return nphotons_escaped, tot_mass_outflowed, mvir

def fesc_mean_time_outflow(snapshot):
    '''
    Integrate the total mass outflowed and photons escaped for all haloes
    '''
    import numpy as np
    from scipy.integrate import trapz
    from seren3.array import SimArray
    from seren3.scripts.mpi import time_int_fesc_all_halos, history_mass_flux_all_halos

    fesc_db = time_int_fesc_all_halos.load(snapshot)
    mass_flux_db = history_mass_flux_all_halos.load(snapshot)
    mass_flux_hids = np.array( [int(res.idx) for res in mass_flux_db] )

    def _integrate_halo(fesc_res, mass_flux_res):
        photons_escaped = SimArray(fesc_res["I1"], "s**-1").in_units("yr**-1")
        # cum_photons_escaped = trapz(photons_escaped, fesc_res["lbtime"].in_units("yr"))
        cum_photons_escaped = fesc_res["tint_fesc_hist"][0]

        F, F_plus, F_minus = mass_flux_res["F"].transpose()
        F_plus = SimArray(F_plus, "Msol yr**-1")
        F_minus = SimArray(F_minus, "Msol yr**-1")

        if (len(F_plus) != len(photons_escaped)):
            return np.nan, np.nan

        lbtime = mass_flux_res["lbtime"]
        F_net_outflow = F_plus - np.abs(F_minus)

        if len(np.where(np.isnan(F_net_outflow))[0] > 0):
            return np.nan, np.nan

        ix = np.where(F_net_outflow < 0.)

        if len(ix[0] == 0):
            return cum_photons_escaped, lbtime[-1]
        else:
            time_outflow = [0]
            for i in ix[0]:
                if (i == 0):
                    continue
                time_outflow.append(lbtime[i - 1])
        time_spent = np.zeros(len(time_outflow) - 1)
        for i in range(len(time_spent)):
            time_spent[i] = time_outflow[i+1] - time_outflow[i]

        return cum_photons_escaped, time_spent.mean()


    nphotons_escaped = np.zeros(len(fesc_db))
    time_spent_net_outflow = np.zeros(len(fesc_db))
    mvir = np.zeros(len(fesc_db))

    for i in range(len(fesc_db)):
        hid = int(fesc_db[i].idx)
        fesc_res = fesc_db[i].result
        mass_flux_res_ix = np.abs(mass_flux_hids - hid).argmin()
        mass_flux_res = mass_flux_db[mass_flux_res_ix].result

        nphotons_escaped[i], time_spent_net_outflow[i] = _integrate_halo(fesc_res, mass_flux_res)
        mvir[i] = fesc_res["Mvir"]

    ix = np.where( np.logical_and( ~np.isnan(nphotons_escaped),\
             np.logical_and(~np.isnan(time_spent_net_outflow),\
             time_spent_net_outflow > 0) ) )

    nphotons_escaped = nphotons_escaped[ix]
    time_spent_net_outflow = time_spent_net_outflow[ix]
    mvir = mvir[ix]
    return nphotons_escaped, SimArray(time_spent_net_outflow, "Gyr"), mvir

def plot(sims, iout, labels, cols, ax=None, **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.analysis import plots

    if (ax is None):
        ax = plt.gca()

    ls = ["-", "--"]
    lw = [3., 1.5]

    for sim, label, col, lsi, lwi in zip(sims, labels, cols, ls, lw):
        snap = sim[iout]

        nphotons_escaped, tot_mass_outflowed, mvir = fesc_tot_outflow(snap)
        print "%e" % nphotons_escaped.sum()

        log_mvir = np.log10(mvir)

        x = np.log10(tot_mass_outflowed)
        y = np.log10(nphotons_escaped)
        ix = np.where(np.logical_and(log_mvir >= 7.5, x>=5.5))

        x = x[ix]
        y = y[ix]

        ix = np.where(np.logical_and(np.isfinite(x), np.isfinite(y)))

        x = x[ix]
        y = y[ix]

        bc, mean, std, sterr = plots.fit_scatter(x, y, ret_sterr=True, **kwargs)

        ax.scatter(x, y, alpha=0.10, s=5, color=col)
        e = ax.errorbar(bc, mean, yerr=std, color=col, label=label,\
             fmt="o", markerfacecolor=col, mec='k',\
             capsize=2, capthick=2, elinewidth=2, linewidth=lwi, linestyle=lsi)

        # ax.plot(bc, mean, color=col, label=None, linewidth=3., linestyle="-")
        # ax.fill_between(bc, mean-std, mean+std, facecolor=col, alpha=0.35, interpolate=True, label=label)

    ax.set_xlabel(r"log$_{10}$ $\int_{0}^{t_{\mathrm{H}}}$ $\vec{F}_{+}(t)$ $dt$ [M$_{\odot}$]", fontsize=20)
    ax.set_ylabel(r'log$_{10}$ $\int_{0}^{t_{\mathrm{H}}}$ $\dot{\mathrm{N}}_{\mathrm{ion}}(t)$ f$_{\mathrm{esc}}$ ($t$) $dt$ [#]', fontsize=20)

    ax.legend(loc='lower right', frameon=False, prop={"size" : 16})