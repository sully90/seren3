import numpy as np
import seren3

# the_mass_bins=[7., 8., 9., 10.]
def plot(path, iout, pickle_path, the_mass_bins=[7., 8., 9., 10,], lab='', ax=None, **kwargs):
    import pickle
    from seren3.analysis.plots import fit_scatter
    from seren3.utils import flatten_nested_array
    import matplotlib.pylab as plt

    sim = seren3.init(path)
    # snap = seren3.load_snapshot(path, iout)
    snap = sim[iout]
    fname = "%s/sfr_halos_%05i.p" % (pickle_path, iout)
    data = pickle.load(open(fname, 'rb'))

    sfr_halos = []; bin_centre_halos = []; Mvir = []

    for i in range(len(data)):
        res = data[i].result
        sfr_halos.append(res["SFR"].in_units("Msol yr**-1"))
        bin_centre_halos.append(res["lookback-time"])
        Mvir.append(res["Mvir"])

    sfr_halos = np.array(sfr_halos); bin_centre_halos = np.array(bin_centre_halos); Mvir = np.array(Mvir)

    idx = np.where(np.log10(Mvir) >= 6.5)
    sfr_halos = sfr_halos[idx]; bin_centre_halos = bin_centre_halos[idx]; Mvir = Mvir[idx]

    # Bin idx
    mass_bins = np.digitize(np.log10(Mvir), the_mass_bins, right=True)
    binned_sfr = {}

    nbins = kwargs.pop("nbins", 100)
    # nbins = kwargs.pop("nbins", 50)
    for i in range(len(the_mass_bins)+1):
        if (i == len(the_mass_bins)):
            x, y = ( flatten_nested_array(bin_centre_halos), flatten_nested_array(sfr_halos) )
            idx = np.where(~np.isnan(y))
            binned_sfr["all"] = fit_scatter(x[idx], y[idx], ret_sterr=True, ret_n=True, nbins=nbins)
        else:
            idx = np.where( mass_bins == i )
            x, y = ( flatten_nested_array(bin_centre_halos[idx]), flatten_nested_array(sfr_halos[idx]) )
            idx = np.where(~np.isnan(y))
            binned_sfr[i] = fit_scatter(x[idx], y[idx], ret_sterr=True, ret_n=True, nbins=nbins)

    binned_sfr['the_mass_bins'] = the_mass_bins

    if ax is None:
        # return binned_sfr
        fig, ax = plt.subplots(figsize=(8,8))
        # ax = plt.gca()

    cols = None
    if "cols" not in kwargs:
        from seren3.utils import plot_utils
        cols = plot_utils.ncols(len(the_mass_bins), cmap=kwargs.pop("cmap", "Set1"))[::-1]
        # cols = plot_utils.ncols(len(the_mass_bins), cmap=kwargs.pop("cmap", "Set1"))[::-1]
        # cols = ["r", "b", "darkorange", "k"]
    else:
        cols = kwargs.pop("cols")
    ls = kwargs.pop("linestyle", "-")
    lw = kwargs.get("lw", 2.5)

    z_fn = sim.redshift_func(zmax=1000., zmin=0.)
    age_fn = sim.age_func()
    sim_age = age_fn(snap.z)

    for i,c in zip(range(len(the_mass_bins)), cols):
    # for i,c in zip([1, len(the_mass_bins)-1], cols):
        x,y,std,stderr,n = binned_sfr[i]

        age = sim_age - x
        age_to_z = z_fn(age)

        if i == len(the_mass_bins) - 1:
            upper = r"$\infty$"
        else:
            upper = "%1.1f" % the_mass_bins[i+1]
        lower = "%1.1f" % the_mass_bins[i]

        # label = "BC03 log(M) = [%s, %s)" % (lower, upper)
        label = "%s log(M) = [%s, %s)" % (lab, lower, upper)
        if kwargs.get("legend", False) is False:
            label = None
        # print label, n

        # ax.errorbar(x, y, yerr=std, label=label, linewidth=3., \
        #         color=c, linestyle=ls, capsize=5)
        from scipy import integrate
        from seren3.array import SimArray
        integrated_mstar = integrate.trapz(y, SimArray(x, "Gyr").in_units("yr"))
        print integrated_mstar
        # ax.step(x, y, label=label, linewidth=2.5, color=c, linestyle=ls)
        ax.step(age_to_z, y, label=label, linewidth=lw, color=c, linestyle=ls)

    # ax.set_xlabel(r"Lookback-time [Gyr]")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")

    if kwargs.pop("legend", False):
        # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.15,
        #                  box.width, box.height * 0.9])

        # Put a legend to the right of the current axis
        # leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        #   fancybox=True, shadow=False, ncol=6, prop={"size":18})
        leg = ax.legend(loc="upper right", prop={"size":14})
        # leg.set_title(r"log$_{10}$(Mvir) [M$_{\odot}$/h]", prop = {'size':'x-large'})

    # ax.set_ylim(-0.05, 1.05)
    ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.set_ylim(1e-6, 5e0)
    ax.set_xlim(6., 20.)
    # ax.set_xlim(0.,)

    return binned_sfr

def main(path, ioutput, pickle_path):
    import seren3
    from seren3.analysis import stars
    from seren3.analysis.parallel import mpi

    snap = seren3.load_snapshot(path, ioutput)

    age_max = snap.array(0.9279320933559091, "Gyr")
    age_min = snap.array(0., "Gyr")
    agerange = [age_min, age_max]

    snap.set_nproc(1)
    halos = snap.halos()
    nhalos = len(halos)

    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest, print_stats=False):
        h = halos[i]

        dset = h.s[["age", "mass"]].flatten()

        if (len(dset["age"]) > 0):
            star_mass = h.s["mass"].flatten()["mass"].in_units("Msol")

            sSFR, SFR, lookback_time, binsize = stars.sfr(h, dset=dset, ret_sSFR=True, agerange=agerange)
            # sSFR = SFR / star_mass.sum()

            rho_sfr = stars.gas_SFR_density(h)

            mpi.msg("%i %e %e" % (h["id"], SFR.max(), rho_sfr.max()))

            sto.idx = h["id"]
            sto.result = {"SFR" : SFR, "sSFR" : sSFR, "lookback-time" : lookback_time,\
                    "binsize" : binsize, "rho_sfr" : rho_sfr, "Mvir" : h["Mvir"]}

    if mpi.host:
        import os, pickle

        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/sfr_halos_%05i.p" % (pickle_path, ioutput)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    ioutput = int(sys.argv[2])
    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, ioutput, pickle_path)