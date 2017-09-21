import seren3
iout = 108
# sim = seren3.load("RT2")
sin_sim = seren3.load("2Mpc_BPASS_SIN")
bin_sim = seren3.load("2Mpc_BPASS_BIN")

sin_snap = sin_sim[iout]
bin_snap = bin_sim[iout]
cosmo = sin_snap.cosmo


import matplotlib
import matplotlib.pylab as plt
from seren3.analysis.plots import fit_scatter, fit_median
nbins=5


# In[60]:

import numpy as np
import pickle
import random
from seren3.scripts.mpi import write_fesc_hid_dict

def load_fesc(snapshot):
    
    db = write_fesc_hid_dict.load_db(snapshot.path, snapshot.ioutput)
    hids = db.keys()

    mvir = np.zeros(len(hids))
    fesc = np.zeros(len(hids))
    nphotons = np.zeros(len(hids))

    count=0.0
    for i in range(len(hids)):
        hid = hids[i]
        res = db[hid]

        ifesc = res["fesc"]
        if ifesc > 1. and ifesc < 10.:
        # if ifesc > 1.:
            fesc[i] = random.uniform(0.9, 1.0)
            count += 1.0
        elif ifesc > 0. and ifesc <= 1.:
            fesc[i] = ifesc
        else:
    #         if (res["fesc"] > 10.):
    #             print "%e   %e" % (res["tot_mass"], res["fesc"])
            continue
        mvir[i] = res["hprops"]["mvir"]
        
        Nion_d_now = db[hid]["Nion_d_now"].in_units("s**-1 Msol**-1")
        star_mass = db[hid]["star_mass"].in_units("Msol")
        nphotons[i] = (Nion_d_now * star_mass).sum()

    print count/float(len(mvir))
    print len(mvir)

    ix = np.where(fesc > 0)
    fesc = fesc[ix]
    mvir = mvir[ix]
    nphotons = nphotons[ix]

    log_mvir = np.log10(mvir)
    log_fesc = np.log10(fesc)

    # ix = np.where(np.logical_and(log_mvir >= 7.5, np.log10(fesc*100.) >= -1))
    # ix = np.where(log_mvir >= 7.5)
    # log_mvir = log_mvir[ix]
    # fesc = fesc[ix]
    # nphotons = nphotons[ix]
    
    ix = np.where(~np.isnan(fesc))
    log_mvir = log_mvir[ix]
    fesc = fesc[ix]
    nphotons = nphotons[ix]
    
    print 'Loaded data for %d halos' % len(log_mvir)
    return log_mvir, fesc, nphotons

# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,6))

# axs = axes[0,:]
# iouts = [ [108, 108], [bin_sim.redshift(8), sin_sim.redshift(8)], [sin_sim.redshift(9), sin_sim.redshift(9)] ]
# for ax, iout_lst in zip(axs.flatten(), iouts):
#     data = {"BIN" : load_fesc(bin_sim[iout_lst[0]]), "SIN" : load_fesc(sin_sim[iout_lst[0]])}
#     binned_data = {}
#     cosmo = bin_sim[iout_lst[0]].cosmo

#     for key, ls, lw, c in zip(data.keys(), ["-", "--"], [3., 1.5], ["r", "b"]):
#         log_mvir, fesc, nphotons = data[key]
#         if (iout_lst[0] == 108):
#             cosmo["z"] = 6
#         if (iout_lst[0] == bin_sim.redshift(8)):
#             cosmo["z"] = 8
#         if (iout_lst[0] == bin_sim.redshift(9)):
#             cosmo["z"] = 9
#         # log_mvir, fesc, tint_fesc, nphotons = data[key]
#         # fesc_percent = tint_fesc * 100.
#         fesc_percent = fesc * 100.
#         bin_centres, mean, std, sterr = fit_scatter(log_mvir, fesc_percent, nbins=nbins, ret_sterr=True)
#         binned_data[key] = (bin_centres, mean, std)
#     #     bin_centres, median = fit_median(log_mvir, fesc_percent, nbins=nbins)
#         ax.scatter(log_mvir, fesc_percent, alpha=0.25, color=c, s=15, marker=".")
#         e = ax.errorbar(bin_centres, mean, yerr=std, color=c, label='%s z=%1.2f' % (key, cosmo['z']), fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=lw)
        
#     print binned_data["BIN"][1] / binned_data["SIN"][1]

# axs[0].set_ylabel(r'f$_{\mathrm{esc}}$(t) [%]', fontsize=20)
# for ax in axs.flatten():
#     ax.legend(loc="lower right", frameon=False, prop={"size":16})
#     # ax.set_xlabel(r'log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]', fontsize=20)
#     # ax.set_ylabel(r'f$_{\mathrm{esc}}$ (<t$_{\mathrm{H}}$) [%]', fontsize=20)
#     ax.set_yscale("log")
#     ax.set_ylim(1e-1, 1.e2)
#     ax.set_xlim(7.5, 10.)

# axs = axes[1,:]

# iouts = [ [108, 108], [bin_sim.redshift(8), sin_sim.redshift(8)], [sin_sim.redshift(9), sin_sim.redshift(9)] ]
# for ax, iout_lst in zip(axs.flatten(), iouts):
#     data = {"BIN" : load_fesc(bin_sim[iout_lst[0]]), "SIN" : load_fesc(sin_sim[iout_lst[0]])}
#     binned_data = {}
#     cosmo = bin_sim[iout_lst[0]].cosmo

#     for key, ls, lw, c in zip(data.keys(), ["-", "--"], [3., 1.5], ["r", "b"]):
#         log_mvir, fesc, nphotons = data[key]
#         if (iout_lst[0] == 108):
#             cosmo["z"] = 6
#         if (iout_lst[0] == bin_sim.redshift(8)):
#             cosmo["z"] = 8
#         if (iout_lst[0] == bin_sim.redshift(9)):
#             cosmo["z"] = 9
#         # log_mvir, fesc, tint_fesc, nphotons = data[key]
#         # fesc_percent = tint_fesc * 100.
#         log_nphotons_esc = np.log10(nphotons * fesc)
#         bin_centres, mean, std, sterr = fit_scatter(log_mvir, log_nphotons_esc, nbins=nbins, ret_sterr=True)
#         binned_data[key] = (bin_centres, mean, std)
#     #     bin_centres, median = fit_median(log_mvir, fesc_percent, nbins=nbins)
#         ax.scatter(log_mvir, log_nphotons_esc, alpha=0.25, color=c, s=15, marker=".")
#         e = ax.errorbar(bin_centres, mean, yerr=std, color=c, label='%s z=%1.2f' % (key, cosmo['z']), fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=lw)
        
#     print binned_data["BIN"][1] / binned_data["SIN"][1]

# axs[0].set_ylabel(r'log$_{10}$ $\dot{\mathrm{N}}_{\mathrm{ion}}(t)$ f$_{\mathrm{esc}}$ ($t$) [#/s]', fontsize=20)
# for ax in axs.flatten():
#     ax.legend(loc="lower right", frameon=False, prop={"size":16})
#     ax.set_xlabel(r'log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]', fontsize=20)
#     ax.set_xlim(7.5, 10.)
#     ax.set_ylim(43.5, 54)

# plt.tight_layout()
# plt.show()


def load_tint_fesc(snapshot):
#     fname = "%s/pickle/ConsistentTrees/fesc_database_%05i.p" % (snapshot.path, snapshot.ioutput)
    fname = "%s/pickle/ConsistentTrees/time_int_fesc_all_halos_%05i.p" % (snapshot.path, snapshot.ioutput)
    data = pickle.load(open(fname, "rb"))
    
    db = write_fesc_hid_dict.load_db(snapshot.path, snapshot.ioutput)

    mvir = []
    fesc = []
    tint_fesc = []
    # hids = []
    nphotons = []

    for i in range(len(data)):
        res = data[i].result

        ifesc = res["fesc"][0]
        # if ifesc > 1. and ifesc < 10.:
        if ifesc > 1.:
            fesc.append(random.uniform(0.9, 1.0))
        elif ifesc > 0. and ifesc <= 1.:
            fesc.append(ifesc)
        else:
    #         if (res["fesc"] > 10.):
    #             print "%e   %e" % (res["tot_mass"], res["fesc"])
            continue
        mvir.append(res["Mvir"])
        tint_fesc.append(res["tint_fesc_hist"][0])
        hid = int(data[i].idx)
        # hids.append(hid)
        
        Nion_d_now = db[hid]["Nion_d_now"].in_units("s**-1 Msol**-1")
        star_mass = db[hid]["star_mass"].in_units("Msol")
        nphotons.append( (Nion_d_now * star_mass).sum() )

    mvir = np.array(mvir)
    fesc = np.array(fesc)
    tint_fesc = np.array(tint_fesc)
    # hids = np.array(hids)
    nphotons = np.array(nphotons)

#     print count/float(len(mvir))

    ix = np.where(fesc > 0)
    fesc = fesc[ix]
    tint_fesc = tint_fesc[ix]
    mvir = mvir[ix]
    # hids = hids[ix]
    nphotons = nphotons[ix]

    log_mvir = np.log10(mvir)
    log_fesc = np.log10(fesc)

    # ix = np.where(np.logical_and(log_mvir >= 7.5, log_fesc > -3))
    # ix = np.where(log_mvir >= 7.4)
    ix = np.where(np.logical_and(log_mvir >= 6.5, log_mvir <= 8.))
    log_mvir = log_mvir[ix]
    fesc = fesc[ix]
    tint_fesc = tint_fesc[ix]
    # hids = hids[ix]
    nphotons = nphotons[ix]
    
    ix = np.where(~np.isnan(tint_fesc))
    log_mvir = log_mvir[ix]
    fesc = fesc[ix]
    tint_fesc = tint_fesc[ix]
    # hids = hids[ix]
    nphotons = nphotons[ix]
    
    print 'Loaded data for %d halos' % len(log_mvir)
    return log_mvir, fesc, tint_fesc, nphotons

from scipy import interpolate
nbins=5
ax = plt.gca()
cosmo["z"] = 6
data = {"BIN" : load_tint_fesc(bin_snap), "SIN" : load_tint_fesc(sin_snap)}
cols = ["r", "b"]
binned_data = {}
for key, ls, lw, c in zip(data.keys(), ["-", "--"], [3., 1.5], cols):
#     log_mvir, fesc, tint_fesc, hids = data[key]
    log_mvir, fesc, tint_fesc, nphotons = data[key]
    fesc_percent = tint_fesc * 100.

    bin_centres, mean, std = fit_scatter(log_mvir, fesc_percent, nbins=nbins)
    binned_data[key] = (bin_centres, mean, std)
    # bin_centres, median = fit_median(log_mvir, log_fesc, nbins=nbins)
    ax.scatter(log_mvir, fesc_percent, alpha=0.1, color=c, s=5)

    fn = interpolate.interp1d(bin_centres, np.log10(mean), fill_value="extrapolate")
    mass_milky_way = 1e12  # approx halo mass in solar masses
    fesc_milky_way = 10**fn(np.log10(mass_milky_way))
    print key, "fesc Milky Way = %1.2f" % fesc_milky_way

    x = np.linspace(8, 12, 100)
    y = 10**fn(x)
    # ax.plot(x, y, color=c, linestyle=":", linewidth=5)

    # e = plt.errorbar(bin_centres, median, yerr=std, color=c, label='%s z=%1.2f' % ("BPASS_BIN", cosmo['z']),\
    #      fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-', linewidth=2.)

    e = ax.errorbar(bin_centres, mean, yerr=std, color=c, label='%s z=%1.2f' % (key, cosmo['z']), fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle=ls, linewidth=lw)
    
print binned_data["BIN"][1] / binned_data["SIN"][1]

ax.legend(loc="lower right", frameon=False, prop={"size":16})
ax.set_xlabel(r'log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]', fontsize=20)
ax.set_ylabel(r'$\langle$f$\rangle$$_{\mathrm{esc}}$ ($\leq t_{\mathrm{H}}$) [%]', fontsize=20)
ax.set_yscale("log")
# ax.set_xlim(7.5, 10.)
ax.set_ylim(1e0, 1.e2)

plt.show()
