
# coding: utf-8

# In[39]:

import seren3
iout = 108
# sim = seren3.load("RT2")
sin_sim = seren3.load("BPASS_SIN")
bin_sim = seren3.load("BPASS_BIN")

sin_snap = sin_sim[iout]
bin_snap = bin_sim[iout]
cosmo = sin_snap.cosmo


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
        if ifesc > 1. and ifesc <= 10.:
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

    ix = np.where(np.logical_and(log_mvir >= 7., log_fesc > -3))
    log_mvir = log_mvir[ix]
    fesc = fesc[ix]
    nphotons = nphotons[ix]
    
    ix = np.where(~np.isnan(fesc))
    log_mvir = log_mvir[ix]
    fesc = fesc[ix]
    nphotons = nphotons[ix]
    
    print 'Loaded data for %d halos' % len(log_mvir)
    return log_mvir, fesc, None, nphotons

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
        if ifesc > 1. and ifesc <= 10.:
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

    ix = np.where(np.logical_and(log_mvir >= 7., log_fesc > -3))
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

data = {"BIN" : load_tint_fesc(bin_snap), "SIN" : load_tint_fesc(sin_snap)}


# In[83]:

import matplotlib
import matplotlib.pylab as plt
from seren3.analysis.plots import fit_scatter, fit_median

plt.rcParams['figure.figsize'] = (8,8)

nbins=4

cols = ["r", "b"]
binned_data = {}
for key, c in zip(data.keys(), cols):
#     log_mvir, fesc, tint_fesc, hids = data[key]
    log_mvir, fesc, tint_fesc, nphotons = data[key]
    fesc_percent = tint_fesc * 100.
    # fesc_percent = fesc * 100.
    bin_centres, mean, std = fit_scatter(log_mvir, fesc_percent, nbins=nbins)
    binned_data[key] = (bin_centres, mean, std)
#     bin_centres, median = fit_median(log_mvir, fesc_percent, nbins=nbins)
    plt.scatter(log_mvir, fesc_percent, alpha=0.75, color=c, s=5)

#     e = plt.errorbar(bin_centres, median, yerr=std, color=c, label='%s z=%1.2f' % (key, cosmo['z']),\
#          fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-', linewidth=2.)

    e = plt.errorbar(bin_centres, mean, yerr=std, color=c, label='%s z=%1.2f' % (key, cosmo['z']),         fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-', linewidth=2.)
    
print binned_data["BIN"][1] / binned_data["SIN"][1]

plt.legend(prop={"size":16})
plt.xlabel(r'log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]', fontsize=20)
plt.ylabel(r'f$_{\mathrm{esc}}$ (<t$_{\mathrm{H}}$) [%]', fontsize=20)
plt.yscale("log")
plt.ylim(1e-1, 1.5e2)
# plt.savefig("/home/ds381/fesc_bpass_bin_sin_00109.pdf", format="pdf")
# plt.show()
plt.figure()


# In[80]:

# Nphotons escaping plot
data = {"BIN" : load_tint_fesc(bin_snap), "SIN" : load_tint_fesc(sin_snap)}

plt.rcParams['figure.figsize'] = (8,8)

nbins=8

cols = ["r", "b"]
binned_data = {}
for key, c in zip(data.keys(), cols):
#     log_mvir, fesc, tint_fesc, hids = data[key]
    log_mvir, fesc, tint_fesc, nphotons = data[key]
    log_nphotons_esc = np.log10(nphotons * tint_fesc)
    bin_centres, mean, std = fit_scatter(log_mvir, log_nphotons_esc, nbins=nbins)
    binned_data[key] = (bin_centres, mean, std)
    # bin_centres, median = fit_median(log_mvir, log_fesc, nbins=nbins)
    plt.scatter(log_mvir, log_nphotons_esc, alpha=0.75, color=c, s=5)

    # e = plt.errorbar(bin_centres, median, yerr=std, color=c, label='%s z=%1.2f' % ("BPASS_BIN", cosmo['z']),\
    #      fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-', linewidth=2.)

    e = plt.errorbar(bin_centres[1:], mean[1:], yerr=std[1:], color=c, label='%s z=%1.2f' % (key, cosmo['z']),         fmt="o", markerfacecolor=c, mec='k', capsize=2, capthick=2, elinewidth=2, linestyle='-', linewidth=2.)
    
print binned_data["BIN"][1] / binned_data["SIN"][1]

plt.legend(prop={"size":16})
plt.xlabel(r'log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]', fontsize=20)
plt.ylabel(r'log$_{10}$ $\dot{\mathrm{N}}_{\mathrm{ion}}$f$_{\mathrm{esc}}$ (<t$_{\mathrm{H}}$) [#/s]', fontsize=20)
# plt.yscale("log")
# plt.ylim(-1, 2.2)
# plt.savefig("/home/ds381/fesc_bpass_bin_sin_00109.pdf", format="pdf")
plt.show()


# 
