'''
A collection of functions for writing neural network training/prediction data
'''

import numpy as np

def _Vc(snapshot, props):
    '''
    Returns halo circular velocity
    '''

    rvir = snapshot.array(props["rvir"], 'kpc a h**-1').in_units("m")
    mvir = snapshot.array(props["mvir"], 'Msol h**-1').in_units("kg")

    # G = C.G.coeff
    G = snapshot.array(snapshot.C.G)

    Vc = np.sqrt( (G*mvir)/rvir )
    return Vc

def _Tvir(snapshot, props):
    '''
    Returns the virial Temperature of the halo
    '''

    mu = 0.59  # Okamoto 2008
    mH = snapshot.array(snapshot.C.mH)
    kB = snapshot.array(snapshot.C.kB)
    Vc = _Vc(snapshot, props)

    Tvir = 1./2. * (mu*mH/kB) * Vc**2
    return snapshot.array(Tvir, Tvir.units)

def load_training_arrays(snapshot, pickle_path=None, weight="mw"):
    '''
    Load the various arrays we need to write the training data
    '''

    import pickle

    if (pickle_path is None):
        pickle_path = "%s/pickle/" % snapshot.path

    # Define some functions for loading the pickle dictionarys
    def _get_halo_av_fname(qty, iout, pickle_path):
        return "%s/%s_halo_av_%05i.p" % (pickle_path, qty, iout)

    def _get_fbaryon_fname(iout, pickle_path):
        return "%s/ConsistentTrees/fbaryon_tdyn_%05i.p" % (pickle_path, iout)

    def _load_pickle_dict(fname):
        return pickle.load(open(fname, "rb"))

    T_fname = _get_halo_av_fname("T", snapshot.ioutput, pickle_path)
    xHII_fname = _get_halo_av_fname("xHII", snapshot.ioutput, pickle_path)
    fb_fname = _get_fbaryon_fname(snapshot.ioutput, pickle_path)

    T_data = _load_pickle_dict(T_fname)
    xHII_data = _load_pickle_dict(xHII_fname)
    fb_data = _load_pickle_dict(fb_fname)

    # Make table with keys as halo ids
    fb_data_table = {}
    for i in range(len(fb_data)):
        di = fb_data[i]
        fb_data_table[int(di.idx)] = di.result

    # Add T and xHII data
    assert len(xHII_data) == len(T_data), "xHII and T databases have different lenghs"
    for i in range(len(xHII_data)):
        fb_data_table[int(xHII_data[i].idx)]["xHII"] = xHII_data[i].result[weight]
        fb_data_table[int(T_data[i].idx)]["T"] = T_data[i].result[weight]
        fb_data_table[int(T_data[i].idx)]["Tvir"] = _Tvir(snapshot, fb_data_table[int(T_data[i].idx)]["hprops"])

    mvir = np.zeros(len(fb_data_table)); fb = np.zeros(len(fb_data_table))
    ftidal = np.zeros(len(fb_data_table)); xHII = np.zeros(len(fb_data_table))
    pid = np.zeros(len(fb_data_table));T = np.zeros(len(fb_data_table))
    np_dm = np.zeros(len(fb_data_table)); np_cell = np.zeros(len(fb_data_table))
    time_since_MM = np.zeros(len(fb_data_table))

    keys = fb_data_table.keys()
    for i in range(len(fb_data_table)):
        res = fb_data_table[keys[i]]
        mvir[i] = res["tot_mass"]
        fb[i] = res["fb"]
        ftidal[i] = res["hprops"]["tidal_force_tdyn"]
        xHII[i] = res["xHII"]
        # print "%e %e %e" % (mvir[i], res["Tvir"], res["T"])
        # T[i] = ((res["T"])/(res["Tvir"]))
        T[i] = np.log10(res["T"]/res["Tvir"])
        pid[i] = res["pid"]
        np_dm[i] = res["np_dm"]
        np_cell = res["ncell"]
        time_since_MM[i] = res["time_since_last_MM"]

    log_mvir = np.log10(mvir)

    idx = np.where(np.logical_and(pid == -1, np.logical_and(np_dm >= 20, np_cell >= 50)))
    # idx = np.where(np.logical_and(pid == -1, np.logical_and(np_dm >= 20, np_cell >= 20)))
    # idx = np.where(np.logical_and(pid == -1, np_dm >= 20))
    # idx = np.where(pid == -1)

    print len(log_mvir)
    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]
    print len(log_mvir)

    idx = np.where(np.logical_or(~np.isnan(xHII), ~np.isnan(T)))

    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]
    print len(log_mvir)

    idx = np.where(xHII > 0.05)

    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]
    print len(log_mvir)

    cosmo = snapshot.cosmo
    fb_cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    fb /= fb_cosmic_mean  # cosmic mean units

    return log_mvir, fb, ftidal, xHII, T#, time_since_MM

# Function to parse topology
def _list_to_string(lst):
    return str(lst)[1:-1].replace(',', '')

# Define scaling functions
def _scale_z(zmax, zmin, z):
    scaled = (z - zmin) / (zmax - zmin)
    scaled -= 0.5  # [-0.5, 0.5]
    scaled *= 2.  # [-1, 1]
    return scaled

def _scale(arr):
    # scaled = (arr - arr.min()) / (arr.max() - arr.min())  # [0, 1]
    # scaled -= 0.5  # [-0.5, 0.5]
    # scaled *= 2.  # [-1, 1]
    # return scaled
    return arr/arr.max()

def _scale2(arr):
    scaled = (arr - arr.min()) / (arr.max() - arr.min())  # [0, 1]
    scaled -= 0.5  # [-0.5, 0.5]
    scaled *= 2.  # [-1, 1]
    return scaled

def write_training_data(snapshot, log_mvir, fb, ftidal, xHII, T, ntrain, topology, out_path, weight):
    '''
    Writes the training data for our neural network
    '''
    import random

    fname = "%s/fb_%05i_%s.train" % (out_path, snapshot.ioutput, weight)

    # Scale the data to our sigmoid range
    log_mvir_scaled = _scale(log_mvir)
    fb_scaled = _scale2(fb)
    # ftidal_scaled = _scale(ftidal)
    # xHII_scaled = _scale(xHII)
    # T_scaled = _scale(T)

    # log_mvir_scaled = log_mvir
    ftidal_scaled = ftidal
    xHII_scaled = xHII
    T_scaled = T

    topology_string = "topology: %s\n" % _list_to_string(topology)

    # Write the file
    with open(fname, "w") as f:
        f.write(topology_string)
        for j in range(ntrain):
            ix = range(len(fb))
            random.shuffle(ix)
            for i in ix:
                l1 = "in: "
                for ii in [log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i]]:
                    l1 += "%f " % ii
                l1 += "\n"
                f.write(l1)

                l2 = "out: %f\n" % fb_scaled[i]
                f.write(l2)

    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled,\
             xHII_scaled, T_scaled, out_path, weight, raw_input_format=True)

    return log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled

def make_all_prediction_files(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled, out_path, weight):
    '''
    Makes all .predict files for our panel plot
    '''

    ftidal_predict = np.ones(len(ftidal_scaled)) * ftidal_scaled.min()
    # TODO - is this the correct way to get mean xHII?
    xHII_predict = np.array( [max(xi, 0.5) for xi in xHII_scaled] )

    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled, out_path, weight)
    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_predict, xHII_scaled, T_scaled, out_path, weight, label="zero_ftidal")
    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_predict, T_scaled, out_path, weight, label="mean_xHII")
    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_predict, xHII_predict, T_scaled, out_path, weight, label="zero_ftidal_mean_xHII")

def write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled, out_path, weight, label=None, **kwargs):
    '''
    Writes the inputs used to get predictions from the neural network
    '''

    raw_input_format = kwargs.pop("raw_input_format", False)

    fname = "%s/fb_%05i_%s" % (out_path, snapshot.ioutput, weight)
    if (raw_input_format):
        fname += ".input"
    elif (label is not None):
        fname += "_%s.predict" % label
    else:
        fname += ".predict"

    print fname

    # Write the file
    with open(fname, "w") as f:
        for i in range(len(log_mvir_scaled)):
            l1 = ""
            if (raw_input_format is False):
                l1 = "in: "
            for ii in [log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i]]:
                l1 += "%f " % ii

            if (raw_input_format):
                l1 += "%f" % fb_scaled[i]
            l1 += "\n"
            f.write(l1)

    return fname


def plot_fb_panels(snapshot, log_mvir, fb, ftidal, xHII, T, weight="M", **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.analysis.plots import fit_scatter

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0)

    fig.suptitle('z = %1.2f' % snapshot.z, fontsize=16)

    mvir = 10**log_mvir

    log_xHI = np.log10(1. - xHII)
    log_ftidal = np.log10(ftidal)

    bc, mean, std, sterr = fit_scatter(log_mvir, fb, nbins=15, ret_sterr=True)

    def _plot(mvir, fb, carr, ax, **kwargs):
        ax.errorbar(10**bc, mean, yerr=std, linewidth=1.5, color="k", linestyle="--")
        return ax.scatter(mvir, fb, c=carr, **kwargs)

    labels = [r"log$_{10} \langle F_{\mathrm{tidal}} \rangle_{t_{\mathrm{dyn}}}$",\
             r"$\langle x_{\mathrm{HII}} \rangle_{\mathrm{%s}}$" % weight,\
             # r"log$_{10} \langle x_{\mathrm{HI}} \rangle_{%s}$" % weight,\
             r"log$_{10} \langle T \rangle_{\mathrm{%s}}$/$T_{\mathrm{vir}}$" % weight]
    for ax, carr, lab in zip(axs.flatten(), [log_ftidal, xHII, T], labels):
        sp = _plot(mvir, fb, carr, ax, **kwargs)
        cbar = fig.colorbar(sp, ax=ax)
        cbar.set_label(lab)
        ax.set_xlim(1e7, 1.5e10)

        # ax.set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
        ax.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
        ax.set_xscale("log")

    axs[-1].set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
    
    plt.show()

def plot_fb_panels2(sim, iouts, pickle_path, weight="mw", weight_label="M", **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.analysis.plots import fit_scatter
    from seren3.utils import plot_utils

    cmap = plot_utils.load_custom_cmaps("parula")
    kwargs["cmap"] = cmap

    fig, axs = plt.subplots(nrows=3, ncols=len(iouts), sharex=True, sharey=True, figsize=(12,12))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0)

    for i in range(len(iouts)):
        iout = iouts[i]
        snapshot = sim[iout]
        log_mvir, fb, ftidal, xHII, T = load_training_arrays(snapshot, pickle_path=pickle_path, weight=weight)

        axs[0,i].set_title("z = %1.2f" % snapshot.z)

        mvir = 10**log_mvir

        log_xHI = np.log10(1. - xHII)
        log_ftidal = np.log10(ftidal)

        bc, mean, std, sterr = fit_scatter(log_mvir, fb, nbins=15, ret_sterr=True)

        def _plot(mvir, fb, carr, ax, z, **kwargs):
            ax.errorbar(10**bc, mean, yerr=std, linewidth=1.5, color="k", linestyle="--")
            return ax.scatter(mvir, fb, c=carr, label="z = %1.2f" % z, **kwargs)

        labels = [r"log$_{10} \langle F_{\mathrm{tidal}} \rangle_{t_{\mathrm{dyn}}}$",\
                 r"$\langle x_{\mathrm{HII}} \rangle_{\mathrm{%s}}$" % weight_label,\
                 # r"log$_{10} \langle x_{\mathrm{HI}} \rangle_{%s}$" % weight,\
                 r"log$_{10} \langle T \rangle_{\mathrm{%s}}$/$T_{\mathrm{vir}}$" % weight_label]
        for ax, carr, lab in zip(axs[:,i].flatten(), [log_ftidal, xHII, T], labels):
            sp = _plot(mvir, fb, carr, ax, snapshot.z, **kwargs)
            cbar = fig.colorbar(sp, ax=ax)

            if (i == len(iouts)-1):
                cbar.set_label(lab)
            ax.set_xlim(5e6, 1.5e10)

            # ax.set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
            # ax.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
        ax.set_xscale("log")

        axs[-1,i].set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")

    for ax in axs[:,0]:
        ax.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")

    # plt.tight_layout()
    plt.show()