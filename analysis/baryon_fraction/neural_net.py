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

def load_training_arrays(snapshot, pickle_path=None):
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
        fb_data_table[int(xHII_data[i].idx)]["xHII"] = xHII_data[i].result["vw"]
        fb_data_table[int(T_data[i].idx)]["T"] = T_data[i].result["vw"]
        fb_data_table[int(T_data[i].idx)]["Tvir"] = _Tvir(snapshot, fb_data_table[int(T_data[i].idx)]["hprops"])

    mvir = np.zeros(len(fb_data_table)); fb = np.zeros(len(fb_data_table))
    ftidal = np.zeros(len(fb_data_table)); xHII = np.zeros(len(fb_data_table))
    pid = np.zeros(len(fb_data_table));T = np.zeros(len(fb_data_table))
    np_dm = np.zeros(len(fb_data_table))#; np_cell = np.zeros(len(fb_data_table))
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
        time_since_MM[i] = res["time_since_last_MM"]

    log_mvir = np.log10(mvir)

    idx = np.where(np.logical_and(pid == -1, np_dm >= 20))

    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]

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
    scaled = (arr - arr.min()) / (arr.max() - arr.min())  # [0, 1]
    scaled -= 0.5  # [-0.5, 0.5]
    scaled *= 2.  # [-1, 1]
    return scaled

def write_training_data(snapshot, log_mvir, fb, ftidal, xHII, T, topology=[4, 14, 1], out_path=None, **kwargs):
    '''
    Write the training data file
    '''
    import os, random

    niter = kwargs.pop("niter", 1)

    if (out_path is None):
        out_path = snapshot.path

    net_dir = "%s/neural-net/%d/" % (out_path, snapshot.ioutput)
    if (os.path.isdir(net_dir) is False):
        os.mkdir(net_dir)

    # Scale between [-1, 1] to fit within neural-net transfer function
    log_mvir_scaled = _scale(log_mvir); fb_scaled = _scale(fb)
    ftidal_scaled = _scale(ftidal); xHII_scaled = _scale(xHII)
    T_scaled = _scale(T)

    ## Scaled redshift
    # sim = seren3.init(snapshot.path)
    # redshifts = sim.redshifts
    # zmin = min(redshifts)
    # zmax = max(redshifts)

    # z_scaled = _scale_z(zmax, zmin, snapshot.z)

    # Write the file
    print "Writing training data..."
    top_line = "topology: %s\n" % (_list_to_string(topology))
    f = open('%s/fb_neural_net_%05i_training_data.txt' % (net_dir, snapshot.ioutput), 'w')
    f.write(top_line)

    # niter = 100
    for j in range(niter):
        ix = range(len(fb_scaled))
        random.shuffle(ix)
        for i in ix:
            # l1 = "in: %f %f %f %f %f\n" % (log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i], z_scaled)
            l1 = "in: %f %f %f %f\n" % (log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i])
            l2 = "out: %f\n" % fb_scaled[i]
            f.write(l1)
            f.write(l2)

    f.close()

    # if kwargs.pop("write_input", False):
    write_neural_net_input_data(snapshot, net_dir, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled)

def write_neural_net_input_data(snapshot, out_path, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled):
    '''
    Writes a file with the scaled input data, for plotting
    '''
    with open("%s/fb_neural_net_%05i_input_data.txt" % (out_path, snapshot.ioutput), "w") as f:

        for i in range(len(log_mvir_scaled)):
            # l = "%f %f %f %f %f %f" % (log_mvir_scaled[i], ftidal_scaled[i], \
            #     xHII_scaled[i], T_scaled[i], z_scaled, fb_scaled[i])
            l = "%f %f %f %f %f" % (log_mvir_scaled[i], ftidal_scaled[i], \
                xHII_scaled[i], T_scaled[i], fb_scaled[i])
            f.write(l)

            if (i < len(log_mvir_scaled) - 1):
                f.write("\n")

def write_prediction_file(snapshot, log_mvir, fb, ftidal, xHII, T, out_path=None, **kwargs):
    '''
    Write the prediction data file
    '''
    import os

    if (out_path is None):
        out_path = snapshot.path

    net_dir = "%s/neural-net/%d/" % (out_path, snapshot.ioutput)
    if (os.path.isdir(net_dir) is False):
        os.mkdir(net_dir)

    zero_ftidal = kwargs.pop("zero_ftidal", False)
    use_halo_xHII = kwargs.pop("use_halo_xHII", True)

    # Scale between [-1, 1] to fit within neural-net transfer function
    log_mvir_scaled = _scale(log_mvir); fb_scaled = _scale(fb)
    ftidal_scaled = _scale(ftidal); xHII_scaled = _scale(xHII)
    T_scaled = _scale(T)

    if (use_halo_xHII is False):
        import pickle

        pickle_path = kwargs.pop("pickle_path", None)

        if (pickle_path is None):
            pickle_path = "%s/pickle/" % snapshot.path

        xHII_hist = pickle.load( open("%s/xHII_reion_history.p" % pickle_path, "rb") )
        xHII_table = {}

        for i in range(len(xHII_hist)):
            d = xHII_hist[i]
            xHII_table[int(d.idx)] = d.result["volume_weighted"]

        # val = np.log10(1. - xHII_table[snapshot.ioutput])
        val = xHII_table[snapshot.ioutput]
        # val = xHII.mean()
        print val
        xHII_global_scaled = _scale_z(xHII.max(), xHII.min(), val)
        xHII_scaled = np.ones(len(xHII)) * xHII_global_scaled

    fname = '%s/fb_neural_net_%05i_prediction_data' % (net_dir, snapshot.ioutput)
    if (zero_ftidal):
        fname = '%s_zeroftidal' % (fname)

    if (use_halo_xHII is False):
        fname = "%s_globalxHII" % fname 

    f = open("%s.txt" % fname, "w")
    print fname

    for i in range(len(fb_scaled)):
        l1 = None
        if (zero_ftidal):
            # l1 = "in: %f %f %f %f %f" % (log_mvir_scaled[i], -1., xHII_scaled[i], T_scaled[i], z_scaled)
            l1 = "in: %f %f %f %f" % (log_mvir_scaled[i], -1., xHII_scaled[i], T_scaled[i])
        else:
            l1 = "in: %f %f %f %f" % (log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i])
        f.write(l1)

        if (i < len(fb_scaled - 1)):
            f.write("\n")

    f.close()

def plot_fb_panels(snapshot, log_mvir, fb, ftidal, xHII, T, **kwargs):
    import numpy as np
    import matplotlib.pylab as plt

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0)

    fig.suptitle('z = %1.2f' % snapshot.z, fontsize=16)

    mvir = 10**log_mvir

    log_xHI = np.log10(1. - xHII)
    log_ftidal = np.log10(ftidal)

    def _plot(mvir, fb, carr, ax, **kwargs):
        return ax.scatter(mvir, fb, c=carr, **kwargs)

    labels = [r"log$_{10} \langle F_{\mathrm{tidal}} \rangle_{tdyn}$",\
             # r"$\langle x_{\mathrm{HII}} \rangle_{V}$",\
             r"log$_{10} \langle x_{\mathrm{HI}} \rangle_{V}$",\
             r"log$_{10} \langle T \rangle_{V}$/$T_{\mathrm{vir}}$"]
    for ax, carr, lab in zip(axs.flatten(), [log_ftidal, log_xHI, T], labels):
        sp = _plot(mvir, fb, carr, ax, **kwargs)
        cbar = fig.colorbar(sp, ax=ax)
        cbar.set_label(lab)

        # ax.set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
        ax.set_ylabel(r"f$_{\mathrm{b}}$[$\Omega_{\mathrm{b}}$/$\Omega_{\mathrm{M}}$]")
        ax.set_xscale("log")

    axs[-1].set_xlabel(r"log$_{10}$ M$_{\mathrm{vir}}$ [M$_{\odot}$/h]")
    
    plt.show()