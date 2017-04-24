def compute_fb(context, mass_unit="Msol h**-1"):
    '''
    Computes the baryon fraction for this container
    '''
    import numpy as np
    
    part_dset = context.p[["id", "mass", "epoch"]].flatten()
    ix_dm = np.where(np.logical_and( part_dset["id"] > 0., part_dset["epoch"] == 0 ))  # index of dm particles
    ix_stars = np.where( np.logical_and( part_dset["id"] > 0., part_dset["epoch"] != 0 ) )  # index of star particles

    gas_dset = context.g["mass"].flatten()

    part_mass_tot = part_dset["mass"].in_units(mass_unit).sum()
    star_mass_tot = part_dset["mass"].in_units(mass_unit)[ix_stars].sum()
    gas_mass_tot = gas_dset["mass"].in_units(mass_unit).sum()

    tot_mass = part_mass_tot + gas_mass_tot
    fb = (gas_mass_tot + star_mass_tot)/tot_mass

    return fb, tot_mass

############################# NEURAL NET #############################

def dump_fb_training_data(snapshot, topology, niter, pickle_path=None, out_path=None, zero_ftidal=False, use_halo_xHII=True):
    '''
    Dump data to disk for training our neural network
    '''
    import seren3
    import numpy as np
    import pickle, os

    if (pickle_path is None):
        pickle_path = "%s/pickle/" % snapshot.path

    def _list_to_string(lst):
        return str(lst)[1:-1].replace(',', '')

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

    mvir = np.zeros(len(fb_data_table)); fb = np.zeros(len(fb_data_table))
    ftidal = np.zeros(len(fb_data_table)); xHII = np.zeros(len(fb_data_table))
    T = np.zeros(len(fb_data_table)); pid = np.zeros(len(fb_data_table))
    np_dm = np.zeros(len(fb_data_table))

    keys = fb_data_table.keys()
    for i in range(len(fb_data_table)):
        res = fb_data_table[keys[i]]
        mvir[i] = res["tot_mass"]
        fb[i] = res["fb"]
        ftidal[i] = res["hprops"]["tidal_force_tdyn"]
        xHII[i] = res["xHII"]
        T[i] = res["T"]
        pid[i] = res["pid"]
        np_dm[i] = res["np_dm"]

    log_mvir = np.log10(mvir)
    idx = np.where(np.logical_and(pid == -1, np_dm >= 20))

    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]

    cosmo = snapshot.cosmo
    fb_cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    fb /= fb_cosmic_mean

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

    # Scale between [-1, 1] to fit within neural-net transfer function
    log_mvir_scaled = _scale(log_mvir); fb_scaled = _scale(fb)
    ftidal_scaled = _scale(ftidal); xHII_scaled = _scale(xHII)
    T_scaled = _scale(T)

    if (out_path is None):
        out_path = snapshot.path

    net_dir = "%s/neural-net/" % out_path
    if (os.path.isdir(net_dir) is False):
        os.mkdir(net_dir)

    sim = seren3.init(snapshot.path)
    redshifts = sim.redshifts
    zmin = min(redshifts)
    zmax = max(redshifts)

    z_scaled = _scale_z(zmax, zmin, snapshot.z)

    # Write the files
    top_line = "topology: %s\n" % (_list_to_string(topology))
    f = open('%s/fb_neural_net_%05i_training_data.txt' % (net_dir, snapshot.ioutput), 'w')
    f.write(top_line)

    # niter = 100
    for j in range(niter):
        for i in range(len(fb_scaled)):
            l1 = "in: %f %f %f %f %f\n" % (log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i], z_scaled)
            l2 = "out: %f\n" % fb_scaled[i]
            f.write(l1)
            f.write(l2)

    f.close()

    if (use_halo_xHII is False):
        xHII_hist = pickle.load( open("%s/xHII_reion_history.p" % pickle_path, "rb") )
        xHII_table = {}

        for i in range(len(xHII_hist)):
            d = xHII_hist[i]
            xHII_table[int(d.idx)] = d.result["volume_weighted"]

        print xHII_table[snapshot.ioutput]
        xHII_global_scaled = _scale_z(xHII.max(), xHII.min(), xHII_table[snapshot.ioutput])
        xHII_scaled = np.ones(len(xHII)) * xHII_global_scaled

    fname = '%s/fb_neural_net_%05i_prediction_data' % (net_dir, snapshot.ioutput)
    if (zero_ftidal):
        fname = '%s_zeroftidal' % (fname)

    if (use_halo_xHII is False):
        fname = "%s_globalxHII" % fname 

    f = open("%s.txt" % fname, "w")

    for i in range(len(fb_scaled)):
        l1 = None
        if (zero_ftidal):
            l1 = "in: %f %f %f %f %f" % (log_mvir_scaled[i], -1., xHII_scaled[i], T_scaled[i], z_scaled)
        else:
            l1 = "in: %f %f %f %f %f" % (log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i], z_scaled)
        f.write(l1)

        if (i < len(fb_scaled - 1)):
            f.write("\n")

    write_neural_net_input_data(snapshot, net_dir, z_scaled, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled)

    return log_mvir, fb, ftidal, xHII, T

def write_neural_net_input_data(snapshot, out_path, z_scaled, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled):

    with open("%s/fb_neural_net_%05i_input_data.txt" % (out_path, snapshot.ioutput), "w") as f:

        for i in range(len(log_mvir_scaled)):
            l = "%f %f %f %f %f %f" % (log_mvir_scaled[i], ftidal_scaled[i], \
                xHII_scaled[i], T_scaled[i], z_scaled, fb_scaled[i])
            f.write(l)

            if (i < len(log_mvir_scaled) - 1):
                f.write("\n")
    