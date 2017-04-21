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

def dump_fbaryon_training_set(snap, fname, out_fname, topology=[2,6,1], out_zero_ftydn=False, n=1):
    import pickle
    import numpy as np

    data = pickle.load(open(fname, 'rb'))
    mvir = np.zeros(len(data))
    fb = np.zeros(len(data))
    ftdyn = np.zeros(len(data))
    pid = np.zeros(len(data))

    def _list_to_string(lst):
        return str(lst)[1:-1].replace(',', '')

    for i in range(len(data)):
        res = data[i].result
        mvir[i] = res["tot_mass"]
        fb[i] = res["fb"]
        pid[i] = res["pid"]
        ftdyn[i] = res["hprops"]["tidal_force_tdyn"]

    cosmo = snap.cosmo
    fb_cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    fb /= fb_cosmic_mean

    idx = np.where(np.logical_and(pid == -1, fb > 0.))
    # fb[fb==0.] += 1e-6
    mvir = mvir[idx]; fb = fb[idx]; ftdyn = ftdyn[idx]; pid = pid[idx]
    fb = np.log10(fb)

    log_mvir = np.log10(mvir)

    fb_scaled = (fb - fb.min()) / (fb.max() - fb.min())
    log_mvir_scaled = (log_mvir - log_mvir.min()) / (log_mvir.max() - log_mvir.min())
    ftdyn_scaled = (ftdyn - ftdyn.min()) / (ftdyn.max() - ftdyn.min())

    if out_zero_ftydn:
        ftdyn_scaled = np.zeros(len(ftdyn_scaled))

    top_line = "topology: %s\n" % (_list_to_string(topology))

    with open(out_fname, 'w') as f:
        f.write(top_line)

        for i in range(n):
            for i in range(len(mvir)):
                l1 = "in: %f %f\n" % (log_mvir_scaled[i], ftdyn_scaled[i])
                l2 = "out: %f\n" % fb_scaled[i]
                f.write(l1)
                f.write(l2)

    return mvir, 10**fb, ftdyn

def load_neural_net_results(fname):
    import numpy as np

    lines = None
    with open(fname, 'r') as f:
        lines = f.readlines()

    inp_vals = []
    out_vals = []

    for l in lines:
        if l.startswith("Inputs"):
            li = l.split()
            inp_vals.append([float(li[1]), float(li[2])])
        elif l.startswith("Outputs"):
            out_vals.append(float(l.split()[1]))

    return np.array(inp_vals), np.array(out_vals)

def rescale_neural_network_vals(inp_vals, out_vals, mvir, fb, ftdyn):
    import numpy as np

    log_mvir = np.log10(mvir)

    def _unscale(arr, ref):
        return (arr * (ref.max() - ref.min())) + ref.min()

    fb_out_unscaled = _unscale(out_vals, np.log10(fb))
    log_mvir_out_unscaled = _unscale(inp_vals[:,0], log_mvir)
    ftdyn_out_unscaled = _unscale(inp_vals[:,1], ftdyn)

    return 10**log_mvir_out_unscaled, 10**fb_out_unscaled, ftdyn_out_unscaled
