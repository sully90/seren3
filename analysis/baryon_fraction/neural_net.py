'''
A collection of functions for writing neural network training/prediction data
'''

import numpy as np

_MVIR_MIN=np.log10(1e7)
_MVIR_MAX=np.log10(5e10)

_FB_MIN=0.
_FB_MAX=.5

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

    _MASS_CUTOFF = 1e7

    # Define some functions for loading the pickle dictionarys
    def _get_fbaryon_fname(iout, pickle_path):
        return "%s/ConsistentTrees/fbaryon_tdyn_%05i.p" % (pickle_path, iout)

    def _load_pickle_dict(fname):
        return pickle.load(open(fname, "rb"))

    fb_fname = _get_fbaryon_fname(snapshot.ioutput, pickle_path)
    fb_data = _load_pickle_dict(fb_fname)

    nrecords = len(fb_data)
    mvir = np.zeros(nrecords); fb = np.zeros(nrecords)
    ftidal = np.zeros(nrecords); xHII = np.zeros(nrecords)
    pid = np.zeros(nrecords);T = np.zeros(nrecords)
    np_dm = np.zeros(nrecords); np_cell = np.zeros(nrecords)
    time_since_MM = np.zeros(nrecords)

    for i in range(nrecords):
        res = fb_data[i].result
        Tvir = _Tvir(snapshot, res["hprops"])

        mvir[i] = res["tot_mass"]
        fb[i] = res["fb"]
        ftidal[i] = res["hprops"]["tidal_force_tdyn"]
        pid[i] = res["pid"]
        np_dm[i] = res["np_dm"]
        np_cell[i] = res["ncell"]
        time_since_MM[i] = res["time_since_last_MM"]

        xHII[i] = res["xHII_mw"]
        T[i] = np.log10(res["T_mw"]/Tvir)

    log_mvir = np.log10(mvir)

    # idx = np.where(np.logical_and(pid == -1, np.logical_and(np_dm >= 20, np_cell >= 50)))
    # idx = np.where(np.logical_and(pid == -1, np.logical_and(np_dm >= 50, np_cell >= 50)))
    # idx = np.where(np.logical_and(pid == -1, np_dm >= 20))
    # idx = np.where(pid == -1)

    # idx = np.where(np.logical_and(np.logical_and(pid == -1, mvir >= _MASS_CUTOFF), np.logical_and(np_dm >= 50, np_cell >= 50)))
    idx = np.where(np.logical_and(mvir >= _MASS_CUTOFF, np.logical_and(np_dm >= 50, np_cell >= 50)))

    # print len(log_mvir)
    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]
    # print len(log_mvir)

    idx = np.where(np.logical_or(~np.isnan(xHII), ~np.isnan(T)))

    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]
    # print len(log_mvir)

    idx = np.where(xHII > 0.05)
    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]
    # print len(log_mvir)

    idx = np.where(ftidal > 0.)
    log_mvir = log_mvir[idx]; fb = fb[idx]; ftidal = ftidal[idx]
    xHII = xHII[idx]; T = T[idx]; time_since_MM = time_since_MM[idx]
    # print len(log_mvir)

    cosmo = snapshot.cosmo
    fb_cosmic_mean = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    # fb /= fb_cosmic_mean  # cosmic mean units

    return log_mvir, fb, ftidal, xHII, T#, time_since_MM


# Function to parse topology
def _list_to_string(lst):
    return str(lst)[1:-1].replace(',', '')

# Define scaling functions
# def _scale_z(zmax, zmin, z):
#     scaled = (z - zmin) / (zmax - zmin)
#     scaled -= 0.5  # [-0.5, 0.5]
#     scaled *= 2.  # [-1, 1]
#     return scaled

# def _scale(arr):
#     # scaled = (arr - arr.min()) / (arr.max() - arr.min())  # [0, 1]
#     # scaled -= 0.5  # [-0.5, 0.5]
#     # scaled *= 2.  # [-1, 1]
#     # return scaled
#     return arr/arr.max()

# def _scale2(arr):
#     scaled = (arr - arr.min()) / (arr.max() - arr.min())  # [0, 1]
#     scaled -= 0.5  # [-0.5, 0.5]
#     scaled *= 2.  # [-1, 1]
#     return scaled

def _scale_mvir(arr):
    scaled = (arr - _MVIR_MIN) / (_MVIR_MAX - _MVIR_MIN)  # [0, 1]
    scaled -= 0.5  # [-0.5, 0.5]
    scaled *= 2.  # [-1, 1]
    return scaled

def _scale_fb(arr):
    scaled = (arr - _FB_MIN) / (_FB_MAX - _FB_MIN)  # [0, 1]
    scaled -= 0.5  # [-0.5, 0.5]
    scaled *= 2.  # [-1, 1]
    return scaled

def _scale_ftidal(arr):
    return np.log10(arr)
    # return _scale_mvir(arr, mvir_min = 0., mvir_max = 5)
    # return arr

def _scale_xHII(arr):
    # return np.log10(arr)
    # return _scale_mvir(arr, mvir_min = 0., mvir_max = 5)
    return arr

def write_training_data(snapshot, log_mvir, fb, ftidal, xHII, T, ntrain, topology, out_path, weight):
    '''
    Writes the training data for our neural network
    '''
    import random

    # Do all scaling here

    fname = "%s/fb_%05i_%s.train" % (out_path, snapshot.ioutput, weight)

    # Scale the data to our sigmoid range
    log_mvir_scaled = _scale_mvir(log_mvir)
    fb_scaled = _scale_fb(fb)
    # ftidal_scaled = _scale(ftidal)
    # xHII_scaled = _scale(xHII)
    # T_scaled = _scale(T)

    # log_mvir_scaled = log_mvir
    # ftidal_scaled = ftidal
    ftidal_scaled = _scale_ftidal(ftidal)
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


def make_all_prediction_files(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled, out_path, weight):
    '''
    Makes all .predict files for our panel plot
    '''

    ftidal_predict = np.ones(len(ftidal_scaled)) * np.log10((10**ftidal_scaled).min())
    # ftidal_predict = np.zeros(len(ftidal_scaled))
    # TODO - is this the correct way to get mean xHII?
    xHII_predict = np.array( [max(xi, 0.5) for xi in xHII_scaled] )
    # xHII_predict = np.ones(len(xHII_scaled)) * np.log10(0.5)
    # xHII_predict = np.array( [max(xi, np.log10(0.5)) for xi in xHII_scaled] )

    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled, out_path, weight)
    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_predict, xHII_scaled, T_scaled, out_path, weight, label="zero_ftidal")
    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_predict, T_scaled, out_path, weight, label="mean_xHII")
    write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_predict, xHII_predict, T_scaled, out_path, weight, label="zero_ftidal_mean_xHII")

###########################################################################################

def random_write_many(sim, ioutputs, pickle_path, niter, fname, topology = [4, 25, 1]):
    '''
    Write many outputs to one file randomly
    '''
    import random

    # This worked well for 106_70.train
    topology_string = "topology: %s\n" % _list_to_string(topology)

    # Load all training data
    data = []
    for ioutput in ioutputs:
        print ioutput
        snapshot = sim[ioutput]

        log_mvir, fb, ftidal, xHII, T = load_training_arrays(snapshot, \
                pickle_path=pickle_path, weight="mw")

        log_mvir_scaled = _scale_mvir(log_mvir)
        fb_scaled = _scale_fb(fb)
        # ftidal_scaled = _scale(ftidal)
        # xHII_scaled = _scale(xHII)
        # T_scaled = _scale(T)

        # log_mvir_scaled = log_mvir
        # ftidal_scaled = ftidal
        ftidal_scaled = _scale_ftidal(ftidal)
        xHII_scaled = xHII
        T_scaled = T

        for i in range(len(log_mvir_scaled)):
            data.append((log_mvir_scaled[i], fb_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i]))

    with open(fname, "w") as f:
        f.write(topology_string)
        # Write to file
        for j in range(niter):
            ix = range(len(data))
            random.shuffle(ix)
            for i in ix:
                log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled = data[i]
                l1 = "in: "
                for ii in [log_mvir_scaled, ftidal_scaled, xHII_scaled, T_scaled]:
                    l1 += "%f " % ii
                l1 += "\n"
                f.write(l1)

                l2 = "out: %f\n" % fb_scaled
                f.write(l2)

    # write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled,\
    #          xHII_scaled, T_scaled, out_path, weight, raw_input_format=True)

def random_write_many2(sim, ioutputs, pickle_path, niter, fname, topology=[4, 25, 1], the_mass_bins=[7., 8., 9., 10.]):
    '''
    Write many outputs to one file randomly
    '''
    import random

    # This worked well for 106_70.train
    topology_string = "topology: %s\n" % _list_to_string(topology)

    # Load all training data
    data = []
    for ioutput in ioutputs:
        snapshot = sim[ioutput]

        log_mvir, fb, ftidal, xHII, T = load_training_arrays(snapshot, \
                pickle_path=pickle_path, weight="mw")

        log_mvir_scaled = _scale_mvir(log_mvir)
        fb_scaled = _scale_fb(fb)
        # xHII_scaled = _scale(xHII)
        # T_scaled = _scale(T)

        # log_mvir_scaled = log_mvir
        ftidal_scaled = _scale_ftidal(ftidal)
        xHII_scaled = xHII
        T_scaled = T

        mass_bins = np.digitize(log_mvir, the_mass_bins, right=True)

        nmin = len(np.where(mass_bins == mass_bins.min())[0])
        # print nmin
        # Evenly add data in bins of mvir
        for i in range(mass_bins.min(), len(the_mass_bins)+1):
            idx = np.where(mass_bins == i)
            nbin = len(idx[0])
            if nbin == 0:
                continue
            npass = int(np.round(nmin / nbin) * max(1, np.round(float(i)/2.)))
            print nmin, nbin, npass, max(1, np.round(float(i)/2.)), i
            for j in range(npass):
                for ix in idx[0]:
                    data.append((log_mvir_scaled[ix], fb_scaled[ix], ftidal_scaled[ix], xHII_scaled[ix], T_scaled[ix]))

    print "Writing %i entries" % (len(data))

    with open(fname, "w") as f:
        f.write(topology_string)
        # Write to file
        for j in range(niter):
            ix = range(len(data))
            random.shuffle(ix)
            for i in ix:
                log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled = data[i]
                l1 = "in: "
                for ii in [log_mvir_scaled, ftidal_scaled, xHII_scaled, T_scaled]:
                    l1 += "%f " % ii
                l1 += "\n"
                f.write(l1)

                l2 = "out: %f\n" % fb_scaled
                f.write(l2)


def random_write_many3(sim, ioutputs, pickle_path, niter, fname, topology=[4, 25, 1], the_mass_bins=None):
    '''
    Write many outputs to one file randomly
    '''
    import random, math

    # This worked well for 106_70.train
    topology_string = "topology: %s\n" % _list_to_string(topology)

    if (the_mass_bins is None):
        the_mass_bins = [7., 7.25, 7.5, 7.75, 8., 8.25, 8.5, 8.75, 9., 9.25, 9.5, 9.75, 10.]

    # Load all training data
    log_mvir_all = []; log_mvir_all_scaled = []; fb_all_scaled = []; ftidal_all_scaled = []
    xHII_all_scaled = []; T_all_scaled = []
    for ioutput in ioutputs:
        print "%05i" % ioutput
        snapshot = sim[ioutput]

        log_mvir, fb, ftidal, xHII, T = load_training_arrays(snapshot, \
                pickle_path=pickle_path, weight="mw")

        log_mvir_scaled = _scale_mvir(log_mvir)
        fb_scaled = _scale_fb(fb)
        # xHII_scaled = _scale(xHII)
        # T_scaled = _scale(T)

        # log_mvir_scaled = log_mvir
        ftidal_scaled = _scale_ftidal(ftidal)
        xHII_scaled = xHII
        T_scaled = T

        for i in range(len(log_mvir_scaled)):
            log_mvir_all.append(log_mvir[i])
            log_mvir_all_scaled.append(log_mvir_scaled[i])
            fb_all_scaled.append(fb_scaled[i])
            ftidal_all_scaled.append(ftidal_scaled[i])
            xHII_all_scaled.append(xHII_scaled[i])
            T_all_scaled.append(T_scaled[i])

    mass_bins = np.digitize(log_mvir_all, the_mass_bins, right=True)

    data = []
    nmin = len(np.where(mass_bins == mass_bins.min())[0])

    nmin = -np.inf
    for i in range(len(the_mass_bins)+1):
        print i
        idx = np.where(mass_bins == i)
        if len(idx[0]) > nmin:
            nmin = len(idx[0])
    print nmin, mass_bins.max()
    # Evenly add data in bins of mvir
    for i in range(len(the_mass_bins)+1):
        idx = np.where(mass_bins == i)
        nbin = len(idx[0])
        if nbin == 0:
            continue
        npass = int(math.ceil(float(nmin) / float(nbin))) # * max(1, np.round(float(i)/2.)))
        print nmin, nbin, npass, i
        for j in range(npass):
            for ix in idx[0]:
                data.append((log_mvir_all_scaled[ix], fb_all_scaled[ix], ftidal_all_scaled[ix], xHII_all_scaled[ix], T_all_scaled[ix]))

    print "Writing %i entries" % (len(data))

    with open(fname, "w") as f:
        f.write(topology_string)
        # Write to file
        for j in range(niter):
            ix = range(len(data))
            random.shuffle(ix)
            for i in ix:
                log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled = data[i]
                l1 = "in: "
                for ii in [log_mvir_scaled, ftidal_scaled, xHII_scaled, T_scaled]:
                    l1 += "%f " % ii
                l1 += "\n"
                f.write(l1)

                l2 = "out: %f\n" % fb_scaled
                f.write(l2)


def run_ANN(dir_name, iout, weight, NN, **kwargs):
    import os
    from seren3.utils import which

    nn_binary = which("neural-net")
    nloop = kwargs.pop("nloop", 1)

    # Switch to correct directory
    cwd = os.getcwd()
    os.chdir(dir_name)

    def _weight_fname(iout, weight):
        return "fb_%05i_%s.weights" % (iout, weight)

    exe = "%s %i -f fb_%05i_%s.train -s %s" % (nn_binary, nloop, iout, weight, _weight_fname(iout, weight))
    os.system(exe)
    print exe

    predict_files = ["", "_zero_ftidal", "_mean_xHII", "_zero_ftidal_mean_xHII"]
    for predict in predict_files:
        predict_fname = "fb_%05i_%s%s.predict" % (iout, weight, predict)
        out_fname = "fb_%05i_NN%i_%s%s.results" % (iout, NN, weight, predict)

        exe = "%s -l %s -p %s -o %s" % (nn_binary, _weight_fname(iout, weight), predict_fname, out_fname)
        os.system(exe)
        print exe

    os.chdir(cwd)

def run_neural_net(sim, ioutputs, pickle_path, out_dir, nloop, N_hidden, weight="mw"):
    '''
    Runs the neural network for a list of redshifts
    '''
    import os

    niter = 1000
    topology = [4, N_hidden, 1]
    NN = sum(topology)

    for ioutput in ioutputs:
        snap = sim[ioutput]

        dir_name = "%s/neural-net/%d_final/" % (out_dir, ioutput)
        if (os.path.isdir(dir_name) is False):
            os.mkdir(dir_name)

        log_mvir, fb, ftidal, xHII, T = load_training_arrays(snap, pickle_path=pickle_path, weight=weight)

        # log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled = write_training_data(snap, log_mvir, \
        #     fb, np.log10(ftidal), np.log10(xHII), T, \
        #     niter, topology, dir_name, weight)

        log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled = write_training_data(snap, log_mvir, \
            fb, ftidal, xHII, T, \
            niter, topology, dir_name, weight)

        # log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled = write_training_data(snap, log_mvir, \
        #     fb, ftidal, np.log10(xHII), T, \
        #     niter, topology, dir_name, weight)

        make_all_prediction_files(snap, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled,\
             dir_name, weight)

        run_ANN(dir_name, ioutput, weight, NN, nloop=nloop)

def compute_ANN_Mc(sim, ioutput, out_dir, NN, alpha=2., use_lmfit=True, weight="mw", **kwargs):
    '''
    Load ANN results and compute Mc
    '''
    import numpy as np
    from seren3.analysis import baryon_fraction

    _X_IDX = 0
    _Y_IDX = 4

    fix_alpha = kwargs.pop("fix_alpha", False)

    def _get_dir_name(out_dir, ioutput):
        return "%s/neural-net/%i_final/" % (out_dir, ioutput)

    def _get_input_fname(out_dir, ioutput, weight):
        return "%s/fb_%05i_%s.input" % (out_dir, ioutput, weight)

    def _get_results_fname(out_dir, ioutput, weight, NN):
        return "%s/fb_%05i_NN%i_%s.results" % (out_dir, ioutput, NN, weight)

    def reverse_scaling(data):
        x,y = (data[_X_IDX], data[_Y_IDX])

        def _reverse_scaling_mvir(arr):
            unscaled = arr/2.
            unscaled += 0.5
            unscaled *= (_MVIR_MAX - _MVIR_MIN)
            unscaled += _MVIR_MIN
            return unscaled

        def _reverse_scaling_fb(arr):
            unscaled = arr/2.
            unscaled += 0.5
            unscaled *= (_FB_MAX - _FB_MIN)
            unscaled += _FB_MIN
            return unscaled

        # return _reverse_scaling(x, x_orig), _reverse_scaling(y, y_orig)
        # return x, _reverse_scaling2(y, y_orig)
        return _reverse_scaling_mvir(x), _reverse_scaling_fb(y)

    # print sim, ioutput, out_dir, NN, weight, kwargs

    # for ioutput in ioutputs:
        # ioutput = sim.redshift(z)
    snap = sim[ioutput]
    cosmo = snap.cosmo

    cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]

    # log_mvir, fb, ftidal, xHII, T = load_training_arrays(snap, pickle_path=pickle_path, weight=weight)

    dir_name = _get_dir_name(out_dir, ioutput)

    # input_fname = _get_input_fname(dir_name, ioutput, weight)
    # input_data = np.loadtxt(input_fname, unpack=True)

    results_fname = _get_results_fname(dir_name, ioutput, weight, NN)
    # print results_fname
    results_data = np.loadtxt(results_fname, unpack=True)

    # log_mvir_unscaled, fb_cosmic_mean_unscaled = reverse_scaling(results_data, log_mvir, fb)
    log_mvir_unscaled, fb_unscaled = reverse_scaling(results_data)
    mvir_unscaled = 10**log_mvir_unscaled
    fb_cosmic_mean_unscaled = fb_unscaled / cosmic_mean_b
    
    fit_dict = baryon_fraction.fit(mvir_unscaled, fb_unscaled, fix_alpha, alpha=alpha, use_lmfit=use_lmfit, **cosmo)

    return fit_dict




