import numpy as np

def _gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def _bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return _gauss(x,mu1,sigma1,A1)+_gauss(x,mu2,sigma2,A2)


def tag_halos_tidal_peaks(snapshot, fname, nbins=20, cl=3):
    P,C,bincenters,dx,hid,tidal_force_tdyn,\
            indexes,peaks_x,params,sigma\
             = tidal_force_pdf(snapshot, fname, plot=False, nbins=20)

    mu1,sigma1,A1,mu2,sigma2,A2 = params
    p1_min, p1_max = ( mu1 - sigma1*cl, mu1 + sigma1*cl )
    p2_min, p2_max = ( mu2 - sigma2*cl, mu2 + sigma2*cl )

    idx1 = np.where( np.logical_and(tidal_force_tdyn >= 10**p1_min, tidal_force_tdyn <= 10**p1_max) )
    idx2 = np.where( np.logical_and(tidal_force_tdyn >= 10**p2_min, tidal_force_tdyn <= 10**p2_max) )

    halo_dict = {i:0 for i in hid}

    for i in hid[idx1]:
        halo_dict[i] = 1  # peak 1
    for i in his[idx2]:
        halo_dict[i] = 2  # peak 2

    return halo_dict


def ftidal_xHII_corr(sim, ioutputs, pickle_path=None):
    '''
    Measure correlation between ftidal and xHII
    '''
    from scipy.stats.stats import pearsonr 
    from seren3.analysis.baryon_fraction import neural_net2

    if (pickle_path is None):
        pickle_path = "%s/pickle/" % sim.path

    corr_coeff = np.zeros(len(ioutputs))
    z = np.zeros(len(ioutputs))

    for ioutput in ioutputs:
        snap = sim[ioutput]

        log_mvir, fb, ftidal, xHII, T, T_U, pid = neural_net2.load_training_arrays(snapshot, pickle_path=pickle_path, weight="mw")
        corr_mat = pearsonr(ftidal, xHII)
        corr_coef[i] = corr_mat[0]
        z[i] = snap.z

    return z, corr_coeff


def pdf_sample_function(snapshot, **kwargs):
    '''
    Return a function for sampling the pdf
    '''
    P,C,bincenters,dx,x,y_2,(ftidal,indexes,peaks_x,params,sigma) = tidal_force_pdf(snapshot, **kwargs)

    fn = lambda: np.random.choice(bincenters, p=P)
    return fn


def generate_neural_net_sample(snapshot, pickle_path=None, **kwargs):
    '''
    Generates a neural net prediction file with random sampling
    from the tidal force PDF
    '''
    from seren3.analysis.baryon_fraction import neural_net2

    reload(neural_net2)

    if (pickle_path is None):
        pickle_path = "%s/pickle/" % snapshot.path

    out_dir = "%s/neural-net2/%d_final/" % (snapshot.path, snapshot.ioutput)
    weight = "mw"

    log_mvir, fb, ftidal, xHII, T, T_U, pid = (None, None, None, None, None, None, None)
    if "data" in kwargs:
        log_mvir, fb, ftidal, xHII, T, T_U, pid = kwargs.pop("data")
    else:
        log_mvir, fb, ftidal, xHII, T, T_U, pid = neural_net2.load_training_arrays(snapshot, pickle_path=pickle_path, weight=weight)

    # Sample fitdal force below biggest peak
    P,C,bincenters,dx,x,y_2,(ftidal,indexes,peaks_x,params,sigma) = tidal_force_pdf(snapshot, **kwargs)

    # print P.sum()

    ftidal_sampled = np.zeros(len(ftidal))
    for i in range(len(ftidal_sampled)):
        sample = np.inf
        while (sample > peaks_x[-1]):
            sample = np.random.choice(bincenters, p=P)
        ftidal_sampled[i] = 10**sample

    xHII_scaled = neural_net2._scale_xHII(xHII)
    T_scaled = T
    pid_scaled = neural_net2._scale_pid(pid)
    ftidal_scaled = neural_net2._scale_ftidal(ftidal_sampled)

    log_mvir_scaled = neural_net2._scale_mvir(log_mvir)
    fb_scaled = neural_net2._scale_fb(fb)

    # idx = np.where(pid == -1)
    # log_mvir_scaled = log_mvir_scaled[idx]; fb_scaled = fb_scaled[idx]; ftidal_scaled = ftidal_scaled[idx]
    # xHII_scaled = xHII_scaled[idx]; T_scaled = T_scaled[idx]; T_U = T_U[idx]; pid_scaled = pid_scaled[idx]

    neural_net2.write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled, T_U, pid_scaled, out_dir, weight, label="ftidal_pdf_sampling", write_mass=True, raw_input_format=True)
    neural_net2.write_input_data(snapshot, log_mvir_scaled, fb_scaled, ftidal_scaled, xHII_scaled, T_scaled, T_U, pid_scaled, out_dir, weight, label="ftidal_pdf_sampling", write_mass=False)


def plot_RT2_panels(**kwargs):
    import seren3

    sim = seren3.load("RT2_nohm")
    ioutputs = [106, 100, 90, 80, 70, 60]

    plot_panels(sim, ioutputs, 2, 3, **kwargs)


def plot_panels(sim, ioutputs, nrows, ncols, nbins=35):
    from seren3.analysis.plots import histograms
    import matplotlib.pylab as plt

    reload(histograms)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(14,10))
    text_pos = (-2.1, 0.085)

    row_count = 0
    col_count = 0
    for ioutput, ax in zip(ioutputs, axes.flatten()):
        snap = sim[ioutput]
        text = "z = %1.2f" % snap.z
        P,C,bincenters,dx,x,y_2,(ftidal,indexes,peaks_x,params,sigma) = tidal_force_pdf(snap, nbins=nbins, plot=False)

        ax1, ax2 = histograms.plot_pdf_cdf(snap, P, bincenters, dx, True, r"", C=C, ax1=ax, label=False, cumul_col="#3333FF")

        ax1.plot(x, y_2, color="r", lw=3, label='model')
        ax1.text(text_pos[0], text_pos[1], text, color="k", size="x-large")

        if col_count == 0:
            ax1.set_ylabel("PDF")
        elif col_count == ncols-1:
            ax2.set_ylabel("Cumulative")

        if row_count == nrows-1:
            ax1.set_xlabel(r"log$_{10}$ $\langle F_{\mathrm{Tidal}} \rangle_{t_{\mathrm{dyn}}}$")

        col_count += 1
        if col_count == ncols:
            row_count += 1
            col_count = 0

        ax1.set_ylim(0.0, 0.1)



def tidal_force_pdf(snapshot, nbins=35, plot=False, **kwargs):
    '''
    Compute and (optional) plot the PDF and CDF of tidal forces for this
    snapshot
    '''
    import pickle
    import peakutils
    from peakutils.peak import centroid
    from seren3.analysis.plots import fit_scatter
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit

    def pdf(arr, bins):
        log_arr = np.log10(arr)
        idx = np.where(np.isinf(log_arr))
        log_arr = np.delete(log_arr, idx)

        P, bin_edges = np.histogram(log_arr, bins=bins,  density=False)
        P = np.array( [float(i)/float(len(arr)) for i in P] )
        bincenters = 0.5*(bin_edges[1:] + bin_edges[:-1])
        dx = (bincenters.max() - bincenters.min()) / bins
        C = np.cumsum(P) * dx
        C = (C - C.min()) / C.ptp()
        return P,C,bincenters,dx

    halos = kwargs.pop("halos", snapshot.halos())
    ftidal = np.zeros(len(halos))

    for i in range(len(halos)):
        h = halos[i]
        ftidal[i] = h["tidal_force_tdyn"]

    idx = np.where(ftidal > 0)
    # ftidal[idx] = 1e-6
    ftidal = ftidal[idx]

    P,C,bincenters,dx = pdf(ftidal, nbins)

    # print P.max()

    # Fit the bimodal guassian
    # Interpolate the PDF
    fn = interp1d(bincenters, P)
    x = np.linspace(bincenters.min(), bincenters.max(), 1000)
    y = fn(x)
    
    # Fit peaks to get initial estimate of guassian properties
    indexes = peakutils.indexes(y, thres=0.02, min_dist=250)
    peaks_x = peakutils.interpolate(x, y, ind=indexes)

    # Do the bimodal fit
    expected = (peaks_x[0], 0.2, 1.0, peaks_x[1], 0.2, 1.0)
    params,cov=curve_fit(_bimodal,x,y,expected)
    sigma=np.sqrt(np.diag(cov))

    # Refit the peaks to improve accuracy
    y_2 = _bimodal(x,*params)
    fn = interp1d(x, y_2)
    indexes = peakutils.indexes(y_2, thres=0.02, min_dist=250)
    peaks_x = peakutils.interpolate(x, y_2, ind=indexes)

    # print y_2.max()

    if plot:
        from seren3.analysis.plots import histograms
        import matplotlib.pylab as plt

        ax = None
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig, ax = plt.subplots()
        histograms.plot_pdf_cdf(snapshot, P, bincenters, dx, True, r"$\langle F_{\mathrm{Tidal}} \rangle_{t_{\mathrm{dyn}}}$", C=C, ax1=ax, **kwargs)

        ax.plot(x, y_2, color="r", lw=3, label='model')

        if "text" in kwargs:
            text_pos = (-2, 0.05)

            ax.text(text_pos[0], text_pos[1], kwargs.pop("text"), color="k", size="large")

    return P,C,bincenters,dx,x,y_2,(ftidal,indexes,peaks_x,params,sigma)
