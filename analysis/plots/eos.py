import seren3
import numpy as np
import matplotlib.pylab as plt
import weakref
from seren3.array import SimArray
from seren3.utils.plot_utils import add_colorbar

def plot_baryfrac(pp1=None, pp2=None):
    path = '/research/prace/david/bpass/bc03/'
    sim = seren3.init(path)

    snap1 = sim[60]
    snap2 = sim[100]

    if (pp1 is None):
        pp1 = PhasePlot(snap1)
    if (pp2 is None):
        pp2 = PhasePlot(snap2)

    fig, axs = plt.subplots(nrows=1, ncols=2)#, figsize=(18,8))
    pp1.draw(ax=axs[0], draw_cbar=False)
    pp2.draw(ax=axs[1], draw_cbar=False)
    # plt.tight_layout()

    cbar1 = fig.colorbar(pp1.im, ax=axs[0])
    cbar2 = fig.colorbar(pp2.im, ax=axs[1])
    cbar1.set_label('f(mass)', labelpad=-5)
    cbar2.set_label('f(mass)', labelpad=-5)

    for ax, snap in zip(axs, [snap1, snap2]):
        ax.set_title("z = %1.2f" % snap.z, fontsize=20)

    # plt.show()
    plt.savefig("./bc03_phase_diag.pdf", format="pdf", dpi=1000)
    return pp1, pp2


class PhasePlot(object):
    '''
    Class to handle equation of state plot and annotations
    '''
    def __init__(self, snapshot, den_field='nH', temp_field='T2', limit_axes=False, load_nH=False, verbose=True, **kwargs):
        from seren3.utils.sge import ncpu

    	nthreads = kwargs.get('nthreads', 8)
        if hasattr(snapshot, "base"):
            snapshot.base.set_nproc(nthreads)
        else:
            snapshot.set_nproc(nthreads)

        self._snapshot = weakref.ref(snapshot)
        self.den_field = den_field
        self.temp_field = temp_field

        self.cmap = kwargs.get('cmap', 'jet_black')
        self.verbose = verbose
        self.annotations = []

        if self.verbose: print 'Reading data on %i threads' % nthreads

        if den_field != 'nH' and load_nH:
            # Need nH for annotations
            self.dset = snapshot.g[[den_field, 'nH', 'mass', temp_field]].flatten()
        else:
            self.dset = snapshot.g[[den_field, temp_field, 'mass']].flatten()

        kwargs['plot'] = False
        kwargs['xlogrange'] = True
        kwargs['ylogrange'] = True
        self.kwargs = kwargs

        if limit_axes:
            self.vmin = kwargs.get('vmin', -10)
            self.vmax = kwargs.get('vmax', -2)
            self.ymin = kwargs.get('ymin', -1)
            self.ymax = kwargs.get('ymax', 8)
            self.xmin = kwargs.get('xmin', -6)
            self.xmax = kwargs.get('xmax', 6)
        else:
            self.vmin = None
            self.vmax = None
            self.ymin = None
            self.ymax = None
            self.xmin = None
            self.xmax = None

        self.im = None

        # xs & ys are logscaled nH / Temperature respectively
        if self.verbose: print 'Creating histogram'
        self.h, self.xs, self.ys = self._profile(**kwargs)

        if self.verbose: print 'Done'

    @property
    def snapshot(self):
        if self._snapshot is None:
            raise Exception("Lost reference to base snapshot")
        return self._snapshot()

    def _profile(self, **kwargs):
        from seren3.analysis.plots import histograms

        xo, yo = (self.dset[self.den_field], self.dset[self.temp_field])
        mass = self.dset['mass'].in_units("Msol")
        totmass = mass.sum()

        nbins = self.nbins
        h, xs, ys = histograms.hist2d(xo, yo, density=False, mass=mass, nbins=nbins, **kwargs)
        h /= totmass

        return h, xs, ys

    def _setup_axes(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(np.log10(self.h), cmap=self.cmap, \
                extent=[self.xs.min(), self.xs.max(),\
                self.ys.min(), self.ys.max()], aspect='auto')
        self.im.set_clim(self.vmin, self.vmax)

    def _teardown(self):
        self.fig = None
        self.ax = None
        self.im = None

    @property
    def nbins(self):
        return self.kwargs.get('nbins', 500)

    def fit(self, nbins_fit):
        from seren3.analysis.plots import fit_scatter

        bc, mean, std, sterr = fit_scatter(np.log10(self.dset[self.den_field]),\
                 np.log10(self.dset[self.temp_field]), nbins=nbins_fit, ret_sterr=True)
        return bc, mean, std, sterr

    def T_poly(self):
        '''
        Computes polytropic temperature
        '''
        from seren3.core.snapshot import NML

        log_nH = np.log10(self.dset['nH'])
        nbins = self.nbins
        bins = np.linspace(log_nH.min(), log_nH.max(), nbins)
        nH = 10**bins

        nml = self.snapshot.nml
        PHYSICS_PARAMS = nml[NML.PHYSICS_PARAMS]
        n_star = SimArray(PHYSICS_PARAMS['n_star'], "cm**-3").in_units("m**-3")
        T2_star = PHYSICS_PARAMS['T2_star']
        g_star = PHYSICS_PARAMS.get('g_star', 2.0)

        return T2_star * (nH / n_star) ** (g_star-1.0)

    def annotate_fit(self):
        self.annotations.append(_annotate_fit)

    def annotate_fit_mw(self):
        self.annotations.append(_annotate_fit_mw)

    def annotate_nstar(self):
        self.annotations.append(_annotate_nstar)

    def annotate_T2star(self):
        self.annotations.append(_annotate_T2star)

    def annotate_T_poly(self):
        self.annotations.append(_annotate_T_poly)

    def annotate_T2_thresh(self):
        self.annotations.append(_annotate_T2_thresh)

    def remove_fit_annotation(self):
        self.annotations.remove(_annotate_fit)

    def remove_nstar_annotation(self):
        self.annotations.remove(_annotate_nstar)

    def remove_T2star_annotation(self):
        self.annotations.remove(_annotate_T2star)

    def remove_T_poly_annotation(self):
        self.annotations.remove(_annotate_T_poly)

    def remove_T2_thresh_annotation(self):
        self.annotations.remove(_annotate_T2_thresh)

    def draw(self, ax=None, label_axes=True, draw_cbar=True, **kwargs):
        if ax is None:
            self._setup_axes()
            ax = self.ax
        else:
            self.im = ax.imshow(np.log10(self.h), cmap=self.cmap, \
                    extent=[self.xs.min(), self.xs.max(),\
                    self.ys.min(), self.ys.max()], aspect='auto')
            self.im.set_clim(self.vmin, self.vmax)

        for anno in self.annotations:
            anno(ax, self, **kwargs)

        # Get appropiate latex
        # temp_latex = seren3.get_derived_field_latex(seren3.Field(('amr', self.temp_field)))
        # den_latex = seren3.get_derived_field_latex(seren3.Field(('amr', self.den_field)))
        temp_latex = self.ys.latex
        den_latex = self.xs.latex

        if label_axes:
            ax.set_xlabel(r'log$_{10}$(%s)' % den_latex)
            ax.set_ylabel(r'log$_{10}$(%s)' % temp_latex)

        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlim(self.xmin, self.xmax)

        if draw_cbar:
            cbar = add_colorbar(self.im)
            cbar.set_label('f(mass)')

    def show(self, block=False):
        self.draw()
        plt.show(block=block)
        self._teardown()

def _annotate_fit(ax, pp, **kwargs):
    bc, mean, std, sterr = pp.fit(nbins_fit = kwargs.pop("nbins_fit", pp.nbins/4))
    # ax.plot(pp.xs, y, linestyle='-', color='k')
    ax.errorbar(bc, mean, yerr=sterr, linestyle='-', color='k')

def _annotate_fit_mw(ax, pp, **kwargs):
    y = pp.fit_mw()
    log_den = np.log10(pp.dset[pp.den_field])
    bins = np.linspace(log_den.min(), log_den.max(), pp.nbins)
    ax.plot(bins, y, linestyle='-', color='b', linewidth=1.5)

def _annotate_nstar(ax, pp, **kwargs):
    from seren3.core.snapshot import NML

    nml = pp.snapshot.nml
    n_star = SimArray(nml[NML.PHYSICS_PARAMS]['n_star'], "cm**-3").in_units("m**-3")

    ymin, ymax = ax.get_ylim()

    if pp.ymin is not None and pp.ymax is not None:
        ymin, ymax = (pp.ymin, pp.ymax)

    ax.vlines(x=np.log10(n_star), ymin=ymin, ymax=ymax, linestyle='--', color='k')

def _annotate_T2star(ax, pp, **kwargs):
    from seren3.core.snapshot import NML

    nml = pp.snapshot.nml
    T2_star = float(nml[NML.PHYSICS_PARAMS]['T2_star'])
    xmin, xmax = ax.get_xlim()

    ax.hlines(y=np.log10(T2_star), xmin=xmin, xmax=xmax, linestyle='--', color='k')

def _annotate_T_poly(ax, pp, **kwargs):
    T_pol = pp.T_poly()
    log10_T_pol = np.log10(T_pol)

    log_nH = np.log10(pp.dset['nH'])
    nbins = pp.nbins
    log_nH = np.linspace(log_nH.min(), log_nH.max(), nbins)

    if pp.den_field == 'nH':
        ax.plot(log_nH, log10_T_pol, linestyle='--', color='k')
    else:
        ax2 = ax.twinx()
        ax2.plot(log_nH, log10_T_pol, linestyle='--', color='k')

def _annotate_T2_thresh(ax, pp, internal_thresh=2.e4, **kwargs):
    # T_pol = pp.T_poly()
    # y = internal_thresh + T_pol
    # log_y = np.log10(y)

    log_nH = None
    if pp.xmin is not None and pp.xmax is not None:
        log_nH = np.linspace(pp.xmin, pp.xmax, 500)
    else:
        log_nH = np.log10(pp.dset['nH'])
        nbins = pp.nbins
        log_nH = np.linspace(log_nH.min(), log_nH.max(), nbins)

    def T_poly(log_nH):
        '''
        Computes polytropic temperature
        '''
        from seren3.core.snapshot import NML

        nbins = pp.nbins
        bins = np.linspace(log_nH.min(), log_nH.max(), nbins)
        nH = 10**bins

        nml = pp.snapshot.nml
        PHYSICS_PARAMS = nml[NML.PHYSICS_PARAMS]
        n_star = SimArray(PHYSICS_PARAMS['n_star'], "cm**-3").in_units("m**-3")
        T2_star = PHYSICS_PARAMS['T2_star']
        g_star = PHYSICS_PARAMS.get('g_star', 2.0)

        return T2_star * (nH / n_star) ** (g_star-1.0)

    y = internal_thresh + T_poly(log_nH)
    log_y = np.log10(y)

    if pp.den_field == 'nH':
        ax.plot(log_nH, log_y, linestyle='--', color='k')
    else:
        ax2 = ax.twinx()
        ax2.plot(log_nH, log_y, linestyle='--', color='k')


def plot_grid(sim, ioutputs, nrows, ncols, plots=None, region=None, **kwargs):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

    if plots is None:
        plots = []
        for i in ioutputs:
            snapshot = sim[i]
            if region is not None:
                snapshot = snapshot[region]
            plots.append(PhasePlot(snapshot, **kwargs))

    for i in range(nrows):
        for j in range(ncols):
            ax = axs.flat[i+j]
            pp = plots[i+j]

            pp.cmap = 'jet_black'
            pp.draw(ax=ax, draw_cbar=False)

            if i != nrows-1:
                ax.set_xlabel('')
            if j != 0:
                ax.set_ylabel('')

            snapshot = sim[ioutputs[i+j]]
            ax.text(1, 0, "z = %1.3f" % snapshot.z)

            # ax.set_ylim(0, 8)
            # ax.set_xlim(-6, 0)

    im = pp.im

    cbar_ax = fig.add_axes([0.85, 0.05, 0.03, 0.9])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label('f(mass)')
    cb.solids.set_rasterized(True) 

    plt.subplots_adjust(right=0.8)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.05, hspace=0.05, wspace=0.05)

    return fig, axs, plots
