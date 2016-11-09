def plot_many(bins, linestyles, bin_labels, age_then=None, z_func=None, colors=None, **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.array import SimArray
    from seren3.utils.plot_utils import ncols

    ax1 = plt.gca()

    legendArtists = []
    for i in range(len(bins)):
        ls = linestyles[i]
        plot_mean_integrated_fesc(bins[i], linestyle=ls, colors=colors,\
                     label=False, legend=False, **kwargs)

        legendArtists.append(plt.Line2D((0,1),(0,0), color='k', linestyle=ls))

    handles, labels = ax1.get_legend_handles_labels()
    display = tuple(range(len(bins[0])))

    ax1.set_xlim(0.2, 0.7)

    # Shrink current axis's height by 10% on the bottom
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    legend_kwargs = {"title" : r"log$_{10}$(Mvir)", "loc" : "upper center", \
                "bbox_to_anchor" : (0.5, -0.1), "fancybox" : True, "shadow" : True, "ncol" : 4}
    ax1.legend([handle for i,handle in enumerate(handles) if i in display]+legendArtists,\
             [label for i,label in enumerate(labels) if i in display]+bin_labels, **legend_kwargs)

    # Redshift axis
    if (z_func is not None) and (age_then is not None):
        ax2 = ax1.twiny()
        ax2.set_position([box.x0, box.y0 + box.height * 0.1,
             box.width, box.height * 0.9])

        def tick_function(X):
            # return ["%1.1f" % i for i in z_func( age_now - X )]
            # We've scaled the x-axis by the age of halos, so need
            # to account for zero-point in age
            return ["%1.1f" % i for i in z_func( SimArray(X+age_then, "Gyr") )]

        xtickpos = ax1.get_xticks()
        print xtickpos
        new_tick_locations = np.linspace(0.215, 0.685, 5)
        print new_tick_locations
        # new_tick_locations = np.array(ax1.get_xticks())

        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel(r"Redshift")

    ax1.set_xlabel(r"t$_{\mathrm{H}}$ [Gyr]")
    ax1.set_ylabel(r"$\langle \mathrm{f}_{\mathrm{esc}} \rangle$ (<t$_{\mathrm{H}}$ [%])")
    # plt.show(block=False)

def plot_mean_integrated_fesc(binned_data, linestyle='-',\
                         colors=None, label=True, legend=False, **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    from seren3.utils.plot_utils import ncols

    if colors is None:
        cmap = kwargs.pop("cmap", "jet")
        print "Using cmap: ", cmap
        colors = ncols(len(binned_data) -1 , cmap=cmap)
    else:
        assert len(colors) == len(binned_data) - 1

    ax = plt.gca()

    for i in range(len(binned_data)):
        mbin = binned_data[i]
        x, y = (mbin.thalo, mbin.mean)
        if mbin.mass_bin == "All":
            ax.semilogy(x, y, label="All", color="k",
                        linewidth=5, linestyle=linestyle)
        else:
            c = colors[i]
            lower = "%1.1f" % mbin.mass_bin
            upper = None
            if i == len(binned_data) - 2:
                upper = r"$\infty$"
            else:
                upper = "%1.1f" % binned_data[i+1].mass_bin

            ax.semilogy(x, y, label="[%s, %s)" % (lower, upper), color=c,\
                     linewidth=1.5, linestyle=linestyle)

    if label:
        plt.xlabel(r"t$_{\mathrm{Lookback}}$ [Gyr]")
        plt.ylabel(r"$\langle \mathrm{f}_{\mathrm{esc}} \rangle$ (<t$_{\mathrm{H}}$ [%])")
    
    if legend:
        plt.legend(loc='lower right')
    # plt.show(block=False)

class MassBin(object):
    def __init__(self, idx, mass_bin):
        self.idx = idx
        self.mass_bin = mass_bin
        self.lbtime = []
        self.z = []
        self.mean = []
        self.std = []

    def __str__(self):
        return "Mass bin: %s" % self.mass_bin

    def __repr__(self):
        return self.__str__()

    @property
    def thalo(self):
        import numpy as np
        return np.max(self.lbtime) - np.array(self.lbtime)

def mean_integrated_fesc(simulation, ioutputs, the_mass_bins=None):
    '''
    Compute the mean integrated fesc at each output in ioutputs
    '''
    import numpy as np

    # Create bin idx
    if the_mass_bins is None:
        the_mass_bins = np.array( [8., 8.5, 9., 9.5, 10.0,] )
    # the_mass_bins = np.array( [8., 9., 10.0] )

    pickle_dir = "%s/pickle/ConsistentTrees/" % simulation.path
    # Sort output numbers from high to low
    sorted_ioutputs = sorted(ioutputs, reverse=True)
    # For computing lookback-time from last snapshot
    age_now = simulation[sorted_ioutputs[0]].age

    binned_data = [ MassBin(i, the_mass_bins[i]) for i in range(len(the_mass_bins)) ]
    binned_data.append( MassBin(len(the_mass_bins)+1, "All") )  # Bin to contain all halos

    def _store(mass_bin, tint_fesc_arr, lbtime, z):
        mass_bin.mean.append(tint_fesc_arr.mean())
        mass_bin.std.append(tint_fesc_arr.std())
        mass_bin.lbtime.append(lbtime)
        mass_bin.z.append(z)

    for i in range(len(sorted_ioutputs)):
        ioutput = sorted_ioutputs[i]
        snapshot = simulation[ioutput]
        fname = "%s/time_int_fesc_all_halos_%05i.p" % (pickle_dir, snapshot.ioutput)
        data = snapshot.pickle_load(fname)

        # Load data into np arrays
        tint_fesc = np.zeros(len(data))
        Mvir = np.zeros(len(data))
        for j in xrange(len(data)):
            result = data[j].result
            Mvir[j] = result["Mvir"]
            # take first value only, which is integrated over entire history of the halo
            tint_fesc[j] = result["tint_fesc_hist"][0] * 100.  # *100. = fraction -> %

        # Lookback time from final snapshot
        lbtime = age_now - snapshot.age

        # Remove nans
        good = np.where( ~np.isnan(tint_fesc) )
        tint_fesc = tint_fesc[good]
        Mvir = Mvir[good]

        mass_bins = np.digitize(np.log10(Mvir), the_mass_bins, right=True)

        # Do the binning. 
        for mbin in binned_data:
            if mbin.mass_bin == "All":
                _store(mbin, tint_fesc, lbtime, snapshot.z)
            else:
                idx = np.where( mass_bins == mbin.idx )
                _store(mbin, tint_fesc[idx], lbtime, snapshot.z)

    return binned_data
