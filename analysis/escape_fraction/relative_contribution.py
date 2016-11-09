# At each output, compute the cumulative sum of photons emitted over all of time,
# and the total number that have escaped and plot
# alternative: fit scatter to distributions?

class MassBin(object):
    def __init__(self, idx, mass_bin):
        self.idx = idx
        self.mass_bin = mass_bin
        self.lbtime = []
        self.z = []
        self.tint_fesc_mean = []
        self.Nion_d_sum = []
        self.rel_contrib = []

    def __str__(self):
        return "Mass bin: %s" % self.mass_bin

    def __repr__(self):
        return self.__str__()

    @property
    def thalo(self):
        import numpy as np
        return np.max(self.lbtime) - np.array(self.lbtime)

def halo_photon_relative_contribution(simulation, ioutputs, the_mass_bins=[8., 9., 10.]):
    import numpy as np

    if not isinstance(the_mass_bins, np.ndarray):
        the_mass_bins = np.array(the_mass_bins)

    pickle_dir = "%s/pickle/ConsistentTrees/" % simulation.path

    # Sort output numbers from high to low
    sorted_ioutputs = sorted(ioutputs, reverse=True)

    # For computing lookback-time from last snapshot
    age_now = simulation[sorted_ioutputs[0]].age

    binned_data = [ MassBin(i, the_mass_bins[i]) for i in range(len(the_mass_bins)) ]
    binned_data.append( MassBin(len(the_mass_bins)+1, "All") )  # Bin to contain all halos

    def _store(mass_bin, tint_fesc_arr, Nion_d_sum, lbtime, z):
        mass_bin.tint_fesc_mean.append(tint_fesc_arr.mean())
        mass_bin.Nion_d_sum.append(Nion_d_sum.sum())
        # mass_bin.std.append(tint_fesc_arr.std())
        mass_bin.lbtime.append(lbtime)
        mass_bin.z.append(z)

    for i in xrange(len(sorted_ioutputs)):
        iout = sorted_ioutputs[i]
        snapshot = simulation[iout]
        fname = "%s/time_int_fesc_all_halos_%05i.p" % (pickle_dir, snapshot.ioutput)
        data = snapshot.pickle_load(fname)

        # Load data into numpy arrays
        tint_fesc = np.zeros(len(data))
        Nion_d_sum = np.zeros(len(data))
        Mvir = np.zeros(len(data))
        for j in range(len(data)):
            result = data[j].result
            tint_fesc[j] = result["tint_fesc_hist"][0] * 100.
            Nion_d_sum[j] = result["I2"].sum()
            Mvir[j] = result["Mvir"]

        lbtime = age_now - snapshot.age

        # Remove nans
        good = np.where( ~np.isnan(tint_fesc) )
        tint_fesc = tint_fesc[good]
        Nion_d_sum = Nion_d_sum[good]
        Mvir = Mvir[good]

        # Conpute relative contribution to ionizing photon budget
        rel_contrib = tint_fesc * Nion_d_sum

        mass_bins = np.digitize( np.log10(Mvir), the_mass_bins, right=True )

        # Do the binning. 
        for mbin in binned_data:
            if mbin.mass_bin == "All":
                _store(mbin, tint_fesc, Nion_d_sum, lbtime, snapshot.z)
            else:
                idx = np.where( mass_bins == mbin.idx )
                _store(mbin, tint_fesc[idx], Nion_d_sum[idx], lbtime, snapshot.z)
                # Relative contributuon
                mbin.rel_contrib.append( rel_contrib[idx].sum() / rel_contrib.sum() )

    return binned_data