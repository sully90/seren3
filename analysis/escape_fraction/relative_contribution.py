# At each output, compute the cumulative sum of photons emitted over all of time,
# and the total number that have escaped and plot
# alternative: fit scatter to distributions?
import numpy as np
from seren3.analysis import plots
from seren3.utils import flatten_nested_array

class MassBin(object):
    def __init__(self, idx, mass_bin):
        self.idx = idx
        self.mass_bin = mass_bin
        self.lbtime = []
        self.nphot_esc = []
        self.relative_contribution = []

    def extend(self, lbtime, I1):
        self.lbtime.extend(lbtime)
        self.nphot_esc.extend(I1)

    def __str__(self):
        return "Mass bin: %s" % self.mass_bin

    def __repr__(self):
        return self.__str__()

    def sum(self, nbins=10):
        reload(plots)

        # x = flatten_nested_array(np.array(self.lbtime))
        # y = flatten_nested_array(np.array(self.nphot_esc))
        x = np.array(self.lbtime)
        y = np.array(self.nphot_esc)
        good = np.where(~np.isnan(y))
        bc, sy = plots.sum_bins(x[good], y[good], nbins=nbins)

        return bc, sy

def halo_photon_relative_contribution(simulation, ioutputs, the_mass_bins=[8., 8.5,  9., 9.5, 10.]):
    import random
    from scipy import interpolate
    from seren3.scripts.mpi import write_fesc_hid_dict

    if not isinstance(the_mass_bins, np.ndarray):
        the_mass_bins = np.array(the_mass_bins)

    pickle_dir = "%s/pickle/ConsistentTrees/" % simulation.path

    snapshot = simulation[ioutputs[-1]]
    age_now = snapshot.age

    binned_data = [ MassBin(i, the_mass_bins[i]) for i in range(len(the_mass_bins)) ]
    binned_data.append( MassBin(len(the_mass_bins)+1, "All") )  # Bin to contain all halos

    def _store(mass_bin, tint_fesc_arr, Nion_d_sum, lbtime, z):
        mass_bin.tint_fesc_mean.append(tint_fesc_arr.mean())
        mass_bin.Nion_d_sum.append(Nion_d_sum.sum())
        # mass_bin.std.append(tint_fesc_arr.std())
        mass_bin.lbtime.append(lbtime)
        mass_bin.z.append(z)

    for ioutput in ioutputs:
        print ioutput
        snapshot = simulation[ioutput]
        data = write_fesc_hid_dict.load_db(simulation.path, ioutput)

        # Compute quantities
        lbtime = age_now - snapshot.age
        Nion_esc = np.zeros(len(data))
        Mvir = np.zeros(len(data))
        for i in range(len(data.keys())):
            hid = data.keys()[i]
            res = data[hid]
            fesc_h = res["fesc"]
            if (fesc_h > 1.):
                fesc_h = random.uniform(0.9, 1.0)

            Nion_esc[i] = fesc_h * (res["Nion_d_now"] * res["star_mass"].in_units("Msol")).sum()            
            Mvir[i] = res["hprops"]["mvir"]

        # Bin data
        mass_bins = np.digitize( np.log10(Mvir), the_mass_bins, right=True )
        lbtime_arr = np.ones(len(Nion_esc)) * lbtime
        # Do the binning
        for mbin in binned_data:
            if mbin.mass_bin == "All":
                mbin.extend(lbtime_arr, Nion_esc)
            else:
                idx = np.where( mass_bins == mbin.idx )
                mbin.extend(lbtime_arr[idx], Nion_esc[idx])

    # return binned_data

    nbins=10
    # Compute the relative contributions
    bc_all, sy_all = binned_data[-1].sum(nbins=nbins)
    fn_all = interpolate.interp1d(bc_all, np.log10(sy_all), fill_value="extrapolate")
    for i in range(len(binned_data) - 1):
        mbin = binned_data[i]
        bc, sy = mbin.sum(nbins=nbins)        
        mbin.relative_contribution = sy / 10**fn_all(bc)
        nbins-=2
    binned_data[-1].relative_contribution = np.ones(len(sy))

    return binned_data

