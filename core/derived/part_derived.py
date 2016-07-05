import seren3
from seren3.array import SimArray
import numpy as np
from pymses.utils import constants as C

@seren3.derived_quantity(requires=["epoch"], unit=C.Gyr)
def part_age(context, dset, **kwargs):
    #return context.quantities.tform(dset['epoch'], **kwargs)
    '''
    Return the formation (lookback) time in units of Gyr
    For formation time since the big bang (age), do age_simu - tform (also in units of Gyr)
    '''
    nml = context.nml
    isCosmoSim = ('cosmo' in nml['RUN_PARAMS'] and nml['RUN_PARAMS']['cosmo'] == '.true.')
    verbose = kwargs.get("verbose", False)
    tform = dset['epoch']

    output = None
    if verbose: print 'part_age isCosmoSim: ', isCosmoSim
    if not isCosmoSim:
        output = tform * context.info['unit_time'].express(context.C.Gyr)  # age in Gyr
    else:
        cosmology = context.cosmo
        h0 = cosmology['h'] * 100

        friedmann = context.friedmann

        axp_out = friedmann['axp_out']
        # hexp_out = friedmann['hexp_out']
        tau_out = friedmann['tau_out']
        t_out = friedmann['t_out']
        # age_tot = friedmann['age_tot']
        # age_simu = friedmann['age_simu']
        time_simu = friedmann['time_simu']

        patch = kwargs.get('patch', context.patch)
        
        if verbose: print 'part_age: patch = ', patch
        if patch == 'rt':
            output = (time_simu - tform) / (h0 * 1e5 / 3.08e24) / (365. * 24. * 3600. * 1e9)
        else:
            ntable = len(axp_out) - 1

            output = np.zeros(len(tform))
            for j in range(0, len(tform) - 1):

                i = 1
                while ((tau_out[i] > tform[j]) and (i < ntable)):
                    i += 1

                # Interpolate time
                time = t_out[i] * (tform[j] - tau_out[i - 1]) / (tau_out[i] - tau_out[i - 1]) + \
                    t_out[i - 1] * (tform[j] - tau_out[i]) / \
                    (tau_out[i - 1] - tau_out[i])

                time = max(
                    (time_simu - time) / (h0 * 1e5 / 3.08e24) / (365 * 24 * 3600 * 1e9), 0)
                output[j] = time

                # output[j] = (time_simu - time)*unit_t/(365 * 24 * 3600 * 1e9)

    result = SimArray(output, "Gyr")
    return result