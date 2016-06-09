import seren3
import numpy as np

@seren3.derived_quantity(requires=["mass", "pos"], latex=r'$\delta_{\mathrm{c}}$')
def part_cic_delta(context, dset, nn=None, deconvolve_cic=False):
    ''' Use Cloud In Cell smoothing to compute the density contrast field from a
    discrete particle distribution. Accounts for grid spacing to return the density
    field in true code units. '''
    from seren3 import cosmology
    if nn is None:
        nn = 2. ** (context.info['levelmin'])
    cosmo = context.cosmo
    omegac_0 = cosmo['omega_M_0'] - cosmo['omega_b_0']
    rho_mean = cosmology.rho_mean_z(omegac_0, **cosmo)
    rho = part_cic_rho(context, dset, nn=nn) * context.info['unit_density'].val
    delta = (rho - rho_mean) / rho_mean

    if deconvolve_cic:
        from seren3 import utils
        return utils.deconvolve_cic(delta, nn)

    return delta


@seren3.derived_quantity(requires=["mass", "pos"], latex=r'$\rho_{\mathrm{c}}$ [code_density]')
def part_cic_rho(context, dset, nn=None, deconvolve_cic=False):
    ''' Use Cloud In Cell smoothing to compute the density field from a
    discrete particle distribution. Accounts for grid spacing to return the density
    field in true code units. '''
    from seren3 import utils
    if nn is None:
        nn = 2. ** (context.info['levelmin'])
    cicfield = utils.cic(dset, nn, 'mass', average=False)
    dx_code = 1. / nn
    cicfield /= (dx_code ** 3.)

    if deconvolve_cic:
        return utils.deconvolve_cic(cicfield, nn)

    return cicfield


@seren3.derived_quantity(requires=["vel", "pos"], latex=r'$\rho_{\mathrm{c}}$ [code_density]')
def part_cic_v2(context, dset, nn=None, deconvolve_cic=False):
    ''' Use Cloud In Cell smoothing to approximate the velocity field
    TODO - Deconvolution with CIC kernel
    '''
    from seren3 import utils
    if nn is None:
        nn = 2. ** (context.info['levelmin'])
    cicfield = utils.cic(dset, nn, 'vel', average=True)

    if deconvolve_cic:
        return utils.deconvolve_cic(cicfield, nn)
    return cicfield


@seren3.derived_quantity(requires=["epoch"], latex=r'Age [Gyr]')
def part_age(context, dset, **kwargs):
    #return context.quantities.tform(dset['epoch'], **kwargs)
    '''
    Return the formation (lookback) time in units of Gyr
    For formation time since the big bang (age), do age_simu - tform (also in units of Gyr)
    '''
    nml = context.nml
    isCosmoSim = ('cosmo' in nml['RUN_PARAMS'] and nml['RUN_PARAMS']['cosmo'] == '.false.')
    verbose = kwargs.get("verbose", False)
    tform = dset['epoch']

    if not isCosmoSim:
        return tform * context.info['unit_time'].express(context.C.Gyr)  # age in Gyr

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
        return (time_simu - tform) / (h0 * 1e5 / 3.08e24) / (365. * 24. * 3600. * 1e9)

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

    return output