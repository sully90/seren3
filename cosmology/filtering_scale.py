'''
Functions to compute the Jeans and Filtering scales and masses.
Note: Filtering scale integral is valid during matter domination only
'''


import numpy as np

def cs_rho_mean(fname):
    import pickle
    from seren3.array import SimArray

    data = pickle.load( open(fname, "rb") )

    aexp = np.zeros(len(data))
    cs = SimArray( np.zeros(len(data)), "m s**-1" )
    rho_mean = SimArray( np.zeros(len(data)), "kg m**-3" )

    for i in range(len(data)):
        res = data[i].result

        aexp[i] = res["aexp"]
        cs[i] = res["mw"].in_units(cs.units)
        rho_mean[i] = res["rho_mean"].in_units(rho_mean.units)

    return aexp, cs, rho_mean

def jeans_scale(aexp, cs, rho):
    from seren3.array import SimArray
    from pymses.utils import constants as C

    if (not isinstance(cs, SimArray) or not isinstance(rho, SimArray)):
        raise Exception("cs and rho must be SimArrays")

    G = SimArray(C.G)
    kJ = ( (2. * np.pi * aexp)/cs ) * np.sqrt( (G * rho) / np.pi )

    return kJ

def jeans_scale_interp_fn(aexp, jeans_scale):
    from scipy import interpolate
    from seren3.analysis.plots import fit_scatter

    bc, mean, std = fit_scatter(np.log10(aexp), np.log10(jeans_scale), nbins=10)
    fn = interpolate.interp1d(bc, mean, fill_value="extrapolate")
    return fn

def jeans_mass(aexp, rho_mean, kJ):
    jeans_mass = (4.*np.pi/3.) * rho_mean * (np.pi * aexp / kJ)**3
    return jeans_mass.in_units("Msol")

# def _filtering_scale_integrand(aprime, aexp, kJ_fn):
#     kJ = 10**kJ_fn(np.log10(aprime))
#     I = kJ**-2 * (1. - np.sqrt(aprime/aexp))
#     if (aprime > aexp):
#         print aprime, aexp
#     return I

# def filtering_scale(aexp, kJ):
#     import scipy.integrate
#     from seren3.array import SimArray

#     kJ_fn = jeans_scale_interp_fn(aexp, kJ)
#     kF = SimArray(np.zeros(len(aexp)), "m**-1")

#     for i in range(1, len(aexp)+1):
#         I = (3./aexp[-i]) * scipy.integrate.quad(
#                     _filtering_scale_integrand, 0., aexp[-i], (aexp[-i], kJ_fn))[0]
#         if (I < 0):
#             print I, aexp[-i], i
#             break

#         kF[i-1] = np.sqrt(1./I)

#     return kF[::-1]

def filtering_scale(aexp, kJ):
    '''
    Computes filtering scale by integrating equation 1.20 in thesis, using the 
    trapezium rule
    '''
    from scipy.integrate import trapz
    from seren3.array import SimArray

    kJ_fn = jeans_scale_interp_fn(aexp, kJ)
    kJ_interp = 10**kJ_fn(np.log10(aexp))

    kF = SimArray(np.zeros(len(aexp)), "m**-1")

    # integrand = (3./aexp[-1]) * kJ**-2 * (1. - np.sqrt(aexp/aexp[-1]))
    for i in range(1, len(aexp)+1):
        # integrand = (3./aexp[-i]) * kJ_intep**-2 * (1. - np.sqrt(aexp/aexp[-i]))
        integrand = (3./aexp[-i]) * kJ_interp[::-1][i:][::-1]**-2 * (1. - np.sqrt(aexp[::-1][i:][::-1]/aexp[-i]))
        integrand = trapz(integrand, aexp[::-1][i:][::-1])
        kF[i-1] = np.sqrt(1./integrand)

    return kF[::-1]

def plot_filtering_mass(sims, labels, cols):
    from matplotlib import rcParams
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14

    rcParams['axes.labelsize'] = 18
    rcParams['xtick.major.pad'] = 10
    rcParams['ytick.major.pad'] = 10
    import matplotlib.pylab as plt
    from seren3.array import SimArray

    fig, ax = plt.subplots(figsize=(8,8))
    sub_axes = plt.axes([.5, .5, .35, .35]) 

    for sim, label, c in zip(sims, labels, cols):
        pickle_path = "%s/pickle/" % sim.path
        fname = "%s/cs_time_averaged_at_rho_mean.p" % pickle_path

        aexp, cs, rho_mean = cs_rho_mean(fname)
        kJ = jeans_scale(aexp, cs, rho_mean)
        kF = filtering_scale(aexp, kJ)

        kJ_fn = jeans_scale_interp_fn(aexp, kJ)
        kJ_interp = SimArray(10**kJ_fn(np.log10(aexp)), "m**-1")

        jeans_mass = (4.*np.pi/3.) * rho_mean * (np.pi * aexp / kJ_interp)**3
        filtering_mass = (4.*np.pi/3.) * rho_mean * (np.pi * aexp / kF)**3
        filtering_mass.convert_units("Msol")
        jeans_mass.convert_units("Msol")

        z = (1./aexp) - 1.

        ax.semilogy(z, filtering_mass, linewidth=2., label=label, color=c)
        ax.semilogy(z, jeans_mass, linewidth=2., color=c, linestyle="--")

        ix = np.where( np.logical_and(z >= 6, z <= 10) )
        sub_axes.semilogy(z[ix], filtering_mass[ix], linewidth=2., color=c)
        sub_axes.semilogy(z[ix], jeans_mass[ix], linewidth=2., color=c, linestyle="--")

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"M$_{F,\mathrm{J}}$ [M$_{\odot}$]")

    sub_axes.set_xlabel(r"$z$")
    sub_axes.set_ylabel(r"M$_{F,\mathrm{J}}$ [M$_{\odot}$]")

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125),
              frameon=False, ncol=2, prop={"size" : 16})

    plt.show()