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

