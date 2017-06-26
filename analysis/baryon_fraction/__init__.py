import numpy as np

_DEFAULT_ALPHA = 2.
_MASS_UNIT = "Msol h**-1"

def Okamoto_Mc_fn():
    from seren3 import config
    from scipy import interpolate
    # from seren3.analysis.interpolate import extrap1d

    fname = "%s/Mc_Okamoto08.txt" % config.get('data', 'data_dir')
    data = np.loadtxt(fname)
    ok_a, ok_z, ok_Mc = data.T
    #ok_1_plus_z, ok_Mc = data.T
    #ok_z = ok_1_plus_z - 1.

    fn = interpolate.interp1d(ok_z, np.log10(ok_Mc), fill_value="extrapolate")
    return lambda z: 10**fn(z)


def interp_Okamoto_Mc(z):
    fn = Okamoto_Mc_fn()
    return fn(z)
    # fn = interpolate.InterpolatedUnivariateSpline(ok_z, np.log10(ok_Mc), k=1)  # interpolate on log mass
    # return 10**fn(z)

def compute_fb(context, return_stats=False, mass_unit="Msol h**-1"):
    '''
    Computes the baryon fraction for this container
    '''    
    part_dset = context.p[["id", "mass", "epoch"]].flatten()
    ix_dm = np.where(np.logical_and( part_dset["id"] > 0., part_dset["epoch"] == 0 ))  # index of dm particles
    ix_stars = np.where( np.logical_and( part_dset["id"] > 0., part_dset["epoch"] != 0 ) )  # index of star particles

    gas_dset = context.g["mass"].flatten()

    part_mass_tot = part_dset["mass"].in_units(mass_unit).sum()
    star_mass_tot = part_dset["mass"].in_units(mass_unit)[ix_stars].sum()
    gas_mass_tot = gas_dset["mass"].in_units(mass_unit).sum()

    tot_mass = part_mass_tot + gas_mass_tot
    fb = (gas_mass_tot + star_mass_tot)/tot_mass

    if return_stats:
        return fb, tot_mass, len(gas_dset["mass"]), len(part_dset["mass"][ix_dm])
    return fb, tot_mass


def gnedin_fitting_func(Mh, Mc, alpha=_DEFAULT_ALPHA, **cosmo):
    f_bar = cosmo["omega_b_0"] /  cosmo["omega_M_0"]
    return f_bar * (1 + (2**(alpha/3.) - 1) * (Mh/Mc)**(-alpha))**(-3./alpha)


def lmfit_gnedin_fitting_func(params, mass, data, **cosmo):
    # For use with the lmfit module

    Mc = params["Mc"].value
    alpha = params["alpha"].value
    f_bar = cosmo["omega_b_0"] /  cosmo["omega_M_0"]
    model = f_bar * (1 + (2**(alpha/3.) - 1) * (mass/Mc)**(-alpha))**(-3./alpha)
    return model - data  # what we want to minimise

def fit(mass, fb, fix_alpha, use_lmfit=True, alpha=_DEFAULT_ALPHA, **cosmo):
    import scipy.optimize as optimization

    # Make an initial guess at Mc
    cosmic_mean_b = cosmo["omega_b_0"] / cosmo["omega_M_0"]
    fb_cosmic_mean = fb/cosmic_mean_b

    idx_Mc_guess = np.abs( fb_cosmic_mean - 0.5 ).argmin()
    Mc_guess = mass[idx_Mc_guess]

    p0 = [Mc_guess]
    if fix_alpha is False:
        alpha_guess = alpha
        p0.append(alpha_guess)

    if use_lmfit:
        # Alternative least-squares fitting routine

        from lmfit import minimize, Parameters
        fit_params = Parameters()
        fit_params.add("Mc", value=p0[0], min=0.)
        fit_params.add("alpha", value=alpha, vary=np.logical_not(fix_alpha), min=0.)
        # print fit_params
        result = minimize( lmfit_gnedin_fitting_func, fit_params, args=(mass, fb), kws=cosmo)
        if result.success:
            Mc_res = result.params['Mc']
            alpha_res = result.params['alpha']
            return {"Mc" : {"fit" : Mc_res.value, "stderr" : Mc_res.stderr},\
                    "alpha" : {"fit" : alpha_res.value, "stderr" : alpha_res.stderr}}
        else:
            raise Exception("Could not fit params: %s" % result.message)
    else:
        # Curve fit
        fn = lambda *args: gnedin_fitting_func(*args, **cosmo)
        popt, pcov = optimization.curve_fit( fn, mass, fb, p0=p0, maxfev=1000 )

        # Fit
        Mc_fit = popt[0]

        # Errors
        sigma_Mc = np.sqrt(pcov[0,0])

        print "Mc = ", Mc_fit

        if fix_alpha:
            return {"Mc" : {"fit" : Mc_fit, "sigma" : sigma_Mc},\
                    "alpha" : {"fit" : alpha, "sigma" : None}}
        else:
            alpha_fit = popt[1]
            print "alpha = ", alpha_fit
            sigma_alpha = np.sqrt(pcov[1,1])

            # correlation between Mc and alpha
            corr = pcov[0,1] / (sigma_Mc * sigma_alpha)
            # print 'corr = ', corr
            return {"Mc" : {"fit" : Mc_fit, "sigma" : sigma_Mc},\
                    "alpha" : {"fit" : alpha_fit, "sigma" : sigma_alpha},\
                    "corr" : corr}
