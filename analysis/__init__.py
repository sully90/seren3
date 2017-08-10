def volume_mass_weighted_average(snapshot, field, dset):
    '''
    Computes volume and mass weighted averages of the desired AMR field
    '''
    import numpy as np

    length_unit = "pc"
    mass_unit = "Msol"
    boxmass = snapshot.quantities.box_mass('b').in_units(mass_unit)
    boxsize = snapshot.array(snapshot.info["boxlen"], snapshot.info["unit_length"]).in_units(length_unit)

    vsum = 0.
    msum = 0.
    dx = dset["dx"].in_units(length_unit)
    mass = dset["mass"].in_units(mass_unit)

    vsum += np.sum(field * dx**3)
    msum += np.sum(field * mass)

    vw = vsum / boxsize**3
    mw = msum / boxmass

    return vw, mw

def mean_T_rho_mean(snapshot):
    '''
    Computes mean (volume and mass weighted) temperature at the mean density
    '''
    import numpy as np
    from seren3.cosmology import rho_mean_z
    from seren3.utils import approx_equal

    cosmo = snapshot.cosmo
    rho_mean = snapshot.array( rho_mean_z(cosmo["omega_b_0"], **cosmo), "kg m**-3" )

    T_sum = 0.
    count = 0

    for dset in snapshot.g[["rho", "T"]]:
        ix = np.where(approx_equal(rho_mean, dset['rho'], tol=None, rel=1e-5))
        if (len(ix[0]) > 0):
            T_sum+=dset['T'][ix].mean()
            count += len(ix[0])

    return T_sum/float(count)