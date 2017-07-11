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