def intep_sed(SED, lambda_A, age, Z):
    '''
    Interpolate a SED for the given wavelengts at fixed age and metallicity
    '''
    import numpy as np
    from seren3.analysis.interpolate import interpolate3d

    agebins, zbins, Ls, SED_grid = SED
    age = np.ones(len(lambda_A)) * age
    Z = np.ones(len(lambda_A)) * Z

    # should age and z be opposite way around?
    return interpolate3d(lambda_A, age, Z, Ls, agebins, zbins, SED_grid)