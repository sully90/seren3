########################################## HALOS ##########################################
def amr_halo_average_qty(context, qty, lengh_unit="pc", halo=None):
    '''
    (Volume) Average a given quantity in the desiered halos virial sphere
    '''
    import numpy as np

    if (halo is None):
        # As an example, here we load a halo catalogue made with rockstar and get
        # the most massive halo from it
        halos = context.halos(finder="rockstar")  # fiinder = "ahf", "rockstar" or "ctrees" (consistent trees)
        halo = halos.sort("Mvir")[0]  # sorts from massive -> small

    # There is a small bug at the moment where the SimArray conversion context (which contains)
    # the scale factor and hubble param.) is not always propagated when operating on the array.
    # So to keep correct unit information for halo volumes, we have to do the following
    halo_volume = halo.sphere.get_volume()
    halo_volume = context.array(halo_volume, halo_volume.units)  # now this array can convert units involving a and h

    dset = halo.g[[qty, "dx"]].flatten()  # load our field and cell-width
    field = dset[qty]
    dx = dset["dx"].in_units(lengh_unit)
    vol = halo_volume.in_units("%s**3" % lengh_unit)

    return np.sum(field * dx**3) / vol

########################################## GAS ##########################################
def spherical_profile(context, field):
    '''
    Made a spherical profile of this container (can be a snapshot, halo or a subsnap)
    '''
    import matplotlib.pylab as plt

    npoints = int(1e6)  # number of sampling points, as the AMR grid is sampled uniformly. it doesn't like XeX syntax, so just parse with int
    nbins=50  # number of bins for the profile
    divide_by_counts = True  # True for AMR profiles, False for particle profiles

    prof, r_bins = context.g.bin_spherical(field, npoints, nbins, divide_by_counts=divide_by_counts)

    # plot it
    prof_unit_latex = prof.units.latex()
    r_bins_latex = r_bins.units.latex()

    plt.loglog(r_bins, prof, linewidth=2.)
    plt.xlabel(r"radius [$%s$]" % r_bins_latex)
    plt.xlabel(r"profile [$%s$]" % prof_unit_latex)

    plt.show()

def splatter_projection_intensive(context):
    '''
    Make a projection of the desired (intensive) field using the splatter engine
    '''
    from seren3.analysis.visualization import engines, operators

    # density is an intensive variable, but the splatter engine requires
    # extensive variables, therefore we use a mass-weighted operator
    field = "rho"
    unit = context.C.kg * context.C.m**-3  # pymses units

    op = operators.MassWeightedOperator("rho", unit)

    # a MassWeightedSpatterEngine already exists, but for this example I demonstrate how to use the custom engine
    # CustomEngines require you set the above operator, which (as above), need to know the correct unit
    engine = engines.CustomSplatterEngine(context.g, field, op)  # gas - context.g

    camera = context.camera()  # the camera object specifying the region we want to image

    # We can make a higher resolution image by modifying the map_max_size param of our camera object
    # this should be a power of 2
    camera.map_max_size = 2048

    projection = engine.process(camera)

    # this function takes args such as colormap etc if you desire.
    # just run help(projection.save_plot()) to see available options
    im = projection.save_plot()
    im.show()  # show the plot

def raytracing_projection_extensive(context):
    '''
    Make a projection of the desired (extensive) field using the raytracer
    '''
    from seren3.analysis.visualization import engines

    # this is much simpler than the above
    field = "nH"  # derived field this time

    # the engine sets the operator and unit for us
    engine = engines.RayTraceEngine(context.g, field)

    camera = context.camera()
    camera.map_max_size = 2048

    # The syntax is the same...
    projection = engine.process(camera)

    # this function takes args such as colormap etc if you desire.
    # just run help(projection.save_plot()) to see available options
    im = projection.save_plot()
    im.show()  # show the plot

def raytracing_projection_extensive_simple(context):
    '''
    There is a shortcut for simple projections on the Family object
    '''
    from seren3.analysis.visualization import EngineMode  # enum that lets us choose projection mode

    projection = context.g.projection("nH", mode=EngineMode.RAYTRACING)
    im = projection.save_plot()
    im.show()  # show the plot


def compute_fb(context, mass_unit="Msol h**-1"):
    '''
    Computes the baryon fraction for this container
    '''
    import numpy as np
    
    part_dset = context.p[["id", "mass", "epoch"]].flatten()
    ix_dm = np.where(np.logical_and( part_dset["id"] > 0., part_dset["epoch"] == 0 ))  # index of dm particles
    ix_stars = np.where( np.logical_and( part_dset["id"] > 0., part_dset["epoch"] != 0 ) )  # index of star particles

    gas_dset = context.g["mass"].flatten()

    part_mass_tot = part_dset["mass"].in_units(mass_unit).sum()
    star_mass_tot = part_dset["mass"].in_units(mass_unit)[ix_stars].sum()
    gas_mass_tot = gas_dset["mass"].in_units(mass_unit).sum()

    tot_mass = part_mass_tot + gas_mass_tot
    fb = (gas_mass_tot + star_mass_tot)/tot_mass

    return fb, tot_mass

########################################## STARS ##########################################
# from seren3.analysis.stars.__init__.py
def sfr(context, ret_sSFR=False, nbins=100, **kwargs):
    '''
    Compute the (specific) star formation rate within this context.
    '''
    import numpy as np
    from seren3.array import SimArray
    from seren3.exceptions import NoParticlesException

    dset = context.s[["age", "mass"]].flatten()
    age = dset["age"].in_units("Gyr")
    mass = dset["mass"].in_units("Msol")

    if len(age) == 0 or len(mass) == 0:
        raise NoParticlesException("No particles found while computing SFR", 'analysis/stars/sfr')

    def compute_sfr(age, mass, nbins=nbins, **kwargs):
        agerange = kwargs.pop('agerange', [age.min(), age.max()])
        binnorm = SimArray(1e-9 * nbins / (agerange[1] - agerange[0]), "yr**-1")

        weights = mass * binnorm

        sfrhist, bin_edges = np.histogram(age, weights=weights, bins=nbins, range=agerange, **kwargs)

        binmps = np.zeros(len(sfrhist))
        binsize = np.zeros(len(sfrhist))
        for i in np.arange(len(sfrhist)):
            binmps[i] = np.mean([bin_edges[i], bin_edges[i + 1]])
            binsize[i] = bin_edges[i + 1] - bin_edges[i]

        return SimArray(sfrhist, "Msol yr**-1"), SimArray(binmps, "Gyr"), SimArray(binsize, "Gyr")

    sfrhist, lookback_time, binsize = compute_sfr(age, mass, **kwargs)
    SFR = sfrhist.in_units("Msol Gyr**-1")

    if ret_sSFR:
        SFR /= mass.sum()  # specific star formation rate

    SFR.set_field_latex("$\\mathrm{SFR}$")
    lookback_time.set_field_latex("$\\mathrm{Lookback-Time}$")
    binsize.set_field_latex("$\Delta$")
    return SFR, lookback_time, binsize  # SFR [Msol Gyr^-1] (sSFR [Gyr^-1]), Lookback Time [Gyr], binsize [Gyr]

def test_sfr(context):
    '''
    Test SFR calculation is working correctly by integrating and comparing to total stellar mass
    '''
    import seren3
    import numpy as np
    from scipy import integrate

    snap_sfr, lbtime, bsize = sfr(context)  # compute the global star formation history using the star particles

    dset = context.s["mass"].flatten()  # load stellar mass dset
    mstar_tot = dset["mass"].in_units("Msol").sum()
    integrated_mstar = integrate.trapz(snap_sfr, lbtime)  # integrate over history of the Universe

    print mstar_tot, integrated_mstar
    # Assert the integrated star formation history is close to the total stellar mass
    assert(np.allclose(mstar_tot, integrated_mstar, rtol=1e-1)), "Error: Integrated stellar mass not close to actual."
    print "Passed"

########################################## CDM ##########################################

def splatter_projection_extensive(context):
    '''
    Make a projection of the desired (extensive) field using the splatter engine
    '''
    from seren3.analysis.visualization import engines

    # this is a lot easier
    field = "mass"

    # the engine sets the operator and unit for us
    engine = engines.SplatterEngine(context.d, field)  # dark matter - context.d

    camera = context.camera()  # the camera object specifying the region we want to image

    # We can make a higher resolution image by modifying the map_max_size param of our camera object
    # this should be a power of 2
    camera.map_max_size = 2048

    projection = engine.process(camera)

    # this function takes args such as colormap etc if you desire.
    # just run help(projection.save_plot()) to see available options
    im = projection.save_plot()
    im.show()  # show the plot

def splatter_projection_extensive_simple(context):
    '''
    There is a shortcut for simple projections on the Family object
    '''
    from seren3.analysis.visualization import EngineMode  # enum that lets us choose projection mode

    projection = context.d.projection("mass", mode=EngineMode.SPLATTER)
    im = projection.save_plot()
    im.show()  # show the plot

def rhoc_cic(context):
    '''
    Performs CIC interpolation to compute CDM density on the simulation coarse grid in units
    kg/m^3
    '''
    import numpy as np
    from seren3.utils.cython import cic
    from seren3.utils import deconvolve_cic

    unit_l = context.array(context.info["unit_length"])  # the code unit length
    dset = context.d["pos"].flatten()  # load the dset
    x,y,z = dset["pos"].in_units(unit_l).T  # separate positions into x,y,z (in code length units)
    x = np.ascontiguousarray(x); y = np.ascontiguousarray(y); z = np.ascontiguousarray(z)  # cic requires c contiguous arrays

    npart = len(x)
    N = 2**context.info['levelmin']
    L = context.info['boxlen']  # box units

    # Perform CIC interpolation. This supports OpenMP threading if OMP_NUM_THREADS env var is set
    rho = np.zeros(N**3)  # init empty grid
    cic.cic(x,y,z,npart,L,N,rho)  # do the CIC. openmp enabled, set env. var OMP_NUM_THREADS

    rho = rho.reshape((N, N, N))
    # Deconvolve CIC kernel
    print "Deconvolving CIC kernel"
    rho = deconvolve_cic(rho, N)  # deconvolve the CIC kernel to recover small-scale power
    print "Done"

    # Compute units
    boxmass = context.quantities.box_mass(species='cdm').in_units("kg")  # total box mass
    pm_mass = boxmass/npart  # particle mass

    boxsize = context.array(context.info['unit_length']).in_units('m')
    dx = boxsize/N  # cell width at levelmin

    rhoc_unit = pm_mass/dx**3
    rho *= rhoc_unit

    # Low-level C I/O routines assemble data as a contiguous, C-ordered (nvars, twotondim, ngrids) numpy.ndarray
    # Swap data => shape : (ngrids, twotondim, nvars)
    ####### WARNING : must keep C-ordered contiguity !!! #######
    return np.ascontiguousarray(np.swapaxes(rho, 0, 2))

def deltac_cic(context, rhoc=None):
    '''
    Returns CDM overdensity field
    '''
    from seren3.cosmology import rho_mean_z

    cosmo = context.cosmo  # get the cosmological context - this is a dictionary
    omega0 = cosmo['omega_M_0'] - cosmo['omega_b_0']  # CDM density param.

    rho_mean = rho_mean_z(omega0, **cosmo)  # mean (CDM) density at this redshift

    if (rhoc is None):
        rhoc = rhoc_cic(context)
    delta = (rhoc - rho_mean) / rho_mean  # the CDM overdensity

    return delta