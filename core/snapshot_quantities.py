'''
Snapshot level quantities that can be calculated, i.e volume/mass weighted averages
'''
import numpy as np
from seren3.array import SimArray

class Quantity(object):
    def __init__(self, snapshot):
        self.base = snapshot
        self._additional_quantities = {}

    def has_attr(self, key):
        return key in self._additional_quantities

    def add_attr(self, key, value):
        self._additional_quantities[key] = value

    def get_attr(self, key):
        return self._additional_quantities[key]

    @property
    def t_star(self):
        '''
        Star formation time scale in Gyr
        '''
        NML = self.base.nml.NML
        nml = self.base.nml

        if ("t_star" in nml[NML.PHYSICS_PARAMS]):
            return nml[NML.PHYSICS_PARAMS]["t_star"]

        n_star = nml[NML.PHYSICS_PARAMS]["n_star"]  # cm^-3
        eps_star = nml[NML.PHYSICS_PARAMS]["eps_star"]

        t_star=0.1635449*(n_star/0.1)**(-0.5)/eps_star  # Gyr
        return self.base.array(t_star, "Gyr")

    @property
    def rhoc(self):
        '''
        Performs CIC interpolation to compute CDM density on the simulation coarse grid in units
        kg/m^3
        '''
        from seren3.utils.cython import cic
        from seren3.utils import deconvolve_cic

        unit_l = self.base.array(self.base.info["unit_length"])
        dset = self.base.d["pos"].flatten()
        x,y,z = dset["pos"].in_units(unit_l).T
        x = np.ascontiguousarray(x); y = np.ascontiguousarray(y); z = np.ascontiguousarray(z)
        npart = len(x)
        N = 2**self.base.info['levelmin']
        L = self.base.info['boxlen']  # box units

        # Perform CIC interpolation. This supports OpenMP threading if OMP_NUM_THREADS env var is set
        rho = np.zeros(N**3)
        cic.cic(x,y,z,npart,L,N,rho)

        rho = rho.reshape((N, N, N))
        # Deconvolve CIC kernel
        print "Deconvolving CIC kernel"
        rho = deconvolve_cic(rho, N)
        print "Done"

        # Compute units
        boxmass = self.box_mass(species='cdm').in_units("kg")
        pm_mass = boxmass/npart

        boxsize = self.base.array(self.base.info['unit_length']).in_units('m')
        dx = boxsize/N

        rhoc_unit = pm_mass/dx**3
        rho *= rhoc_unit

        # Low-level C I/O routines assemble data as a contiguous, C-ordered (nvars, twotondim, ngrids) numpy.ndarray
        # Swap data => shape : (ngrids, twotondim, nvars)
        ####### WARNING : must keep C-ordered contiguity !!! #######
        return np.ascontiguousarray(np.swapaxes(rho, 0, 2))

    @property
    def deltac(self):
        '''
        Returns CDM overdensity field
        '''
        from seren3.cosmology import rho_mean_z

        cosmo = self.base.cosmo
        omega0 = cosmo['omega_M_0'] - cosmo['omega_b_0']

        rho_mean = rho_mean_z(omega0, **cosmo)

        rhoc = self.rhoc
        delta = (rhoc - rho_mean) / rho_mean

        return delta

    def rho_mean(self, species='baryon'):
        '''
        Mean density at current redshift of baryons or cdm
        '''
        from seren3.cosmology import rho_mean_z
        cosmo = self.base.cosmo
        omega_0 = 0.
        if (species == 'b') or (species == 'baryon'):
            omega_0 = cosmo['omega_b_0']
        elif (species == 'c') or (species == 'cdm'):
            omega_0 = cosmo['omega_M_0'] - cosmo['omega_b_0']
        else:
            raise Exception("Unknown species %s" % species)

        rho_mean = rho_mean_z(omega_0, **cosmo)
        return SimArray(rho_mean, "kg m**-3")

    def box_mass(self, species='baryon'):
        snap = self.base
        rho_mean = self.rho_mean(species)  # kg / m^3
        boxsize = snap.info['unit_length'].express(snap.C.m)
        mass = rho_mean * boxsize**3.  # kg
        return self.base.array(mass, "kg")

    def particle_mass(self):
        boxmass = self.box_mass(species="cdm")
        npart = (2**self.base.info["levelmin"])**3
        return boxmass.in_units("Msol h**-1") / npart

    def age_of_universe_gyr(self):
        fr = self.base.friedmann
        age_simu = fr["age_simu"]
        return SimArray(age_simu, "Gyr")

    def volume_weighted_average(self, field, mem_opt = False):
        '''
        Computes the volume weighted average for the desired field
        '''
        boxsize = SimArray(self.base.info["boxlen"], self.base.info["unit_length"]).in_units("pc")
        if mem_opt:  # slower
            vsum = 0.
            for dset in self.base.g[[field, 'dx']]:
                dx = dset["dx"].in_units("pc")
                vsum += np.sum(dset[field] * dx**3)

            return vsum / boxsize**3
        else:
            dset = self.base.g[[field, 'dx']].flatten()
            dx = dset["dx"].in_units("pc")
            return np.sum(dset[field] * dx**3) / boxsize**3

    def mass_weighted_average(self, field, mem_opt = False):
        '''
        Computes the mass weighted average for the desired field
        '''
        snap = self.base
        boxmass = self.box_mass('b').in_units("Msol")
        if mem_opt:
            msum = 0.
            for dset in snap.g[[field, 'mass']]:
                msum += np.sum(dset[field] * dset['mass'].in_units("Msol"))

            return (msum / boxmass)
        else:
            dset = snap.g[[field, 'mass']].flatten()
            return np.sum(dset[field] * dset['mass'].in_units("Msol")) / boxmass

            