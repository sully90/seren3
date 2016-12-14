import numpy as np
import seren3
from seren3.array import SimArray
from seren3.utils.f90.fortranfile import FortranFile

_derived_field_registry = {}

def derived_quantity():
    def wrap(fn):
        _derived_field_registry[fn.__name__] = fn
        return fn
    return wrap

class GrafICHeader(object):
    """
    Class to read/store header information
    """
    def __init__(self, level_dir):

        with self._open_file(level_dir) as f:
            nn, dx, origin, cosmo = self.read_header(f)

            self.N = nn[0]
            self.nn = nn
            self.dx = dx
            self.origin = origin
            self.cosmo = cosmo

        self.dx *= self.cosmo['h']  # Mpccm/h
        self.dx = SimArray(self.dx, "Mpc a h**-1", snapshot=self)
        self.boxsize = SimArray(self.N * self.dx, "Mpc a h**-1", snapshot=self)  # Mpccm/h
        self.cosmo['z'] = (1. / self.cosmo["aexp"]) - 1.

    @staticmethod
    def _open_file(level_dir):
        import glob
        fname = glob.glob("%s/ic_*" % level_dir)[0]
        return FortranFile(fname)

    @staticmethod
    def _read_dummy(f):
        d = np.fromfile(f, dtype=np.int32, count=1)
        assert(d == 44)

    @staticmethod
    def read_header(f):
        import glob

        nn = dx = origin = cosmo = None

        # Skip dummp
        GrafICHeader._read_dummy(f)
        (np1, np2, np3) = np.fromfile(f, dtype=np.int32, count=3)
        (dx, x0, y0, z0, aexp, omegam, omegal, H0) = np.fromfile(
            f, dtype=np.float32, count=8)
        GrafICHeader._read_dummy(f)

        nn = np.array([np1, np2, np3])
        origin = np.array([x0, y0, z0])
        cosmo = {'omega_M_0': omegam, 'omega_lambda_0': omegal,
                 'aexp': aexp, 'h': H0 / 100.}

        return nn, dx, origin, cosmo


class GrafICSnapshot(object):
    """
    Class for handing grafIC initial conditions
    """
    def __init__(self, path, level, set_fft_sample_spacing=True, **kwargs):
        from pymses.utils import constants as C

        self.path = path
        self.level = level

        self.header = GrafICHeader(self.level_dir)
        self.header.cosmo["omega_b_0"] = kwargs.pop("omega_b_0", 0.045)
        self.boxsize = self.header.boxsize
        self.dx = self.header.dx
        self.z = self.header.cosmo["z"]
        self.cosmo = self.header.cosmo
        self.C = C

        if set_fft_sample_spacing:
            self.set_fft_sample_spacing()

    def __getitem__(self, field):
        """
        Read initial condition map and return data
        """
        import os

        if os.path.isfile("%s/ic_%s" % (self.level_dir, field)):
            print 'Found field %s on disk - loading' % field
            # TODO -> Parallel load
            fname = self.field_fname(field)

            f = FortranFile(fname)
            self.header.read_header(f)

            (np1, np2, NP3) = self.header.nn
            field = np.zeros((NP3, np2, np1))
            # field = np.zeros((np1, np2, NP3))

            for i1 in range(np1):
                # same ordering as CIC smoothed fields
                field[i1, :, :] = f.readReals().reshape(np2, NP3)

                # same ordering as header
                # field[:, :, i3] = f.readReals().reshape(np1, np2)

            return field
        elif field in _derived_field_registry:
            print 'Deriving field %s' % field
            fn = _derived_field_registry[field]
            return fn(self)
        else:
            raise Exception("Unable to load field: %s" % field)

    def lazy_load_periodic(self, field, origin, N):
        '''
        Partial load a field from disk, with periodic boundary conditions
        '''
        import os

        fname = self.field_fname(field)
        if not os.path.isfile(fname):
            raise IOError("Could not locate field %s on disk." % field)

        ff = FortranFile(fname)
        data = np.zeros((N, N, N))
        (np1, np2, np3) = self.header.nn

        # Fast-forward to lowest x slice within box
        header_size = 44 + 8  # 8 bytes for header and footer block
        bytes_per_slice = long((np2 * np3) * 4)

        xx = 0
        for i1 in range(origin[0], origin[0] + N):
            # Seek to slice
            x = int(i1 % self.header.N)
            pos = long(header_size + ((bytes_per_slice + 8) * x))
            ff.seek(pos)

            # Read the slice and place it in the subpatch
            slc = ff.readReals().reshape(np2, np3)
            yy = 0
            for i2 in range(origin[1], origin[1] + N):
                zz = 0
                for i3 in range(origin[2], origin[2] + N):
                    y = int(i2 % self.header.N)
                    z = int(i3 % self.header.N)
                    data[xx, yy, zz] = slc[y, z]
                    zz+=1
                yy+=1
            xx+=1
        ff.close()
        return data

    def rho_mean(self, species):
        '''
        Compute mean density of given species at current redshift
        '''
        from seren3 import cosmology
        cosmo = self.header.cosmo

        omega0 = 0.
        if 'b' == species:
            omega0 = cosmo['omega_b_0']
        elif 'c' == species:
            omega0 = cosmo['omega_M_0'] - cosmo['omega_b_0']
        else:
            raise Exception("Unknown species: %s" % species)
        rho_mean = SimArray(cosmology.rho_mean_z(omega0, **cosmo), "kg m**-3")
        return rho_mean

    @property
    def particle_mass(self):
        '''
        Compute DM particle mass
        '''
        rho_mean = self.rho_mean('c')
        boxsize = self.boxsize.in_units("m")

        box_mass = rho_mean * boxsize**3  # kg
        return box_mass / float(self.header.N**3)

    @property
    def level_dir(self):
        return "%s/level_%03i/" % (self.path, self.level)

    def field_fname(self, field):
        return '%s/ic_%s' % (self.level_dir, field)

    def set_fft_sample_spacing(self):
        from seren3.cosmology import _power_spectrum
        self.kx, self.ky, self.kz = _power_spectrum.fft_sample_spacing_components(
            self.header.N)
        fac = (2. * np.pi / self.boxsize)
        self.k = np.sqrt(self.kx ** 2. + self.ky ** 2. + self.kz ** 2.) * fac

################################## WRITING #################################################

    def _write_header(self, f):
        '''
        Write a grafIC header to f
        '''
        (np1, np2, np3) = self.header.nn
        nn = np.array([np1, np2, np3], dtype=f.ENDIAN + 'i')
        dx = self.header.dx.in_units("Mpc a")
        origin = self.header.origin
        cosmo = self.header.cosmo

        lbytes = 44

        f._write_check(lbytes)
        f.write(nn.tostring())
        f.write(np.array([dx, origin[0], origin[1], origin[2], cosmo['aexp'], cosmo['omega_M_0'],
                          cosmo['omega_lambda_0'], cosmo['h'] * 100.], dtype=f.ENDIAN + 'f').tostring())
        f._write_check(lbytes)

    def write_field(self, data, field_name, **kwargs):
        # Open file for writing
        assert(all(data.shape == self.header.nn))

        fname = self.field_fname(field_name)
        ff = FortranFile(fname, '@', 'i', 'w')
        self._write_header(ff)

        # Write the field data
        (np1, np2, np3) = data.shape
        for i1 in range(np1):
            tmp = data[i1, :, :].flatten()
            ff.writeReals(tmp)


################################## grafIC derived fields ######################################

@derived_quantity()
def zeldovich_offset(ics):
    '''
    Computes the Zel'Dovich offset for DM particles
    '''
    from seren3 import cosmology

    cosmo = ics.header.cosmo
    lingrowthfac = cosmology.lingrowthfac(cosmo['z'], **cosmo)
    bdot_by_b = cosmology.rate_linear_growth(**cosmo) / lingrowthfac  # km Mpc^-1 s^-1

    a = cosmo['aexp']

    vx, vy, vz = (ics['velcx'].flatten(), ics['velcy'].flatten(), ics['velcz'].flatten())
    vel = np.array([vx, vy, vz]).T

    offset = vel / (a * bdot_by_b)
    offset *= cosmo['h']
    return offset  # Mpc / a / h

@derived_quantity()
def posc(ics):
    '''
    Compute DM particle positions using the Zel'Dovich approx.
    '''
    from pynbody import _util

    N = ics.header.N
    slc = slice(0, N**3, 1)

    pos = _util.grid_gen(slc, N, N, N)
    pos *= ics.header.dx * ics.header.N

    offset = ics['zeldovich_offset']
    posc = pos + offset

    shape = posc.shape
    posc = posc.flatten()
    idx = np.where(posc < 0.)
    posc[idx] = ics.boxsize + posc[idx]

    idx = np.where(posc > ics.boxsize)
    posc[idx] = posc[idx] - ics.boxsize

    posc = posc.reshape(shape)
    return posc


@derived_quantity()
def rhoc(ics):
    from seren3.utils.cython import cic

    pos = ics['posc']
    x = np.ascontiguousarray(pos.T[0])
    y = np.ascontiguousarray(pos.T[1])
    z = np.ascontiguousarray(pos.T[2])

    N = ics.header.N
    L = ics.header.boxsize
    nPart = N**3
    rho = np.zeros(nPart)

    # cic.cic(np.ascontiguousarray(x), np.ascontiguousarray(y),\
         # np.ascontiguousarray(z), nPart, L, N, rho)

    cic.cic(x, y, z, nPart, L, N, rho)
    rho = rho.reshape(ics.header.nn)  # number per dx**3

    part_mass = ics.particle_mass.in_units("kg")
    dx = ics.header.dx.in_units("m")
    rho *= (part_mass / dx**3)  # kg / m^3

    # return np.ascontiguousarray(np.swapaxes(rho, 0, 2))
    return rho

@derived_quantity()
def deltac(ics):
    '''
    DM overdensity
    '''
    rhoc = ics['rhoc']  # kg/m^3
    rho_mean = ics.rho_mean('c')  # kg/m^3

    delta = (rhoc - rho_mean) / rho_mean
    return delta


@derived_quantity()
def rhob(ics):
    deltab = ics["deltab"]
    rho_mean = ics.rho_mean('b')

    rhob = (deltab * rho_mean) + rho_mean
    return rhob


@derived_quantity()
def velb(ics):
    vx, vy, vz = (ics['velbx'], ics['velby'], ics['velbz'])
    return np.sqrt(vx**2 + vy**2 + vz**2)

@derived_quantity()
def velc(ics):
    vx, vy, vz = (ics['velcx'], ics['velcy'], ics['velcz'])
    return np.sqrt(vx**2 + vy**2 + vz**2)

@derived_quantity()
def vbc(ics):
    return ics["velb"] - ics["velc"]