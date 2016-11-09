'''
Functions to read files (i.e ramses binaries)
'''
import numpy as np
import scipy.fftpack as fft

def read_c2ray_vel(vel_fname, dens_fname):
    '''
    Helper method to invoke c2raytools and read velocity field in correct units
    '''
    from c2raytools.vel_file import VelocityFile
    vel_file = VelocityFile(filename=vel_fname)
    vel = vel_file.get_kms_from_density(dens_fname)
    return vel


def read_c2ray_vel_mag(vel_fname, dens_fname):
    vel = read_c2ray_vel(vel_fname, dens_fname)
    return np.sqrt(vel[0, :] * vel[0, :] + vel[1, :] * vel[1, :] + vel[2, :] * vel[2, :])


def skip(f):
    '''
    Skip a 4-byte record in an unformatted fortran90 binary file
    '''
    return np.fromfile(file=f, dtype=np.int32, count=1)


def read_dimensions(f):
    '''
    Read nx, ny, nz from unformatted fortran90 ramses binary file
    '''
    nn = np.zeros(3)
    nn = np.fromfile(file=f, dtype=np.int32, count=3)
    nx = np.int64(nn[0])
    ny = np.int64(nn[1])
    nz = np.int64(nn[2])
    return nx, ny, nz


def read_binary_cube(fname, vector=False, reshape=False, unformatted=True, astype=None):

    with open(fname, 'rb') as f:
        # Read the header
        if(unformatted):
            skip(f)
        nx, ny, nz = read_dimensions(f)
        if(unformatted):
            skip(f)

        # Read the data
        ncount = nx * ny * nz
        if vector:
            ncount *= 3
        data = np.zeros(ncount)
        if(unformatted):
            skip(f)
        if astype is not None:
            data = np.fromfile(
                file=f, dtype=np.float32, count=ncount).astype(astype)
        else:
            data = np.fromfile(
                file=f, dtype=np.float32, count=ncount).astype(astype)
        if(unformatted):
            skip(f)

        if reshape:
            if vector:
                data = np.reshape(data, [nx, ny, nz, 3])
            else:
                data = np.reshape(data, [nx, ny, nz])
        return nx, ny, nz, data


def read_binary_map(fname, reshape=False, unformatted=True, astype=None):

    with open(fname, 'rb') as f:
        # Read the header
        if (unformatted):
            skip(f)
        nn = np.fromfile(file=f, dtype=np.int32, count=2)
        nx, ny = nn
        if(unformatted):
            skip(f)

        ncount = nx * ny

        # Read the data
        data = np.zeros(ncount)
        if(unformatted):
            skip(f)
        if astype is not None:
            data = np.fromfile(
                file=f, dtype=np.float32, count=ncount).astype(astype)
        else:
            data = np.fromfile(
                file=f, dtype=np.float32, count=ncount).astype(astype)
        if(unformatted):
            skip(f)

        if reshape:
            data = np.reshape(data, [nx, ny])
        return nx, ny, data
