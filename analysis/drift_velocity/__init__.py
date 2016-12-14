"""
Utility functions to include drift velocity in grafIC ics by computing/convolving density
power spectrum k dependent bias. Contains routines to run CICsASS
"""
import numpy as np

def vbc_rms(vbc_field):
    '''
    Computes the rms vbc in the box
    '''
    rms = np.sqrt(np.mean(vbc_field ** 2))
    return rms


def vbc_ps_fname(rms, z, boxsize):
    import os
    cwd = os.getcwd()
    if not os.path.isdir("%s/vbc_TFs_out" % cwd):
        os.mkdir("%s/vbc_TFs_out" % cwd)
    return '%s/vbc_TFs_out/vbc_%f_z%f_B%1.2f.dat' % (cwd, rms, z, boxsize)


def run_cicsass(boxsize, z, rms_vbc_z1000, out_fname, N=256):
    import subprocess
    from seren3.utils import which

    exe = which('transfer.x')

    if exe is None:
        raise Exception("Unable to locate transfer.x executable")

    # Example execution for RMS vbc=30km/s @ z=1000.:
    # ./transfer.x -B0.2 -N128 -V30 -Z100 -D3 -SinitSB_transfer_out

    # Run with N=256
    CICsASS_home = "/lustre/scratch/astro/ds381/CICsASS/matt/Dropbox/CICASS/vbc_transfer/"
    cmd = 'cd %s && %s -B%1.2f -N%d -V%f -Z%f -D3 -SinitSB_transfer_out > %s' % (
        CICsASS_home, exe, boxsize, N, rms_vbc_z1000, z, out_fname)
    # print 'Running:\n%s' % cmd
    # Run CICsASS and wait for output
    code = subprocess.check_call(cmd, shell=True)
    if code != 0:
        raise Exception("CICsASS returned non-zero exit code: %d", code)
    return code


def compute_bias(ics, vbc):
    """ Calculate the bias to the density power spectrum assuming
    COHERENT vbc at z=1000. """
    import os, time
    from seren3.array import SimArray
   
    # Compute size of grid and boxsize (for this patch)
    N = vbc.shape[0]
    boxsize = ics.boxsize.in_units("Mpc a h**-1") * (float(N) / float(ics.header.N))

    # Compute vbc @ z=1000
    z = ics.z
    rms = vbc_rms(vbc)
    rms_recom = rms * (1001./z)

    # Check for PS and run CICsASS if needed
    fname_vbc0 = vbc_ps_fname(0., z, boxsize)
    if not os.path.isfile(fname_vbc0):
        exit_code = run_cicsass(boxsize, z, 0., fname_vbc0)

    fname_vbcrecom = vbc_ps_fname(rms_recom, z, boxsize)
    if not os.path.isfile(fname_vbcrecom):
        exit_code = run_cicsass(boxsize, z, rms_recom, fname_vbcrecom)

    # Load power spectra and compute bias
    ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
    ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)

    # Should have same lenghts if finished writing
    count = 0
    while len(ps_vbcrecom[1]) != len(ps_vbc0[1]):
        count += 1
        if count > 10:
            raise Exception("Reached sleep limit. Filesizes still differ")
        time.sleep(5)
        ps_vbc0 = np.loadtxt(fname_vbc0, unpack=True)
        ps_vbcrecom = np.loadtxt(fname_vbcrecom, unpack=True)

    #CDM bias
    b_cdm = ps_vbcrecom[1] / ps_vbc0[1]
    # Baryon bias
    b_b = ps_vbcrecom[2] / ps_vbc0[2]
    # Wavenumber
    k_bias = SimArray(ps_vbcrecom[0] / ics.cosmo["h"], "h Mpc**-1")

    return k_bias, b_cdm, b_b
