import numpy as np
from seren3.utils.io import skip

def read_seds_from_lists(seddir, nGroups=3, nIons=3):
    '''
    Read SEDs from .list files
    '''
    # First, load bins to get nAge and nZ
    from seren3.array import SimArray

    fname = "%s/SEDtable%d.list" % (seddir, 1)
    nAge = nZ = 0
    with open(fname, 'r') as f:
        l = f.readline().split()
        nAge = int(l[0])
        nZ = int(l[1])

    agebins = zbins = None

    # Read from .list group files
    nv = 3+2*nIons  # L,Lacc,egy,nions*(csn,egy)
    SEDs = np.zeros((nAge, nZ, nGroups, nv))

    for igroup in range(nGroups):
        count = 0
        fname = "%s/SEDtable%d.list" % (seddir, igroup+1)
        data = np.loadtxt(fname, skiprows=1)

        if agebins is None or zbins is None:
            agebins = []; zbins = []
            for a in data.T[0]:
                if a not in agebins:
                    agebins.append(a)
            for z in data.T[1]:
                if z not in zbins:
                    zbins.append(z)
            agebins = np.array(agebins)
            zbins = np.array(zbins)

        for j in range(nZ):
            for i in range(nAge):
                SEDs[i, j, igroup, 0] = data[count][2]  # Luminosity [# photons/s]
                SEDs[i, j, igroup, 1] = data[count][3]  # CUMULAIVE photon no. at time t
                SEDs[i, j, igroup, 2] = data[count][4]  # Avg. Egy
                SEDs[i, j, igroup, 3] = data[count][5]  # Ion1 csn
                SEDs[i, j, igroup, 4] = data[count][6]  # Ion1 cse
                SEDs[i, j, igroup, 5] = data[count][7]  # Ion2 csn
                SEDs[i, j, igroup, 6] = data[count][8]  # Ion2 cse
                SEDs[i, j, igroup, 7] = data[count][9]  # Ion3 csn
                SEDs[i, j, igroup, 8] = data[count][10]  # Ion3 cse
                count += 1

    return SimArray(agebins, "Gyr"), zbins, SEDs

def read_bpass_seds(sed_type="bin"):
    import os
    seddir = os.getenv(
        'BPASS_RAW_SED_DIR', '/lustre/scratch/astro/ds381/SEDs/BPASS/SEDS/')

    zbins = np.array(["001", "004", "008", "020", "040"])
    nZ = len(zbins)

    # Load one dset to get number of flux bins
    data = np.loadtxt(
        '%s/sed.bpass.instant.nocont.%s.z%s' % (seddir, sed_type, zbins[0])).T
    nAge = data.shape[0] - 1
    log_agebins = np.linspace(6, 10, nAge)
    agebins = 10**log_agebins
    SEDs = np.zeros((len(data[0]), nAge, nZ))

    for i, z in zip(range(nZ), zbins):
        data = np.loadtxt(
            '%s/sed.bpass.instant.nocont.%s.z%s' % (seddir, sed_type, z)).T
        for j in range(nAge):
            SEDs[:, j, i] = data[j + 1] / 1e6  # Lsun/Msun/A

    Ls = data[0]
    zbins = [float("0.%s" % s) for s in zbins]
    print zbins
    return agebins, zbins, Ls, SEDs


def read_and_smooth_bpass_seds(sed_type="bin", nbins=6900):
    from seren3.analysis import plots

    agebins, zbins, Ls, SEDs = read_bpass_seds(sed_type=sed_type)

    Ls_smoothed = np.zeros(nbins)
    SEDs_smoothed = np.zeros((nbins, len(agebins), len(zbins)))

    for i in range(len(agebins)):
        for j in range(len(zbins)):
            bc, mean, std = plots.fit_scatter(Ls, np.log10(SEDs[:,i,j]), nbins=nbins)
            SEDs_smoothed[:,i,j] = 10**mean
    Ls_smoothed = bc

    return agebins, zbins, Ls_smoothed, SEDs_smoothed


def interp_bpass_to_bc03(sed_type="bin", nbins=500):
    # from seren3.analysis import interpolate
    from scipy.interpolate import interp1d, interp2d

    # agebins, zbins, Ls, SEDs = read_and_smooth_bpass_seds(sed_type=sed_type, nbins=nbins)
    agebins, zbins, Ls, SEDs = read_bpass_seds(sed_type=sed_type)
    bc03age, bc03z, bc03Ls, bc03SEDs = read_seds()

    zbins = np.array(zbins)

    SEDs_interpolated1 = np.zeros((len(Ls), len(bc03age), len(bc03z)))

    for i in range(len(Ls)):
        # print i
        fn = interp2d(zbins, agebins, SEDs[i,:,:])
        SEDs_interpolated1[i,:,:] = fn(bc03z, bc03age)

    SEDs_interpolated2 = np.zeros((len(bc03Ls), len(bc03age), len(bc03z)))

    # Do wavelength
    for i in range(len(bc03age)):
        for j in range(len(bc03z)):
            fn = interp1d(Ls, SEDs_interpolated1[:,i,j], fill_value="extrapolate")
            SEDs_interpolated2[:,i,j] = fn(bc03Ls)

    idx = np.where(SEDs_interpolated2 < 0.)
    SEDs_interpolated2[idx] = 0.

    return bc03age, bc03z, bc03Ls, SEDs_interpolated2

def read_seds():
    '''
    Read all_seds.dat file
    '''
    import os
    seddir = os.getenv('RAMSES_SED_DIR', './')
    # seddir = './'

    agebins = np.delete(np.loadtxt('%s/age_bins.dat' % seddir), 0)
    # agebins *= 1e-9  # convert from yr to Gyr
    zbins = np.delete(np.loadtxt('%s/metallicity_bins.dat' % seddir), 0)
    nAge = len(agebins)
    nZ = len(zbins)
    with open('%s/all_seds.dat' % seddir, 'rb') as f:
                #  Skip dummy
        skip(f)

        #  Read nLs=
        nLs = np.fromfile(file=f, dtype=np.int32, count=1)[0]

        # Skip 3 records
        [skip(f) for i in range(3)]

        # Read wavelength bins
        Ls = np.zeros(nLs)
        Ls = np.fromfile(file=f, dtype=np.float64, count=nLs)

        # Skip two records
        skip(f)

        # print nLs, nAge, nZ
        SEDs = np.zeros((nLs, nAge, nZ))
        for i in range(nZ):
            for j in range(nAge):
                skip(f)
                SEDs[:, j, i] = np.fromfile(
                    file=f, dtype=np.float64, count=nLs)
                skip(f)

        return agebins, zbins, Ls, SEDs


def write_seds(agebins, zbins, Ls, SEDs, seddir=None):
    import os

    if seddir is None:
        seddir = os.getenv('RAMSES_NEW_SED_DIR', './')

    assert(SEDs.dtype == np.float64)

    if agebins[0] != 0.:
        # Add zero age bin from lowest available
        shape = SEDs.shape
        SEDs_tmp = np.zeros((shape[0], shape[1] + 1, shape[2]))
        for i in range(len(zbins)):
            SEDs_tmp[:, 0, i] = SEDs[:, 0, i]

        for i in range(len(zbins)):
            for j in range(len(agebins)):
                SEDs_tmp[:, j + 1, i] = SEDs[:, j, i]

        agebins = np.insert(agebins, 0., 0.)
        SEDs = SEDs_tmp

    nZ = len(zbins)
    nAge = len(agebins)

    # agebins = np.insert(agebins, 0, len(agebins))
    # zbins = np.insert(zbins, 0, len(zbins))

    with open('%s/age_bins.dat' % seddir, 'w') as f:
        f.write("%d\n" % len(agebins))
        for a in agebins:
            f.write("%e\n" % a)

    with open('%s/metallicity_bins.dat' % seddir, 'w') as f:
        f.write("%d\n" % len(zbins))
        for z in zbins:
            f.write("%e\n" % z)
    #np.savetxt('%s/age_bins.dat' % seddir, agebins)
    #np.savetxt('%s/metallicity_bins.dat' % seddir, zbins)

    from seren.utils.f90.fortranfile import FortranFile
    f = FortranFile('%s/all_seds.dat' % seddir, '@', 'i', 'w')

    # Write number of wavelength and age bins
    f._write_check(np.int32(8))
    f._write_check(np.int32(len(Ls)))
    f._write_check(np.int32(nAge))
    f._write_check(np.int32(8))
    f._write_check(np.int32(len(Ls) * 8))

    # Write Ls
    f.write(Ls.tostring())

    # Write number of bytes for wavelength bins (len(Ls) * 8)
    f._write_check(np.int32(len(Ls) * 8))
    # f.write(np.int32(len(Ls) * 8).tostring())
    # f.write(np.int32(len(Ls) * 8).tostring())

    for i in range(nZ):
        for j in range(nAge):
            f._write_check(np.int32(len(Ls) * 8))
            f.write(SEDs[:, j, i].tostring())
            f._write_check(np.int32(len(Ls) * 8))


def plot_sed(agebins, zbins, Ls, SEDs, ax=None, fs=20, label_ion_freqs=False, show_legend=False, **kwargs):
    import numpy as np
    import matplotlib.pylab as plt
    import matplotlib.cm as cm
    from seren3.utils.plot_utils import ncols

    # ages = np.array([1, 11, 21, 31, 41, 100])
    # ages=[1., 10., 100., 1000., 10000.]
    ages = kwargs.pop("ages", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, \
                    50, 100, 502, 1005, 10005])
    colors = kwargs.pop("colors", ncols(len(ages), cmap=kwargs.pop("cmap", "rainbow")))
    # colors = kwargs.pop("colors", ncols(len(ages), cmap=kwargs.pop("cmap", "Set1")))
    # colors = ["r", "b", "darkorange"]

    zbins = np.array(zbins); agebins = np.array(agebins)

    Z_sun = 0.02  # Solar metallicity
    z_idx = (np.abs(Z_sun - zbins)).argmin()

    if (ax is None):
        fig, ax = plt.subplots(figsize=(8,8))

    print kwargs
    for age, c in zip(ages, colors):
        age_idx = (np.abs(age*1e6 - agebins)).argmin()
        print age_idx
        plot_every = 1
        ax.semilogy(Ls[::plot_every], SEDs[:, age_idx, z_idx][::plot_every], label="%i Myr" % (float(agebins[age_idx])/1e6) if show_legend else None, color=c, **kwargs)

    ax.set_xlim(100, 1000)

    if show_legend:
        box = ax.get_position()
        ax.set_position([box.x0 - box.width * 0.05, box.y0 + box.height * 0.05,
                         box.width, box.height * 0.9])
        ax.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1, prop={'size':14})

        # ax.legend(loc='lower right', prop={"size":16})
        # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

        # # Put a legend to the right of the current axis
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda$ [$\AA$]', fontsize=fs)
    ax.set_ylabel(r'J$_{\lambda}$ [L$_{\odot}$ M$_{\odot}$$^{-1}$ $\AA^{-1}$]', fontsize=fs)

    if label_ion_freqs:
        HI = 912  # A
        HeI = 505
        HeII = 228
        ion_freq_ls = '-.'
        ax.axvline(x=HI, linestyle=ion_freq_ls, color='k')
        ax.axvline(x=HeI, linestyle=ion_freq_ls, color='k')
        ax.axvline(x=HeII, linestyle=ion_freq_ls, color='k')

        ypos = ax.get_ylim()[1]
        ypos *= 1.5
        ax.text(HI, ypos, r'HI', fontsize=20, ha='center')
        ax.text(HeI, ypos, r'HeI', fontsize=20, ha='center')
        ax.text(HeII, ypos, r'HeII', fontsize=20, ha='center')

    ax.set_ylim(1e-15, 1e1)
    ax.set_xlim(100, 1200)


def read_bouwens_2015():
    import os
    from seren3 import config

    fname = "bouwens_2015.csv"
    data_dir = os.path.join(config.get("data", "data_dir"), "obs/")
    data = {}

    with open("%s/%s" % (data_dir, fname), "Urb") as f:
        import csv
        reader = csv.reader(f)
        z = np.inf
        for line in reader:
            if line[0].startswith('z'):
                z = int(line[0][4])
                data[z] = {'M1600': [], 'phi': [], 'err': []}
            elif z is not np.inf:
                M, phi, err = [l.strip() for l in line]
                data[z]['M1600'].append(float(M))
                if phi.endswith('b'):
                    # Upper limit
                    data[z]['phi'].append(phi)
                    data[z]['err'].append(np.nan)
                else:
                    data[z]['phi'].append(float(phi))
                    data[z]['err'].append(float(err))

    return data


# def plot_seds(ages=[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 16., 25., 50., 100., 252., 502., 1005., 10005.]):
#     '''
#     Test function to ensure SED tables are read correctly
#     '''
#     import matplotlib.cm as cm
#     import matplotlib.pylab as plt
#     import os

#     HI = 912  # A
#     HeI = 505
#     HeII = 228

#     agebins, zbins, Ls, SEDs = read_seds(os.getenv("RAMSES_SED_DIR"))
#     nLs = len(Ls)

#     Z_sun = 0.02  # Solar metallicity
#     z_idx = (np.abs(Z_sun - zbins)).argmin()

#     lambda_range = np.linspace(100., 1000., 250)

#     # ages = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 16.,
#     #         25., 50., 100., 252., 502., 1005., 10005.]  # Myr

#     ages = [1., 10.]  # , 100., 1000., 10000.]  # Myr

#     color = cm.rainbow(np.linspace(0, 1, len(ages)))

#     ax = plt.subplot(111)

#     for age, c in zip(ages, color):
#         y = []
#         age_idx = (np.abs(age * 1e6 - agebins)).argmin()
#         for lambda_A in lambda_range:
#             ii = 0
#             while (ii < nLs) and (Ls[ii] < lambda_A):
#                 ii += 1
#             y.append(SEDs[ii, age_idx, z_idx])

#         ax.plot(lambda_range, y, color=c, label='BC03 %1.0f Myrs' %
#                 (age), linestyle='--', linewidth=3.)

#     print 'PLOTTING BPASS'
#     data = np.loadtxt(
#         '/lustre/scratch/astro/ds381/SEDs/BPASS/SEDS/sed.bpass.instant.nocont.bin.z020').T
#     idx = np.where(np.logical_and(data[0] >= 100., data[0] <= 1000.))

#     ages = [1, 11]  # , 21, 31, 41]

#     labels = [6, 7, 8, 9, 10]

#     c = plt.cm.rainbow(np.linspace(0, 1, len(ages)))

#     for i, col in zip(range(len(ages)), c):
#         plt.semilogy(data[0][idx], data[ages[i]][idx] / 1.e6, color=col, label=r'BPASS %1.0f Myr' %
#                      ((10.**labels[i]) / 10.**6), linewidth=1., linestyle='-')

#     plt.legend()
#     # Shrink current axis by 20%
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

#     # Put a legend to the right of the current axis
#     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12.})
#     ax.set_yscale('log')
#     plt.xlabel(r'$\lambda$ [$\AA$]')
#     plt.ylabel(r'J$_{\lambda}$ [L$_{\odot}$/M$_{\odot}$/$\AA$]')
#     ax.axvline(x=HI, linestyle='--', color='k')
#     ax.axvline(x=HeI, linestyle='--', color='k')
#     ax.axvline(x=HeII, linestyle='--', color='k')

#     plt.show()