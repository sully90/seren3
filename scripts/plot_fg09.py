#from __future__ import division
import numpy as np
# from matplotlib import use
# use('Agg')
import matplotlib.pyplot as plt
import sys

from matplotlib import rcParams
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

rcParams['axes.labelsize'] = 18
rcParams['xtick.major.pad'] = 10
rcParams['ytick.major.pad'] = 10

R = 10973731.6 # m**-1
c = 2.998e8 # ms**-1
h = 6.62606957e-34

nu_HI = c*R

HI = 912
HeI = 505
HeII = 228

def line_wavelength(Z, n1, n2):
    if n2 > 0:
        assert (n1 < n2)    
        inv_lambda = Z**2 * R * (1/n1**2 - 1/n2**2)
    else:
        inv_lambda = Z**2 * R * ((1/n1**2))
    return (1/inv_lambda)/1e-10

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

def load(fname):
    return np.loadtxt(fname, comments='#', usecols=(0, 1), unpack=True)

def write(f, redshifts, wavelengths, fluxs):
    #Write as follows:
    '''
    a) number of redshifts
    b) redshifts (increasing order)
    c) number of wavelengths
    d) wavelengths (increasing order) [Angstrom]
    e) fluxes per (redshift,wavelength) [photons cm-2 s-1 A-1 sr-1]
    '''
    num_redshifts = len(redshifts)
    num_wavelenghts = len(wavelengths[0])*num_redshifts
    for w in wavelengths:
        assert (len(w)*num_redshifts == num_wavelenghts)

    print 'Writing %d redshifts and %d wavelengths'%(num_redshifts, num_wavelenghts)
    f.write('%d\n'%num_redshifts)
    for z in redshifts:
        f.write('%.1f\n'%z)
    f.write('%d\n'%num_wavelenghts)
    for w in wavelengths:
        for wavelength in w:
            f.write('%f\n'%wavelength)
    for  l in fluxes:
        for flux in l:
            f.write('%f\n'%flux)


def main(fname, z, ax):
    data = load(fname)

    #Convert x axis to wavelength in amstrongs
    wavelength = c/(data[0]*nu_HI)

    #Integrate out frequency
    #data[1] = data[1] / (data[0]*nu_HI)

    #Convert to photon flux
    #Divide by photon energy: hc/lambda and convert from erg to J
    data[1] = data[1] / ((h*c/wavelength)/1e-7) # 1e-7 erg -> J
    data[1] = data[1] * 1e-21

    #Integrate out frequency
    data[1] = data[1]*(data[0]*nu_HI)

    #Convert wavelength to Angstoms
    wavelength = wavelength / 1e-10 # A

    #Photon flux per unit wavelength in Angstroms
    data[1] = data[1]/wavelength

    #Integrate out solid angle
    #NB: Ramses-RT will integrate out solid angle, so input tabel should be in units [photons cm-2 s-1 A-1 sr-1]
    data[1] = data[1] * (4*np.pi)

    #plt.loglog(data[0], data[1], linestyle='-')
    ax.plot(wavelength, data[1], linestyle='-', label='z = %.1f'%z)
    return wavelength, data[1]

if __name__ == "__main__":

    prefix = 'fg_uvb_dec11_z_'
    suffix = '.dat'
    #redshifts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    redshifts = np.arange(0, 9.2, 1)

    NUM_COLORS = len(redshifts)

    cm = plt.get_cmap('gist_rainbow')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Plot BC03 SED
    from seren3.utils.sed import io
    agebins, zbins, Ls, SEDs = io.read_seds()
    io.plot_sed(agebins, zbins, Ls, SEDs, ax=axs[0], show_legend=True, label_ion_freqs=True)

    # FG09

    ax = axs[1]
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    # out_file = open('./flux_zz_ramses_FG.out', 'w')
    wavelengths = []
    fluxes = []

    for z in redshifts:
        fname = '%s%.1f%s'%(prefix, z, suffix)
        wavelength, flux = main(fname, z, ax)
        wavelengths.append(wavelength)
        fluxes.append(flux)

    # write(out_file, redshifts, wavelengths, fluxes)
    # out_file.close()

    # adjustFigAspect(fig)

    box = ax.get_position()
    ax.set_position([box.x0 - box.width * 0.05, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
    #         fancybox=True, shadow=True, ncol=6, prop={'size':8})

    ax.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1, prop={'size':14})

    ax.set_xlabel(r'$\lambda [\AA]$', fontsize=20)
    #plt.xlabel(r'$\nu/\nu_{HI}$')
    ax.set_ylabel(r'J$_{\lambda}$ [photons cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', fontsize=20)
    #plt.xlim(0, 950)
    #plt.ylim(0, 700)

    ax.set_xlim(100, 1200)
    ax.set_ylim(10, 1e5)

    ax.set_yticks(np.arange(0, 701, 100))

    #HI = line_wavelength(1, 1, 0)
    #HeI = line_wavelength(2, 3, 4)
    #HeII = line_wavelength(2, 4, 0)

    #print HI, HeI, HeII

    #ax.axvline(x=1216, linestyle='--', color='k')
    ax.axvline(x=HI, linestyle='--', color='k')
    ax.axvline(x=HeI, linestyle='--', color='k')
    ax.axvline(x=HeII, linestyle='--', color='k')

    ypos = ax.get_ylim()[1]
    ypos *= 1.7
    ax.text(HI, ypos, r'HI', fontsize=20, ha='center')
    ax.text(HeI, ypos, r'HeI', fontsize=20, ha='center')
    ax.text(HeII, ypos, r'HeII', fontsize=20, ha='center')

    #plt.loglog()
    ax.set_yscale('symlog',linthreshy=1)
    # plt.savefig('plot.png')
    plt.show()
