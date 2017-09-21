from seren3.core import grafic_snapshot
import numpy as np
import matplotlib.pylab as plt
import sys


def int_vbc_pdf(vbc, sigma_vbc):
    ''' Integrate the PDF to compute P(>vbc) '''
    from scipy.integrate import quad
    return quad(vbc_pdf, vbc, np.inf, args=(sigma_vbc))


def vbc_pdf(v, sigma_vbc):
    return ((3.) / (2. * np.pi * (sigma_vbc ** 2.))) ** (3. / 2.) * 4. * np.pi * (v ** 2.) * np.exp(-((3. * v ** 2) / (2. * sigma_vbc ** 2)))

ics_path = sys.argv[1]
level = int(sys.argv[2])
nn = 2 ** level

ic = grafic_snapshot.GrafICSnapshot(ics_path, level, set_fft_sample_spacing=False)

print 'Computing vbc field...'
# vbc = ic['vbc']
deltab = ic['deltab']
deltac = ic['deltac']

vbx, vby, vbz = ic.linear_velocity(delta=deltab, species='b')
vcx, vcy, vcz = ic.linear_velocity(delta=deltac, species='c')

vb = np.sqrt(vbx**2 + vby**2 + vbz**2)
vc = np.sqrt(vcx**2 + vcy**2 + vcz**2)
vbc = vb - vc
print 'Done'

vbc_abs = np.sqrt((vbc ** 2).flatten())
sigma = vbc_abs.std()
mean = np.sqrt(np.mean(vbc**2))
x_plot = np.linspace(0, max(vbc_abs), 50)

vbc_abs_lin = np.sqrt((vbc_lin ** 2).flatten())
sigma_lin = vbc_abs_lin.std()
mean_lin = np.sqrt(np.mean(vbc_lin**2))
x_plot_lin = np.linspace(0, max(vbc_abs_lin), 50)

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)

axs[0].hist(vbc_abs, normed=True, histtype='step',
        color='r', label='Realisation + Bias', bins=100)

axs[1].hist(vbc_abs_lin, normed=True, histtype='step',
        color='b', label='Realisation', bins=100)
#ax.plot(x_plot, norm.pdf(x_plot, mean, sigma), color='r', linewidth=3, linestyle='-', label='Gaussian Fit')
axs[0].plot(x_plot, vbc_pdf(x_plot, np.sqrt(np.mean(vbc ** 2))),
        label='Tseliakhovich 2011 PDF', linestyle='--', color='k')
axs[1].plot(x_plot_lin, vbc_pdf(x_plot_lin, np.sqrt(np.mean(vbc_lin ** 2))),
        label='Tseliakhovich 2011 PDF', linestyle='--', color='k')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.15,
#                  box.width, box.height * 0.9])

# Put a legend below current axis
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#           fancybox=True, shadow=True, ncol=3, prop={'size': 11})

for ax in axs.flatten():
    ax.legend(loc='upper right', frameon=False, prop={'size': 14})
    # ax.set_ylabel("PDF", fontsize=20)
    ax.set_xlabel(r"$|v_{bc}|$ [km/s]", fontsize=20)
axs[0].set_ylabel("PDF", fontsize=20)
plt.show()
