import numpy as np
import pickle

_MASS_UNIT = "Msol h**-1"

def unpack(path, iout):
    fname = '%s/abundance_%05i.p' % (path, iout)
    dest = pickle.load( open(fname, 'rb') )
    tot_mass = []
    star_mass = []
    for rank in dest:
        for item in dest[rank]:
            res = item.result
            tot_mass.append(res["tot_mass"])
            star_mass.append(res["star_mass"])

    return np.array(star_mass), np.array(tot_mass)

def plot(snap, fname, plot_baryon_fraction=True, compare=True, dm_particle_cutoff=100, star_particle_cutoff=1):
    import pickle
    from pynbody.plot.stars import moster, behroozi
    import matplotlib.pylab as plt

    cosmo = snap.cosmo
    data = None
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    nrecords = len(data)
    mass = np.zeros(nrecords); stellar_mass = np.zeros(nrecords)
    np_dm = np.zeros(nrecords); np_star = np.zeros(nrecords)

    for i in range(nrecords):
        res = data[i].result
        mass[i] = res["tot_mass"]
        stellar_mass[i] = res["star_mass"]
        np_dm[i] = res["np_dm"]
        np_star[i] = res["np_star"]
        del res

    idx = np.where(np.logical_and( np_dm >= dm_particle_cutoff, np_star >= star_particle_cutoff ))
    mass = mass[idx]; stellar_mass = stellar_mass[idx]

    if plot_baryon_fraction:
        cosmic_mean = cosmo["omega_b_0"]/cosmo["omega_M_0"]
        baryon_fraction = mass*cosmic_mean
        plt.loglog(mass, baryon_fraction, linestyle='dotted', label='f_b')
        plt.loglog(mass, 0.1*baryon_fraction, linestyle='dotted', label='0.1f_b')


    plt.title('(z = %f)'% snap.z )

    ax = plt.gca()
    ax.scatter(mass, stellar_mass, color='k', alpha=0.5, zorder=10)
    ax.set_xscale("log"); ax.set_yscale("log")
    # plt.xlim(min(mass), max(mass))

    mass = snap.array(mass, _MASS_UNIT)
    if compare:
        ystarmasses, errors = moster(mass.in_units("Msol"), snap.z)
        ystarmasses = snap.array(ystarmasses, "Msol").in_units(_MASS_UNIT)
        plt.fill_between(mass, np.array(ystarmasses)/np.array(errors),
                         y2=np.array(ystarmasses)*np.array(errors),
                         facecolor='#BBBBBB',color='#BBBBBB')
        plt.loglog(mass, ystarmasses, label='Moster et al (2013)', color='k')

        ystarmasses, errors = behroozi(mass.in_units("Msol"), snap.z)
        ystarmasses = snap.array(ystarmasses, "Msol").in_units(_MASS_UNIT)
        plt.fill_between(mass, np.array(ystarmasses)/np.array(errors),
                         y2=np.array(ystarmasses)*np.array(errors),
                         facecolor='g',color='g')
        plt.loglog(mass, ystarmasses, label='Behroozi et al (2013)', color='g', alpha=0.5)

    ax.set_xlabel(r'log$_{10}$(M$_{\mathrm{h}}$ [M$_{\odot}$/h])')
    ax.set_ylabel(r'log$_{10}$(M$_{*}$ [M$_{\odot}$/h])')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13),
              fancybox=True, shadow=True, ncol=5, prop={'size':10.5})

    plt.show()


def stars(dset):
    return np.where( np.logical_and(dset["epoch"] != 0, dset["id"] > 0.) )

def main(path, iout, pickle_path=None):
    import seren3
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading data")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)  # disable multiprocessing

    halos = snap.halos(finder='ctrees')
    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        mpi.msg("Working on halo %i \t %i" % (i, h.hid))

        part_dset = h.p[["mass", "epoch", "id"]].flatten()
        gas_dset = h.g["mass"].flatten()
        gas_mass = gas_dset["mass"].in_units(_MASS_UNIT)

        star_idx = stars(part_dset)
        part_mass = part_dset["mass"].in_units(_MASS_UNIT)

        tot_mass = part_mass.sum() + gas_mass.sum()
        star_mass = part_mass[star_idx].sum()

        np_star = len(part_dset["mass"][star_idx])
        np_dm = len(part_dset["mass"]) - np_star

        sto.idx = h["id"]
        sto.result = {"tot_mass" : tot_mass, "star_mass" : star_mass, "np_dm" : np_dm, "np_star" : np_star}

    if mpi.host:
        import os
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/abundance_%05i.p" % (pickle_path, iout)
        with open(fname, 'wb') as f:
            pickle.dump( mpi.unpack(dest), f )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, iout, pickle_path)