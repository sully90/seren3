import numpy as np

def _volume_weighted_average(field, dset, volume, lengh_unit="pc"):
    qty = dset[field]
    dx = dset["dx"].in_units(lengh_unit)
    vol = volume.in_units("%s**3" % lengh_unit)

    return np.sum(qty * dx**3) / vol

def main(path, iout, field, pickle_path=None):
    import seren3
    import pickle, os
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading data")
    snap = seren3.load_snapshot(path, iout)
    snap.set_nproc(1)  # disbale multiprocessing/threading

    halos = snap.halos()
    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    dest = {}
    for i, sto in mpi.piter(halo_ix, storage=dest):
        h = halos[i]

        mpi.msg("Working on halo %i \t %i" % (i, h.hid))

        halo_volume = h.sphere.get_volume()
        halo_volume = snap.array(halo_volume, halo_volume.units)

        dset = h.g[[field, "dx"]].flatten()
        vw = _volume_weighted_average(field, dset, halo_volume)

        sto.idx = h["id"]
        sto.result = {"vw" : vw}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        fname = "%s/%s_halo_av_%05i.p" % (pickle_path, field, iout)
        pickle.dump( mpi.unpack(dest), open( fname, "wb" ) )


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    iout = int(sys.argv[2])
    field = sys.argv[3]
    pickle_path = None
    if len(sys.argv) > 4:
        pickle_path = sys.argv[4]

    main(path, iout, field, pickle_path)

    def scale(arr):
        scaled = (arr - arr.min()) / (arr.max() - arr.min())  # [0, 1]
        scaled -= 0.5  # [-0.5, 0.5]
        scaled *= 2.  # [-1, 1]
        return scaled

def write_training_data():
    # fb_scaled = fb_scaled[idx]; log_mvir_scaled = log_mvir_scaled[idx]; ftidal_scaled = ftidal_scaled[idx]
    # xHII_scaled = xHII_scaled[idx]; T_scaled = T_scaled[idx]

    topology = [4, 14, 1]
    top_line = "topology: %s\n" % (_list_to_string(topology))
    f = open('./fb_neural_net_00100_training_data.txt', 'w')
    f.write(top_line)
    for j in range(100):
        for i in range(len(fb_scaled)):
            l1 = "in: %f %f %f %f\n" % (log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i])
            l2 = "out: %f\n" % fb_scaled[i]
            f.write(l1)
            f.write(l2)

    f.close()

    f = open('./fb_neural_net_00100_prediction_data.txt', 'w')
    for i in range(len(fb_scaled)):
        l1 = "in: %f %f %f %f\n" % (log_mvir_scaled[i], ftidal_scaled[i], xHII_scaled[i], T_scaled[i])
        f.write(l1)

In [119]:     def scale(arr):
     ...:         scaled = (arr - arr.min()) / (arr.max() - arr.min())  # [0, 1]
     ...:         scaled -= 0.5  # [-0.5, 0.5]
     ...:         scaled *= 2.  # [-1, 1]
     ...:         return scaled

    f.close()
