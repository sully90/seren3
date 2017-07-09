_LAMBDA_A = 1600.

def main(path, ioutput, pickle_path):
    import pickle, os
    import seren3
    from seren3.analysis.parallel import mpi
    from seren3.utils.sed import io

    mpi.msg("Loading snapshot")
    snap = seren3.load_snapshot(path, ioutput)
    snap.set_nproc(1)
    halos = None  

    halos = snap.halos(finder="ctrees")
    halo_ix = None
    if mpi.host:
        halo_ix = halos.halo_ix(shuffle=True)

    nml = snap.nml
    RT_PARAMS_KEY = nml.NML.RT_PARAMS

    SED = io.read_seds()

    dest = {}
    kwargs = {"lambda_A" : _LAMBDA_A, "sed" : SED}
    for i, sto in mpi.piter(halo_ix, storage=dest, print_stats=True):
        h = halos[i]
        if (len(h.s) > 0):
            dset = h.s[["luminosity", "age"]].flatten(**kwargs)
            sto.idx = int(h["id"])
            sto.result = {"age" : dset["age"], "L" : dset["luminosity"], "lambda_A" : _LAMBDA_A, \
                    "hprops" : h.properties}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/%s/" % (path, halos.finder.lower())
        # pickle_path = "%s/" % path
        if os.path.isdir(pickle_path) is False:
            os.mkdir(pickle_path)
        unpacked_dest = mpi.unpack(dest)
        lum_dict = {}
        for i in range(len(unpacked_dest)):
            lum_dict[int(unpacked_dest[i].idx)] = unpacked_dest[i].result
        pickle.dump( lum_dict, open( "%s/luminosity_lambdaA_%s_database_%05i.p" % (pickle_path, int(_LAMBDA_A), ioutput), "wb" ) )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    ioutput = int(sys.argv[2])

    pickle_path = None
    if len(sys.argv) > 3:
        pickle_path = sys.argv[3]

    main(path, ioutput, pickle_path)