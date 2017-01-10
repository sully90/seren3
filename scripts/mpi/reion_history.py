'''
Computes the reionization history for a given simulation in MPI parallel
'''

MEM_OPT=False

def ioutput_xHII(xHII, xHII_type="volume_weighted", table=None, **kwargs):
    '''
    Returns the snapshot number closest to this ionised fraction
    '''
    import numpy as np

    if table is None:
        table = load_xHII_table(**kwargs)

    ioutputs = np.array(table.keys())
    xHII_hist = np.array( [table[i][xHII_type] for i in ioutputs] )
    idx = np.abs(xHII_hist - xHII).argmin()
    
    return ioutputs[idx]


def load_xHII_table(path='./'):
    import pickle, os
    fname = "{PATH}/xHII_reion_history.p".format(PATH=path)
    if os.path.isfile(fname):
        data = pickle.load( open(fname, "rb") )
        table = {}
        for item in data:
            table[item.idx] = item.result
        return table
    else:
        raise IOError("No such file: {FNAME}".format(FNAME=fname))

   
def unpack(dest):
    import numpy as np

    result = []
    for rank in dest:
        for item in dest[rank]:
            result.append(item)
    key = lambda item: item.result["z"]
    return np.array(sorted(result, key=key, reverse=True))


def main(path, pickle_path):
    import pickle
    from seren3.core.simulation import Simulation
    from seren3.analysis.parallel import mpi

    mpi.msg("Loading simulation")
    sim = Simulation(path)
    ioutputs = sim.numbered_outputs

    dest = {}
    for i, sto in mpi.piter(ioutputs, storage=dest):
        mpi.msg("Analysing snapshot %05i" % i)
        snap = sim[i]
        snap.set_nproc(1)  # disable multiprocessing on dset read

        z = snap.z
        vw = snap.quantities.volume_weighted_average("xHII", mem_opt=MEM_OPT)
        mw = snap.quantities.mass_weighted_average("xHII", mem_opt=MEM_OPT)

        sto.idx = i
        sto.result = {"z" : z, "volume_weighted" : vw, "mass_weighted" : mw}

    if mpi.host:
        if pickle_path is None:
            pickle_path = "%s/pickle/" % sim.path
        unpacked_dest = unpack(dest)
        pickle.dump(unpacked_dest,  open("%s/xHII_reion_history.p" % pickle_path, "wb") )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]

    pickle_path=None
    if len(sys.argv) > 2:
        from seren3.analysis.parallel import mpi
        pickle_path = sys.argv[2]
        if mpi.host:
            mpi.msg("pickle path: %s" % pickle_path)

    try:
        main(path, pickle_path)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        mpi.terminate(500, e=e)