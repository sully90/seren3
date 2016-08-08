def plot_Mc(sims, labels, cols=None, show=False, plot_Okamoto=False):
    '''
    Plots Mc vs z
    '''
    import pickle
    import numpy as np
    import matplotlib.pylab as plt

    fig, ax = plt.subplots()
    if cols is None:
        from seren3.utils import plot_utils
        cols = plot_utils.ncols(len(sims))

    for sim, label, c in zip(sims, labels, cols):
        data = pickle.load( open("%s/Mc_alpha_fit.p" % sim.path, "rb") )
        Mc = np.zeros(len(data))
        sigma_Mc = np.zeros(len(data))
        z = np.zeros(len(data))

        for i in range(len(data)):
            result = data[i].result
            Mc[i], sigma_Mc[i] = result["Mc"]
            z[i] = result["z"]

        ax.semilogy(z, Mc, label=label, color=c, linewidth=1.5)

        if plot_Okamoto:
            from seren3.scripts.mpi.fbaryon import gnedin_fitting_func

            data = np.loadtxt('~/Mc_Okamoto08.txt', unpack=True)
            alpha_Okamoto = 2.
            z_Okamoto, Mc_Okamoto = (data[1], data[2])
            raise NotImplementedError("Not finished")

    ax.set_xlabel(r"z")
    ax.set_ylabel(r"log$_{10}$(M$_{\mathrm{c}}$(z) [M$_{\odot}$])")
    if show:
        plt.legend()
        plt.show(block=False)
    return fig, ax

def main(path):
    import os
    from seren3.core.simulation import Simulation
    from seren3.analysis.parallel import mpi
    from seren3.scripts.mpi import fbaryon

    mpi.msg("Loading simulation")
    sim = Simulation(path)
    ioutputs = sim.numbered_outputs

    dest = {}
    for i, sto in mpi.piter(ioutputs, storage=dest):
        if os.path.isfile("%s/pickle/fbaryon_%05i.p" % (sim.path, i)):
            mpi.msg("Analysing snapshot %05i" % i)
            snap = sim[i]
            snap.set_nproc(1)

            Mc, alpha, corr = fbaryon.fit(snap)
            sto.idx = i
            sto.result = {"Mc" : Mc, "alpha" : alpha, "corr" : corr, "z" : snap.z}
        else:
            mpi.msg("Pickle file not found, skipping output %05i" % i)
            sto.result = None

    if mpi.host:
        import pickle
        pickle.dump( mpi.unpack(dest), open("%s/Mc_alpha_fit.p" % path, "wb") )

if __name__ == "__main__":
    import sys
    path = sys.argv[1]

    try:
        main(path)
    except Exception as e:
        from seren3.analysis.parallel import mpi
        mpi.terminate(500, e=e)