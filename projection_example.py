'''
This example shows how to easily make images of a given halo
'''

def main(path, iout, finder):
    import seren3
    from seren3.analysis.visualization import EngineMode

    # First we load a Simulation object which stores the path to the dataset and provides functions to load a snapshot
    simulation = seren3.init(path)
    snapshot = simulation[iout]

    # Now we load the halo catalogue. The correct paths should be set in the config file (/path/to/seren3/._seren3_config.txt)
    halos = snapshot.halos(finder=finder)

    # Lets sort the halo catalogue by virial mass (descending), and get the most massive halo
    sorted_halos = halos.sort("Mvir")
    halo = sorted_halos[0]

    # Now lets make some projections. We can use raytracing of FFT convolutions over the AMR levels (splatting)
    amr_nH = halo.g.projection('nH', mode=EngineMode.RAYTRACING)
    dm_mass = halo.d.projection('mass', mode=EngineMode.SPLATTER)

    # Show the plots
    amr_nH.save_plot().show()
    dm_mass.save_plot(cmap='jet_black').show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print "Usage: python projection_examply.py <sim_path> <sim_ioutput> <halo_finder>"
        sys.exit()
    sim_path = sys.argv[1]
    sim_iout = int(sys.argv[2])
    halo_finder = sys.argv[3]
    main(sim_path, sim_iout, halo_finder)
