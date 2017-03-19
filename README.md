# Seren3 #
This is the seren3 package, a python module which uses the PyMSES (http://irfu.cea.fr/Projets/PYMSES/intro.html) for RAMSES data access.
This package is designed for working with memory intensive datasets, such as those with deep AMR hierarchys. It contains modules for writing/analysing
halo catalogues, data access/filtering, derived fields, visualization (3D projections) and other utility functions.

## Installing ##
First, clone and install the modified PyMSES repository here: https://bitbucket.org/david_sullivan_/pymses-rt_4.1.3
Then, clone the seren3 directory, and type 'make'. This should compile all the c/f90 sources to their correct directories and run the inplace setup.py.
Finally, symlink seren3 to your $PYTHON_PATH