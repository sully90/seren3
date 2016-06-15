from seren3.analysis.parallel import piter
import numpy as np

from mpi4py import MPI
# MPI runtime variables
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
host = (rank == 0)

if host:
    data = np.linspace(1., 10., num=10)
    print data
else:
    data = None

dest = {}

for i, res in piter(data, storage=dest):
    #print rank, i
    res.result = (i**2.)

if host:
    print rank, dest