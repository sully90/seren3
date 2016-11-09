from mpi import piter, msg, host, rank
import numpy as np

if host:
    data = np.linspace(1, 10, num=10, dtype=np.int32)
    print data
else:
    data = None

dest = {}

for i, res in piter(data, storage=dest):
    #print rank, i
    res.idx = i
    res.result = (rank+1)*(i**2)

# print rank, dest
msg(dest)