'''
This script tests the parallel for loop seren3.analysis.parallel.mpi.piter and shows basic usage
'''

from seren3.analysis.parallel import mpi
import numpy as np

# If we are host (rank 0), then initialise the iterable
if mpi.host:
    data = np.linspace(1, 10, num=10, dtype=np.int32)
    print data
# Set the variable to None, host will scatter data to us
else:
    data = None

# Dictionaries can be passed to piter to store the results. This is automatically gathered to host
# upon for loop exit
dest = {}
for i, res in mpi.piter(data, storage=dest):
    # Set the idx variable of the yielded Result object to the value we were given
    res.idx = i

    # Set the result variable to whatever we want
    res.result = (mpi.rank+1)*(i**2)

# Print the results along with our rank. Rank 1 -> size should show empty dictionaries, host has
# all the results.
mpi.msg(dest)

# We can unpack the dictionary easily
if mpi.host:
    unpacked_dest = mpi.unpack(dest)
    mpi.msg(unpacked_dest)