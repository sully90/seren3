import numpy as np
from seren3.analysis.parallel import Result

# Setup MPI environment
from mpi4py import MPI
# MPI runtime variables
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
host = (rank == 0)

def is_host():
    return host

def piter(iterable, storage=None, keep_None=False, print_stats=False):
    '''
    Chunks and scatters iterables before gathering the results at the end
    Parameters
    ----------
    iterable : np.ndarray
        List to parallelise over - must be compatible with pickle (i.e numpy arrays, required by mpi4py)
    storage = None : dict
        Dictionary to store final (reduced) result on rank 0, if desired. Must be an empty dictionary

    Example
    ----------
    test = np.linspace(1, 10, 10)

    dest = {}

    for i, sto in piter(test, storage=dest):
        sto.result = i**2

    # Access result in dest (keys are rank numbers, values are lists of Result objects)
    '''

    # Chunk the iterable and prepare to scatter
    if host:
        if not hasattr(iterable, "__iter__"):
            terminate(500, e=Exception("Argument %s is not iterable!" % iterable))

        if (storage is not None) and (not isinstance(storage, dict)):
            raise Exception("storage must be a dict")

    if host:
        chunks = np.array_split(iterable, size)
    else:
        chunks = None

    # Scatter
    local_iterable = comm.scatter(chunks, root=0)
    del chunks

    if print_stats:
        msg("Received %d items" % len(local_iterable))

    local_results = []
    # yield the iterable
    for i in xrange(len(local_iterable)):
        #init the result
        res = Result(rank, i)

        # yield to the for loop
        yield local_iterable[i], res

        if keep_None is False and res.result is None:
            continue
        # store the result
        local_results.append(res)

    # If the executing code sets dest, then reduce it to rank 0
    if storage is not None:
        if print_stats:
            msg("Pending gather")
        # comm.Barrier()
        results = comm.gather(local_results, root=0)
        del local_results

        if host:
            # Append results to storage
            for irank in range(size):
                local_results = results[irank]
                storage[irank] = local_results

def unpack(dest):
    result = []
    for rank in dest:
        for item in dest[rank]:
            result.append(item)
    return result

def msg(message):
    print '[rank %d   ]: %s' % (rank, message)


def terminate(code, e=None):
    if e:
        msg("Caught exception: %s" % e)
    else:
        msg("Terminating")
    comm.Abort(code)
