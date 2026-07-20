import numpy as np
import subprocess
import os
from mpi4py import MPI

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def FDq_Mat(N, q):
    """Compute FDq differentiation matrices for N+1 points at order q.

    Writes N+1 and q to FDq/inputs/input_size.dat, calls the Fortran binary
    FDq/bin/FDq, then reads D1, D2 and the grid from FDq/output/.

    File I/O and the Fortran binary are executed on rank 0 only;
    results are broadcast to all other MPI ranks to avoid race conditions.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = None
    if rank == 0:
        # Write input file and call Fortran binary
        input_size = str(N + 1) + "\n" + str(q)
        with open(os.path.join(_ROOT, 'FDq', 'inputs', 'input_size.dat'), 'w') as f:
            f.write(input_size)

        subprocess.call(os.path.join(_ROOT, 'FDq', 'bin', 'FDq'),
                        cwd=os.path.join(_ROOT, 'src'))

        D1_FDq = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'd1.dat'))
        D2_FDq = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'd2.dat'))
        grid   = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'eta.dat'))
        result = [D1_FDq, D2_FDq, grid]

    # Broadcast from rank 0 to all other ranks
    result = comm.bcast(result, root=0)
    return result
