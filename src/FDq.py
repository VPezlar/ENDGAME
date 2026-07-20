import numpy as np
import subprocess
import os
from mpi4py import MPI

# Compute the absolute path to the project root (one level above this file's
# directory). Used to build all paths relative to the repo root regardless of
# where the script is run from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def FDq_Mat(N, q):
    """
    Compute 1-D finite-difference differentiation matrices for N+1 grid nodes
    using a Fornberg-type stencil of full size q.

    The computation is delegated to the Fortran binary FDq/bin/FDq, which uses
    the Hermanns & Hernandez (2008) piecewise polynomial interpolation algorithm
    to find optimal (Gauss-extrema) node positions and the resulting FD weights.

    Returns
    -------
    D1  : (N+1, N+1) numpy array  — first-derivative matrix  (d/dξ)
    D2  : (N+1, N+1) numpy array  — second-derivative matrix (d²/dξ²)
    grid: (N+1,)      numpy array  — optimal node positions on [-1, 1]

    MPI behaviour
    -------------
    Only rank 0 does the file I/O and subprocess call to avoid race conditions
    (all ranks writing the same file simultaneously would corrupt it). Rank 0
    then broadcasts the result to all other ranks so every rank holds identical
    copies of D1, D2, and grid.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialise to None on every rank. After the broadcast, all ranks will
    # hold the same list. Non-zero ranks never touch this variable before bcast.
    result = None

    if rank == 0:
        # The Fortran binary expects two integers: the number of nodes and
        # the stencil size q, on separate lines.
        input_size = str(N + 1) + "\n" + str(q)
        with open(os.path.join(_ROOT, 'FDq', 'inputs', 'input_size.dat'), 'w') as f:
            f.write(input_size)

        # subprocess.call() blocks until the Fortran binary finishes.
        # cwd=src/ is required because the Fortran binary uses relative paths
        # like '../FDq/output/d1.dat' to write its output files.
        subprocess.call(os.path.join(_ROOT, 'FDq', 'bin', 'FDq'),
                        cwd=os.path.join(_ROOT, 'src'))

        # Read the three output files written by the Fortran binary.
        # d1.dat and d2.dat are (N+1)×(N+1) matrices (space-separated rows).
        # eta.dat is a 1-D array of node positions on [-1, 1].
        D1_FDq = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'd1.dat'))
        D2_FDq = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'd2.dat'))
        grid   = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'eta.dat'))
        result = [D1_FDq, D2_FDq, grid]

    # comm.bcast() is a collective call — EVERY rank must reach this line.
    # Rank 0 sends its `result` list to all other ranks.
    # Other ranks receive it and overwrite their local `result = None`.
    # After this line, all P ranks hold identical [D1, D2, grid] lists.
    result = comm.bcast(result, root=0)
    return result
