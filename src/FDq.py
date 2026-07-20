import numpy as np
import subprocess
import os
from mpi4py import MPI

# Binary lives at the shared project root
_ROOT    = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_FDQ_BIN = os.path.join(_ROOT, 'FDq', 'bin', 'FDq')


def FDq_Mat(N, q):
    """
    Compute 1-D FD matrices for N+1 nodes, stencil q.

    Uses os.getcwd() for I/O — cwd is set to the case directory by
    main.py at startup. Each case has its own FDq/inputs/ and
    FDq/output/, so concurrent jobs never touch the same files.
    Binary cwd = case_dir/src/ so '../FDq/...' resolves inside the case.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = None
    if rank == 0:
        case_dir = os.getcwd()
        inputs   = os.path.join(case_dir, 'FDq', 'inputs')
        outputs  = os.path.join(case_dir, 'FDq', 'output')
        cwd_bin  = os.path.join(case_dir, 'run_cwd')
        os.makedirs(cwd_bin, exist_ok=True)  # ../FDq/ resolves here

        with open(os.path.join(inputs, 'input_size.dat'), 'w') as f:
            f.write(f"{N + 1}\n{q}")

        subprocess.call(_FDQ_BIN, cwd=cwd_bin)

        D1   = np.loadtxt(os.path.join(outputs, 'd1.dat'))
        D2   = np.loadtxt(os.path.join(outputs, 'd2.dat'))
        grid = np.loadtxt(os.path.join(outputs, 'eta.dat'))

        assert D1.shape == (N+1, N+1), \
            f"FDq size error: got {D1.shape}, expected ({N+1},{N+1})"
        print(f"[FDq] N={N} -> D1:{D1.shape}  case:{case_dir}", flush=True)
        result = [D1, D2, grid]

    result = comm.bcast(result, root=0)
    return result
