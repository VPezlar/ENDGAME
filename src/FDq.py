import numpy as np
import subprocess
import os
import shutil
import tempfile
from mpi4py import MPI

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FDQ_BIN = os.path.join(_ROOT, 'FDq', 'bin', 'FDq')


def FDq_Mat(N, q):
    """
    Compute 1-D FD matrices for N+1 grid nodes, stencil size q.
    Uses a private per-call temp directory to avoid NFS race conditions.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = None
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix='fdq_')
        try:
            os.makedirs(os.path.join(tmpdir, 'src'))
            os.makedirs(os.path.join(tmpdir, 'FDq', 'inputs'))
            os.makedirs(os.path.join(tmpdir, 'FDq', 'output'))

            with open(os.path.join(tmpdir, 'FDq', 'inputs', 'input_size.dat'), 'w') as f:
                f.write(f"{N + 1}\n{q}")

            subprocess.call(_FDQ_BIN, cwd=os.path.join(tmpdir, 'src'))

            D1_FDq = np.loadtxt(os.path.join(tmpdir, 'FDq', 'output', 'd1.dat'))
            D2_FDq = np.loadtxt(os.path.join(tmpdir, 'FDq', 'output', 'd2.dat'))
            grid   = np.loadtxt(os.path.join(tmpdir, 'FDq', 'output', 'eta.dat'))

            # ASSERTION: verify the binary returned the correct size
            expected = N + 1
            assert D1_FDq.shape == (expected, expected), \
                f"FDq_Mat(N={N}): D1 shape {D1_FDq.shape} != expected ({expected},{expected})"
            assert len(grid) == expected, \
                f"FDq_Mat(N={N}): grid length {len(grid)} != expected {expected}"

            print(f"[FDq_Mat] N={N} -> D1:{D1_FDq.shape} D2:{D2_FDq.shape} grid:{len(grid)} tmpdir:{tmpdir}",
                  flush=True)
            result = [D1_FDq, D2_FDq, grid]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    result = comm.bcast(result, root=0)

    # ASSERTION on all ranks: verify broadcast delivered consistent data
    assert result is not None, "FDq_Mat: bcast returned None"
    assert len(result[2]) == N + 1, \
        f"[Rank {rank}] FDq_Mat(N={N}): after bcast grid length={len(result[2])}, expected {N+1}"

    return result
