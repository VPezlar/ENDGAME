import numpy as np
import subprocess
import os
from mpi4py import MPI

# _ROOT is computed from THIS file's location.
# When a job rsyncs the project to a per-job RUNDIR, __file__ is
# RUNDIR/src/FDq.py, so _ROOT = RUNDIR — fully isolated per job.
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FDQ_BIN = os.path.join(_ROOT, 'FDq', 'bin', 'FDq')


def FDq_Mat(N, q):
    """
    Compute 1-D FD matrices for N+1 grid nodes, stencil size q.

    Reads/writes FDq/inputs/ and FDq/output/ relative to _ROOT.
    Race-condition safety is guaranteed by the job runner: each PBS job
    rsyncs the project to a per-job scratch directory, so _ROOT is
    unique per job and no two jobs ever touch the same files.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = None
    if rank == 0:
        input_path = os.path.join(_ROOT, 'FDq', 'inputs', 'input_size.dat')
        with open(input_path, 'w') as f:
            f.write(f"{N + 1}\n{q}")

        # Binary uses paths relative to cwd=_ROOT/src
        subprocess.call(_FDQ_BIN, cwd=os.path.join(_ROOT, 'src'))

        D1_FDq = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'd1.dat'))
        D2_FDq = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'd2.dat'))
        grid   = np.loadtxt(os.path.join(_ROOT, 'FDq', 'output', 'eta.dat'))

        # Hard assertion: catch any size mismatch immediately
        assert D1_FDq.shape == (N + 1, N + 1), \
            f"[FDq] WRONG SIZE: expected ({N+1},{N+1}), got {D1_FDq.shape}"
        assert len(grid) == N + 1, \
            f"[FDq] WRONG GRID: expected {N+1}, got {len(grid)}"

        print(f"[FDq] N={N} -> D1:{D1_FDq.shape} grid:{len(grid)} root:{_ROOT}", flush=True)
        result = [D1_FDq, D2_FDq, grid]

    result = comm.bcast(result, root=0)
    assert result is not None, "FDq_Mat: bcast returned None"
    assert len(result[2]) == N + 1, \
        f"[Rank {comm.Get_rank()}] bcast size mismatch: got {len(result[2])}, expected {N+1}"
    return result
