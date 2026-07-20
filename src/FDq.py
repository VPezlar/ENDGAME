import numpy as np
import subprocess
import os
import shutil
import tempfile
from mpi4py import MPI

# Absolute path to the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Absolute path to the FDq binary (never changes).
_FDQ_BIN = os.path.join(_ROOT, 'FDq', 'bin', 'FDq')


def FDq_Mat(N, q):
    """
    Compute 1-D finite-difference matrices for N+1 grid nodes, stencil size q.

    The Fortran binary FDq/bin/FDq uses hardcoded relative paths:
        ../FDq/inputs/input_size.dat   (input)
        ../FDq/output/d1.dat           (output)
        ../FDq/output/d2.dat           (output)
        ../FDq/output/eta.dat          (output)
    resolved relative to the working directory it is launched with.

    RACE-CONDITION FIX (multi-job HPC):
    On a cluster, multiple jobs share the same NFS home directory. If two jobs
    call FDq_Mat concurrently, they overwrite each other's input_size.dat and
    output files, producing silently wrong results (size mismatches, corrupted
    matrices).

    The fix: each call creates a private temporary directory that mirrors the
    expected layout. The binary is launched with cwd=tmpdir/src, so all file
    I/O happens inside that private directory.  The tmpdir is deleted after use.

    MPI: only rank 0 does the I/O and subprocess call; the result is broadcast
    to all other ranks afterwards.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    result = None
    if rank == 0:
        # Create a unique per-call temp directory.
        # tempfile.mkdtemp() uses the system temp dir (/tmp on Linux),
        # which is node-local and not shared across NFS — zero race risk.
        tmpdir = tempfile.mkdtemp(prefix='fdq_')
        try:
            # Recreate the directory structure the Fortran binary expects.
            # binary cwd = tmpdir/src
            # binary reads: ../FDq/inputs/input_size.dat
            #               = tmpdir/FDq/inputs/input_size.dat
            # binary writes: ../FDq/output/d1.dat
            #               = tmpdir/FDq/output/d1.dat
            os.makedirs(os.path.join(tmpdir, 'src'))
            os.makedirs(os.path.join(tmpdir, 'FDq', 'inputs'))
            os.makedirs(os.path.join(tmpdir, 'FDq', 'output'))

            # Write the input file into the private directory.
            with open(os.path.join(tmpdir, 'FDq', 'inputs', 'input_size.dat'), 'w') as f:
                f.write(f"{N + 1}\n{q}")

            # Call the Fortran binary with cwd pointing to tmpdir/src.
            # subprocess.call() blocks until the binary finishes.
            subprocess.call(_FDQ_BIN, cwd=os.path.join(tmpdir, 'src'))

            # Read back the outputs from the private directory.
            D1_FDq = np.loadtxt(os.path.join(tmpdir, 'FDq', 'output', 'd1.dat'))
            D2_FDq = np.loadtxt(os.path.join(tmpdir, 'FDq', 'output', 'd2.dat'))
            grid   = np.loadtxt(os.path.join(tmpdir, 'FDq', 'output', 'eta.dat'))
            result = [D1_FDq, D2_FDq, grid]
        finally:
            # Always clean up, even if the binary crashes.
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Broadcast from rank 0 to all other ranks.
    # Before bcast: result is a list on rank 0, None on all others.
    # After bcast:  result is identical on all ranks.
    result = comm.bcast(result, root=0)
    return result
