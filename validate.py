"""
Quick end-to-end validation on a small grid (Nx=Ny=Nz=10, ~1331 DOF).
Target metric = 3 (n=m=k=1, the fundamental mode of the unit cube).
Prints VALIDATION PASSED if the integer metric error < 0.1.
Run as: mpiexec -n N python -u validate.py
"""
import sys
import petsc4py; petsc4py.init(sys.argv)
from mpi4py import MPI
from petsc4py import PETSc
from src.operators import assemble_distributed
from src.solver import solve_evp
import numpy as np
import time

Nx, Ny, Nz      = 10, 10, 10
q               = 6
xi_half = eta_half = zeta_half = 0.501
target_metric   = 3.0    # n=m=k=1  →  n²+m²+k²=3
num_modes       = 5
krylov_size     = 30

if __name__ == "__main__":
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()
    N_total = (Nx+1)*(Ny+1)*(Nz+1)

    if rank == 0:
        print(f"=== ENDGAME VALIDATION ===")
        print(f"Grid: {Nx}x{Ny}x{Nz} = {N_total} DOF | ranks={size} | target metric={target_metric}")

    t0 = time.time()
    A, B, x, y, z = assemble_distributed(Nx, Ny, Nz, q, xi_half, eta_half, zeta_half)
    int_m, lsq, evecs = solve_evp(A, B, target_metric, num_modes, krylov_size)

    if rank == 0:
        print(f"Wall time: {time.time()-t0:.2f}s")
        if int_m is not None:
            for i in range(min(3, len(int_m))):
                err = abs(int_m[i] - round(int_m[i]))
                print(f"  mode {i}: metric={int_m[i]:.6f}  |err|={err:.2e}")
            passed = all(abs(int_m[i]-round(int_m[i])) < 0.1 for i in range(min(3, len(int_m))))
            print("VALIDATION PASSED" if passed else "WARNING: CHECK RESULTS")
        else:
            print("ERROR: no modes converged")
