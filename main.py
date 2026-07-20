import os
import sys
import time
import numpy as np

# Initialize PETSc environment BEFORE other imports
import petsc4py
petsc4py.init(sys.argv)
from mpi4py import MPI

from src.operators import assemble_matrices
from src.solver import solve_evp

# ==========================================
# Master Controls
# ==========================================
Nx, Ny, Nz = 30, 30, 30      # Grid points per direction
q = 6                          # FD stencil half-width

xi_half, eta_half, zeta_half = 0.501, 0.501, 0.501   # BL stretching half-domain

target_metric = 43.0           # Target integer metric for shift-and-invert
num_modes = 50                 # Number of eigenpairs to extract
krylov_size = 150              # Krylov subspace size (must be > num_modes)

# ==========================================
# Execution Engine
# ==========================================
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    global_start = time.time()

    N_total    = (Nx + 1) * (Ny + 1) * (Nz + 1)
    N_interior = (Nx - 1) * (Ny - 1) * (Nz - 1)
    target_sigma = -target_metric * (np.pi**2 / 4)

    if rank == 0:
        print("==========================================")
        print(" TRI-GLOBAL HPC ENGINE (PETSc/SLEPc/MPI)")
        print("==========================================")
        print(f"  MPI ranks    : {size}")
        print(f"  Grid         : {Nx}×{Ny}×{Nz}  (order q={q})")
        print(f"  Total DOF    : {N_total:,}")
        print(f"  Interior DOF : {N_interior:,}")
        print(f"  BL stretch   : ξ½={xi_half}, η½={eta_half}, ζ½={zeta_half}")
        print(f"  Target metric: {target_metric}  →  σ={target_sigma:.4f}")
        print(f"  Modes / NCV  : {num_modes} / {krylov_size}")
        print("==========================================")
        print(f"\n[1/3] Assembling operators...")

    # All cores assemble the baseline SciPy operators
    t0 = time.time()
    A_global, B_global, x, y, z = assemble_matrices(
        Nx, Ny, Nz, q, xi_half, eta_half, zeta_half
    )
    t_assemble = time.time() - t0

    if rank == 0:
        nnz_A   = A_global.nnz
        density = 100.0 * nnz_A / N_total**2
        print(f"  Done in {t_assemble:.2f}s")
        print(f"  A: {N_total}×{N_total},  nnz={nnz_A:,}  ({density:.4f}% fill)")
        print(f"\n[2/3] Distributing matrices and running SLEPc...")

    # SLEPc handles MPI distribution automatically inside this function
    int_metrics, lambda_sq, evecs = solve_evp(
        A_global, B_global, target_metric, num_modes, krylov_size
    )

    # 3. I/O Export (Strictly isolated to Rank 0)
    if rank == 0 and int_metrics is not None:
        print(f"\n[3/3] Exporting results to output/...")

        print("\n--- Top Extracted Modes (target metric = {:.0f}) ---".format(target_metric))
        print(f"  {'#':>4}  {'Int. Metric':>13}  {'λ²':>14}  {'|ε|':>10}")
        print(f"  {'─'*4}  {'─'*13}  {'─'*14}  {'─'*10}")
        for i in range(min(10, len(int_metrics))):
            err = abs(int_metrics[i] - round(int_metrics[i]))
            print(f"  {i:>4}  {int_metrics[i]:>13.6f}  {lambda_sq[i]:>14.6f}  {err:>10.2e}")

        os.makedirs("output/eigenvalues", exist_ok=True)
        os.makedirs("output/eigenvectors", exist_ok=True)

        eig_data = np.column_stack((int_metrics, lambda_sq))
        np.savetxt("output/eigenvalues/helmholtz_eigenvalues.txt", eig_data, fmt="%.8e",
                   header="Integer_Metric Lambda_Squared", comments="")
        np.savetxt("output/eigenvalues/grid_x.txt", x, fmt="%.8e")
        np.savetxt("output/eigenvalues/grid_y.txt", y, fmt="%.8e")
        np.savetxt("output/eigenvalues/grid_z.txt", z, fmt="%.8e")
        np.savetxt("output/eigenvectors/helmholtz_eigenvectors.csv",
                   np.real(evecs), delimiter=",", fmt="%.8e")

        total_time = time.time() - global_start
        print("\n==========================================")
        print(f" TOTAL EXECUTION TIME : {total_time:.2f}s")
        print("==========================================")
