import os
import sys
import time
import numpy as np

# PETSc must be initialised BEFORE mpi4py is imported.
# petsc4py.init() parses command-line arguments (e.g. -log_view, -ksp_monitor)
# and sets up the PETSc internal state that MPI will later attach to.
# If this line comes after `from mpi4py import MPI`, PETSc cannot control
# MPI initialisation and the run will fail or behave incorrectly.
import petsc4py
petsc4py.init(sys.argv)

# mpi4py provides the Python interface to MPI (Message Passing Interface).
# MPI is the standard protocol that lets multiple processes communicate.
from mpi4py import MPI

# PETSc (Portable, Extensible Toolkit for Scientific Computation):
# sparse matrix algebra, linear solvers, and data structures for distributed computing.
from petsc4py import PETSc

from src.operators import assemble_distributed
from src.solver import solve_evp

# ==========================================
# Master Controls
# ==========================================

# Number of grid intervals in each spatial direction.
# The actual number of nodes per direction is Nx+1, Ny+1, Nz+1.
# Total degrees of freedom = (Nx+1)*(Ny+1)*(Nz+1).
Nx, Ny, Nz = 30, 30, 30

# Stencil size in the Fornberg/Hermanns FD scheme.
# q is the FULL number of points used in each derivative stencil,
# NOT a half-width. q=6 means 6 nodes participate in each derivative.
# Higher q = higher-order accuracy but wider stencil (more nonzeros per row).
q = 6

# Boundary-layer stretching parameter for each direction.
# This is the value of the reference coordinate ξ at which the physical
# coordinate reaches half the domain width. Values close to 0.5 give
# very strong clustering near the boundary; values near 1.0 give uniform spacing.
# 0.501 is nearly uniform — suitable for the Helmholtz test case.
xi_half, eta_half, zeta_half = 0.501, 0.501, 0.501

# The integer metric M = n² + m² + k² we want to target.
# For the 3-D Helmholtz equation on [-1,1]³ with Dirichlet BCs, the exact
# eigenvalues are λ = -π²(n²+m²+k²)/4, so M is always an exact integer.
# We look for all modes whose M is close to this value.
target_metric = 43.0

# How many eigenpairs (eigenvalue + eigenvector) to extract from the solver.
# The solver may return more than this if the Krylov space converges additional modes.
num_modes = 50

# Size of the Krylov subspace built by the Krylov-Schur algorithm before
# it restarts. Must be strictly greater than num_modes.
# Larger value = more work per iteration but fewer restarts needed.
# Rule of thumb: krylov_size ~ 3 * num_modes.
krylov_size = 150

# ==========================================
# Execution Engine
# ==========================================
if __name__ == "__main__":
    # MPI.COMM_WORLD is the global communicator — it represents all P processes
    # that were launched by mpiexec. Every process can use it to send/receive data.
    comm = MPI.COMM_WORLD

    # Each process has a unique rank from 0 to P-1.
    # Rank 0 is the "master" — it handles terminal output and file I/O.
    rank = comm.Get_rank()

    # Total number of MPI processes (set by `mpiexec -n P`).
    size = comm.Get_size()

    global_start = time.time()

    # Total number of grid nodes = (Nx+1)*(Ny+1)*(Nz+1).
    # This equals the size of the matrices A and B.
    N_total    = (Nx + 1) * (Ny + 1) * (Nz + 1)

    # Interior nodes = nodes NOT on any face of the cube.
    # Only these contribute active equations; boundary nodes get identity rows.
    N_interior = (Nx - 1) * (Ny - 1) * (Nz - 1)

    # The spectral shift σ for shift-and-invert.
    # Analytical eigenvalues of the Helmholtz operator are λ = -π²·M/4 (negative).
    # We shift to σ = -target_metric·π²/4, so eigenvalues NEAR our target become
    # the LARGEST in the shifted-inverted problem — easy to find with Krylov methods.
    target_sigma = -target_metric * (np.pi**2 / 4)

    # Only rank 0 prints. All other ranks skip this block silently.
    # This avoids 8 identical copies of the header appearing in the terminal.
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

    # assemble_distributed() is a collective call — all ranks execute it together.
    # Each rank builds only its own slice of the PETSc matrices A and B.
    # No global SciPy matrix is ever constructed anywhere.
    # Returns: A_petsc (Laplacian + BCs), B_petsc (mass/identity matrix),
    #          x, y, z (physical-space grid coordinates as 1-D numpy arrays).
    t0 = time.time()
    A_petsc, B_petsc, x, y, z = assemble_distributed(
        Nx, Ny, Nz, q, xi_half, eta_half, zeta_half
    )
    t_assemble = time.time() - t0

    # getInfo() with GLOBAL_SUM is a collective MPI operation:
    # all ranks contribute their local nnz count and PETSc returns the global total.
    # ALL ranks must call this line — it would deadlock if only rank 0 called it.
    # The result is available on every rank, but only rank 0 prints it.
    info  = A_petsc.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
    if rank == 0:
        nnz_A   = int(info['nz_used'])         # total non-zeros in A across all ranks
        density = 100.0 * nnz_A / N_total**2   # fill fraction (almost always < 0.1%)
        print(f"  Done in {t_assemble:.2f}s")
        print(f"  A: {N_total}×{N_total},  nnz={nnz_A:,}  ({density:.4f}% fill)")
        print(f"\n[2/3] Running SLEPc eigensolver...")

    # solve_evp() is fully collective — all ranks participate in the MUMPS
    # factorisation and Krylov-Schur iterations.
    # Returns (on rank 0): integer metrics, λ² values, eigenvector matrix.
    # Returns (on all other ranks): (None, None, None) — they have no data to return.
    int_metrics, lambda_sq, evecs = solve_evp(
        A_petsc, B_petsc, target_metric, num_modes, krylov_size
    )

    # File export is strictly rank 0. Other ranks do nothing here.
    # `int_metrics is not None` guards against the case where no modes converged.
    if rank == 0 and int_metrics is not None:
        print(f"\n[3/3] Exporting results to output/...")

        print("\n--- Top Extracted Modes (target metric = {:.0f}) ---".format(target_metric))
        print(f"  {'#':>4}  {'Int. Metric':>13}  {'λ²':>14}  {'|ε|':>10}")
        print(f"  {'─'*4}  {'─'*13}  {'─'*14}  {'─'*10}")
        for i in range(min(10, len(int_metrics))):
            # |ε| is the distance from the nearest integer.
            # For the exact analytical answer, int_metric[i] would be a perfect integer.
            # Non-zero |ε| is pure numerical discretisation error.
            err = abs(int_metrics[i] - round(int_metrics[i]))
            print(f"  {i:>4}  {int_metrics[i]:>13.6f}  {lambda_sq[i]:>14.6f}  {err:>10.2e}")

        # exist_ok=True means mkdir does not raise an error if the directory exists.
        os.makedirs("output/eigenvalues", exist_ok=True)
        os.makedirs("output/eigenvectors", exist_ok=True)

        # Stack integer metrics and λ² into a two-column array for easy plotting.
        eig_data = np.column_stack((int_metrics, lambda_sq))
        np.savetxt("output/eigenvalues/helmholtz_eigenvalues.txt", eig_data, fmt="%.8e",
                   header="Integer_Metric Lambda_Squared", comments="")

        # Save the physical grid coordinates so post-processing scripts know
        # where each node sits in space.
        np.savetxt("output/eigenvalues/grid_x.txt", x, fmt="%.8e")
        np.savetxt("output/eigenvalues/grid_y.txt", y, fmt="%.8e")
        np.savetxt("output/eigenvalues/grid_z.txt", z, fmt="%.8e")

        # Eigenvectors: each column is one mode shape (real part only — the
        # Helmholtz operator is real-symmetric so imaginary parts are negligible).
        np.savetxt("output/eigenvectors/helmholtz_eigenvectors.csv",
                   np.real(evecs), delimiter=",", fmt="%.8e")

        total_time = time.time() - global_start
        print("\n==========================================")
        print(f" TOTAL EXECUTION TIME : {total_time:.2f}s")
        print("==========================================")
