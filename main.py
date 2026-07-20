import os
import sys
import time
import json
import numpy as np

# PETSc must be initialised BEFORE mpi4py is imported.
import petsc4py
petsc4py.init(sys.argv)
from mpi4py import MPI
from petsc4py import PETSc
from src.operators import assemble_distributed
from src.solver import solve_evp

# ==========================================
# Master Controls
# All parameters can be overridden via environment variables, e.g.:
#   ENDGAME_NX=60 mpiexec -n 64 python main.py
# ==========================================
Nx = int(os.environ.get("ENDGAME_NX", 30))
Ny = int(os.environ.get("ENDGAME_NY", Nx))   # default: cubic grid
Nz = int(os.environ.get("ENDGAME_NZ", Nx))

# Full Fornberg stencil size (not half-width)
q  = int(os.environ.get("ENDGAME_Q", 6))

# BL stretching: 0.501 = nearly uniform (suitable for Helmholtz test)
xi_half   = float(os.environ.get("ENDGAME_XI_HALF",   0.501))
eta_half  = float(os.environ.get("ENDGAME_ETA_HALF",  0.501))
zeta_half = float(os.environ.get("ENDGAME_ZETA_HALF", 0.501))

# Target integer metric M = n^2+m^2+k^2 for shift-and-invert
target_metric = float(os.environ.get("ENDGAME_TARGET", 43.0))

# Eigenpairs to extract
num_modes   = int(os.environ.get("ENDGAME_MODES", 100))

# Krylov subspace size (must be > num_modes; minimum 200 per study design)
krylov_size = int(os.environ.get("ENDGAME_NCV",  300))

# ==========================================
# Execution Engine
# ==========================================
if __name__ == "__main__":
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    global_start = time.time()

    N_total    = (Nx + 1) * (Ny + 1) * (Nz + 1)
    N_interior = (Nx - 1) * (Ny - 1) * (Nz - 1)
    target_sigma = -target_metric * (np.pi**2 / 4)

    # One output folder per (N, q, P) combination — runs never overwrite each other.
    run_tag = f"N{Nx}_q{q}_P{size}"
    out_dir = os.path.join("output", run_tag)
    eig_dir = os.path.join(out_dir, "eigenvalues")
    vec_dir = os.path.join(out_dir, "eigenvectors")

    if rank == 0:
        print("==========================================")
        print(" TRI-GLOBAL HPC ENGINE (PETSc/SLEPc/MPI)")
        print("==========================================")
        print(f"  Run tag      : {run_tag}")
        print(f"  MPI ranks    : {size}")
        print(f"  Grid         : {Nx}x{Ny}x{Nz}  (order q={q})")
        print(f"  Total DOF    : {N_total:,}")
        print(f"  Interior DOF : {N_interior:,}")
        print(f"  BL stretch   : xi={xi_half}, eta={eta_half}, zeta={zeta_half}")
        print(f"  Target metric: {target_metric}  ->  sigma={target_sigma:.4f}")
        print(f"  Modes / NCV  : {num_modes} / {krylov_size}")
        print(f"  Output dir   : {out_dir}")
        print("==========================================")
        print(f"\n[1/3] Assembling operators...")

    t0 = time.time()
    A_petsc, B_petsc, x, y, z = assemble_distributed(
        Nx, Ny, Nz, q, xi_half, eta_half, zeta_half
    )
    t_assemble = time.time() - t0

    # getInfo(GLOBAL_SUM) is collective — all ranks must call it
    info    = A_petsc.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
    if rank == 0:
        nnz_A   = int(info["nz_used"])
        density = 100.0 * nnz_A / N_total**2
        print(f"  Done in {t_assemble:.2f}s")
        print(f"  A: {N_total}x{N_total},  nnz={nnz_A:,}  ({density:.4f}% fill)")
        print(f"\n[2/3] Running SLEPc eigensolver...")

    # solve_evp returns a 4-tuple; only rank 0 has non-None values
    int_metrics, lambda_sq, evecs, solver_timing = solve_evp(
        A_petsc, B_petsc, target_metric, num_modes, krylov_size
    )

    # Explicitly release PETSc matrices to free distributed memory
    A_petsc.destroy()
    B_petsc.destroy()

    if rank == 0 and int_metrics is not None:
        print(f"\n[3/3] Exporting results to {out_dir}/...")

        print("\n--- Top Extracted Modes (target metric = {:.0f}) ---".format(target_metric))
        print(f"  {'#':>4}  {'Int. Metric':>13}  {'lambda^2':>14}  {'|eps|':>10}")
        print(f"  {'---':>4}  {'---':>13}  {'---':>14}  {'---':>10}")
        for i in range(min(10, len(int_metrics))):
            err = abs(int_metrics[i] - round(int_metrics[i]))
            print(f"  {i:>4}  {int_metrics[i]:>13.6f}  {lambda_sq[i]:>14.6f}  {err:>10.2e}")

        os.makedirs(eig_dir, exist_ok=True)
        os.makedirs(vec_dir, exist_ok=True)

        eig_data = np.column_stack((int_metrics, lambda_sq))
        np.savetxt(os.path.join(eig_dir, "helmholtz_eigenvalues.txt"), eig_data,
                   fmt="%.8e", header="Integer_Metric Lambda_Squared", comments="")
        np.savetxt(os.path.join(eig_dir, "grid_x.txt"), x, fmt="%.8e")
        np.savetxt(os.path.join(eig_dir, "grid_y.txt"), y, fmt="%.8e")
        np.savetxt(os.path.join(eig_dir, "grid_z.txt"), z, fmt="%.8e")
        np.savetxt(os.path.join(vec_dir, "helmholtz_eigenvectors.csv"),
                   np.real(evecs), delimiter=",", fmt="%.8e")

        total_time = time.time() - global_start

        # Structured timing record — read by collect_timing.py after all runs finish
        timing = {
            "run_tag":       run_tag,
            "Nx": Nx, "Ny": Ny, "Nz": Nz,
            "q":             q,
            "mpi_ranks":     size,
            "N_total":       N_total,
            "N_interior":    N_interior,
            "nnz_A":         nnz_A,
            "target_metric": target_metric,
            "num_modes":     num_modes,
            "krylov_size":   krylov_size,
            "t_assemble_s":  round(t_assemble, 3),
            "t_mumps_s":     solver_timing.get("t_mumps_s", -1),
            "t_krylov_s":    solver_timing.get("t_krylov_s", -1),
            "t_total_s":     round(total_time, 3),
            "nconv":         solver_timing.get("nconv", -1),
        }
        with open(os.path.join(out_dir, "timing.json"), "w") as f:
            json.dump(timing, f, indent=2)

        print("\n==========================================")
        print(f" TOTAL EXECUTION TIME : {total_time:.2f}s")
        print("==========================================")
