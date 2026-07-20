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
# Master Controls — all overridable via env vars
# ==========================================
Nx = int(os.environ.get("ENDGAME_NX", 30))
Ny = int(os.environ.get("ENDGAME_NY", Nx))
Nz = int(os.environ.get("ENDGAME_NZ", Nx))
q  = int(os.environ.get("ENDGAME_Q",  6))
xi_half   = float(os.environ.get("ENDGAME_XI_HALF",   0.501))
eta_half  = float(os.environ.get("ENDGAME_ETA_HALF",  0.501))
zeta_half = float(os.environ.get("ENDGAME_ZETA_HALF", 0.501))
target_metric = float(os.environ.get("ENDGAME_TARGET", 43.0))
num_modes     = int(os.environ.get("ENDGAME_MODES", 100))
krylov_size   = int(os.environ.get("ENDGAME_NCV",   300))
imag_shift = float(os.environ.get("ENDGAME_IMAG_SHIFT", 0.0))
# Non-zero adds i*imag_shift*B to A: eigenvalues become lambda_j + i*imag_shift

# ==========================================
# Execution Engine
# ==========================================
if __name__ == "__main__":
    # Move every MPI rank to this script's own directory.
    # mpiexec may spawn remote ranks from HOME; this one line
    # ensures all ranks see the same FDq/, src/, and output/ layout.
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
    del _os

    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    # Wrap everything in try/except so any unhandled exception on ANY rank
    # immediately aborts ALL ranks via MPI.Abort. This prevents the common
    # failure mode where one rank crashes and others stall forever in a
    # collective call (bcast, assemblyEnd, eps.solve, etc.).
    try:
        global_start = time.time()
        N_total    = (Nx + 1) * (Ny + 1) * (Nz + 1)
        N_interior = (Nx - 1) * (Ny - 1) * (Nz - 1)
        target_sigma = -target_metric * (np.pi**2 / 4)

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
            print(f"  Grid         : {Nx}x{Ny}x{Nz}  (q={q})")
            print(f"  Total DOF    : {N_total:,}")
            print(f"  Target metric: {target_metric}  ->  sigma={target_sigma:.4f}")
            print(f"  Modes / NCV  : {num_modes} / {krylov_size}")
        if imag_shift:
            print(f"  Imag shift   : {imag_shift}  (complex eigenvalues active)")
            print(f"  Output dir   : {out_dir}")
            print("==========================================")
            print(f"\n[1/3] Assembling operators...")

        t0 = time.time()
        A_petsc, B_petsc, x, y, z = assemble_distributed(
            Nx, Ny, Nz, q, xi_half, eta_half, zeta_half, imag_shift=imag_shift
        )
        t_assemble = time.time() - t0

        info = A_petsc.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
        if rank == 0:
            nnz_A   = int(info["nz_used"])
            density = 100.0 * nnz_A / N_total**2
            print(f"  Done in {t_assemble:.2f}s")
            print(f"  A: {N_total}x{N_total},  nnz={nnz_A:,}  ({density:.4f}% fill)")
            print(f"\n[2/3] Running SLEPc eigensolver...")

        int_metrics, lambda_sq, evecs, solver_timing = solve_evp(
            A_petsc, B_petsc, target_metric, num_modes, krylov_size
        )

        A_petsc.destroy()
        B_petsc.destroy()

        if rank == 0 and int_metrics is not None:
            print(f"\n[3/3] Exporting results to {out_dir}/...")
            print("\n--- Top Modes (target={:.0f}) ---".format(target_metric))
            print(f"  {'#':>4}  {'Int.Metric':>12}  {'lambda^2':>13}  {'|eps|':>9}")
            for i in range(min(10, len(int_metrics))):
                err = abs(int_metrics[i] - round(int_metrics[i]))
                print(f"  {i:>4}  {int_metrics[i]:>12.6f}  {lambda_sq[i]:>13.6f}  {err:>9.2e}")

            os.makedirs(eig_dir, exist_ok=True)
            os.makedirs(vec_dir, exist_ok=True)

            eig_data = np.column_stack((int_metrics, lambda_sq))
            np.savetxt(os.path.join(eig_dir, "helmholtz_eigenvalues.txt"),
                       eig_data, fmt="%.8e",
                       header="Integer_Metric Lambda_Squared", comments="")
            np.savetxt(os.path.join(eig_dir, "grid_x.txt"), x, fmt="%.8e")
            np.savetxt(os.path.join(eig_dir, "grid_y.txt"), y, fmt="%.8e")
            np.savetxt(os.path.join(eig_dir, "grid_z.txt"), z, fmt="%.8e")
            np.savetxt(os.path.join(vec_dir, "helmholtz_eigenvectors.csv"),
                       np.real(evecs), delimiter=",", fmt="%.8e")

            total_time = time.time() - global_start
            timing = {
                "run_tag": run_tag, "Nx": Nx, "Ny": Ny, "Nz": Nz, "q": q,
                "mpi_ranks": size, "N_total": N_total, "N_interior": N_interior,
                "nnz_A": nnz_A, "target_metric": target_metric,
                "num_modes": num_modes, "krylov_size": krylov_size,
                "t_assemble_s": round(t_assemble, 3),
                "t_mumps_s":    solver_timing.get("t_mumps_s", -1),
                "t_krylov_s":   solver_timing.get("t_krylov_s", -1),
                "t_total_s":    round(total_time, 3),
                "nconv":        solver_timing.get("nconv", -1),
            }
            with open(os.path.join(out_dir, "timing.json"), "w") as f:
                json.dump(timing, f, indent=2)

            print("\n==========================================")
            print(f" TOTAL EXECUTION TIME : {total_time:.2f}s")
            print("==========================================")

    except Exception as exc:
        # Print the full traceback on this rank, then abort all ranks immediately.
        # Without Abort(), other ranks stall forever in the next collective call.
        import traceback
        print(f"\n[RANK {rank}] FATAL EXCEPTION — aborting all ranks:\n",
              flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        comm.Abort(1)
