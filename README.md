# ENDGAME

Tri-global hydrodynamic stability solver using high-order finite differences, PETSc/SLEPc, and MPI.

Solves the generalised eigenvalue problem arising from a 3-D Helmholtz operator discretised on a stretched boundary-layer grid. The solver targets a user-specified spectral shift and extracts the nearest eigenpairs using the Krylov–Schur method with MUMPS as the shift-and-invert factoriser.

---

## Project layout

```
ENDGAME/
├── main.py                  # Entry point — controls and execution engine
├── Makefile                 # Top-level build (compiles FDq Fortran binary)
├── environment.yml          # Conda environment (Python + PETSc/SLEPc stack)
├── src/
│   ├── FDq.py               # Python wrapper for the FDq Fortran binary
│   ├── grid.py              # Boundary-layer and physical-space coordinate maps
│   ├── operators.py         # 3-D Kronecker assembly of the Laplacian operator
│   └── solver.py            # SLEPc eigensolver and result gathering
├── FDq/
│   ├── bin/Makefile         # Fortran build rules (cross-platform BLAS/LAPACK)
│   ├── src/                 # Fortran 90 sources for differentiation matrices
│   ├── inputs/              # Runtime input written by Python before each call
│   └── output/              # Runtime output read back by Python after each call
├── jobs/
│   ├── run_macbook.sh       # Local 8-core MPI run
│   └── submit_hpc.sh        # SLURM job script (16 ranks, 1 h)
└── output/
    ├── eigenvalues/         # helmholtz_eigenvalues.txt + grid_*.txt
    └── eigenvectors/        # helmholtz_eigenvectors.csv
```

---

## Dependencies

| Layer | Packages |
|---|---|
| Python | `python=3.10`, `numpy`, `scipy` |
| MPI | `mpi4py` |
| PETSc / SLEPc | `petsc`, `petsc4py`, `slepc`, `slepc4py` |
| Fortran compiler | `gfortran` (macOS: via Xcode CLT or Homebrew; Linux: gcc) |
| BLAS / LAPACK | macOS: Accelerate framework (automatic); Linux: MKL or OpenBLAS |
| Sparse direct solver | MUMPS (bundled with the PETSc conda-forge build) |

All Python / MPI / PETSc dependencies are pinned in `environment.yml`.

---

## Setup

### 1 — Create the conda environment

```bash
conda env create -f environment.yml
conda activate tri_engine
```

### 2 — Build the Fortran binary

```bash
make          # produces FDq/bin/FDq
```

On Linux the Makefile auto-detects Intel MKL (if `$MKL_ROOT` exists) and falls back to OpenBLAS.

---

## Running

### Local (MacBook / workstation)

```bash
bash jobs/run_macbook.sh      # 8 MPI ranks, unbuffered output
```

Or directly:

```bash
OMP_NUM_THREADS=1 mpiexec -n 8 python -u main.py
```

### HPC (SLURM)

```bash
sbatch jobs/submit_hpc.sh     # 16 ranks, 1-hour wall time
```

Edit `#SBATCH` directives at the top of the script to match your cluster's queue policy.

---

## Master controls

All user-facing parameters live at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `Nx, Ny, Nz` | 30, 30, 30 | Grid points per direction |
| `q` | 6 | FD stencil half-width |
| `xi_half, eta_half, zeta_half` | 0.501 | BL stretching half-domain |
| `target_metric` | 43.0 | Target integer metric for shift-and-invert |
| `num_modes` | 50 | Number of eigenpairs to extract |
| `krylov_size` | 150 | Krylov subspace size (must exceed `num_modes`) |

---

## Output

Results are written to `output/` by rank 0 after convergence:

- `output/eigenvalues/helmholtz_eigenvalues.txt` — integer metric and λ² for each converged mode  
- `output/eigenvalues/grid_{x,y,z}.txt` — physical-space coordinate arrays  
- `output/eigenvectors/helmholtz_eigenvectors.csv` — real part of eigenvectors (column-major)

---

## How it works

1. **FDq** — Rank 0 calls the Fortran binary to compute high-order finite-difference matrices D¹ and D² on a uniform reference grid of N+1 points with stencil half-width q.  
2. **Grid mapping** — The reference grid is stretched via a boundary-layer algebraic map (`BL_Map`) and then rescaled to [−1, 1] (`Map1D`).  
3. **Operator assembly** — Second-derivative operators are assembled in 3-D via Kronecker products. Boundary rows are replaced by identity (Dirichlet BCs) through a mass-matrix masking strategy.  
4. **SLEPc solve** — The SciPy CSR matrices are distributed across MPI ranks as PETSc `MPIAIJ` matrices. A Krylov–Schur eigensolver with shift-and-invert (MUMPS LU) extracts the requested modes near the spectral target.
