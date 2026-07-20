# ENDGAME — Algorithm & Code Walkthrough

This document explains every layer of the codebase from scratch: the mathematics,
the numerical method, the coordinate mappings, and every MPI/PETSc/SLEPc concept.
No prior knowledge of parallel computing is assumed.

---

## 1. The Problem We Are Solving

We solve the **3-D Helmholtz eigenvalue problem** on the cube [−1, 1]³:

```
∇²u = λ u        with u = 0 on all six faces  (Dirichlet BCs)
```

where ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z² is the Laplacian.

The **analytical solution** on [−1, 1]³ is known exactly:

```
λ_{n,m,k} = −π²(n² + m² + k²) / 4        n, m, k = 1, 2, 3, ...
```

Each eigenvalue is labelled by three positive integers (n, m, k).
The **integer metric** M = n² + m² + k² is an integer, so we can verify
numerical accuracy by checking how close the computed M is to an exact integer.

For example, `target_metric = 43` means we want all mode triplets satisfying
n² + m² + k² = 43 (e.g. (1,3,5) since 1+9+25=35 — wait, 43 has no obvious
decomposition, which is what makes it interesting as a test: the solver
must find whichever triplets sum to 43 without being told which ones they are).

**Why Helmholtz?** Because the exact answer is known, so every digit of
numerical error is measurable. The same solver will later be pointed at
linearised Navier–Stokes operators where there is no analytical answer.

---

## 2. Discretisation: The FDq Finite-Difference Method

We discretise the domain with a 3-D grid of (Nx+1) × (Ny+1) × (Nz+1) nodes.
In each direction we use a **Fornberg-type high-order finite-difference** stencil
of size `q` (i.e. `q` grid points participate in each derivative approximation).

The key object is the **differentiation matrix** D¹ and D² for each direction.
These are (N+1) × (N+1) dense matrices where:

```
(D¹ u)_i  ≈  du/dξ  at node i
(D² u)_i  ≈  d²u/dξ² at node i
```

Only `q` entries per row of D¹ and D² are non-zero (the stencil nodes).

### Where does the Fortran binary come in?

Computing D¹ and D² via the Fornberg/Hermanns algorithm requires solving
an optimisation problem for the grid node positions (Gauss-Lobatto-style
optimal grids). This is done in Fortran for numerical precision.

`FDq/bin/FDq` (the binary built from `FDq/src/`) takes:
- **input**: N+1 (number of nodes), q (stencil size)
- **output**: D¹, D² as text files, plus the grid coordinates η

The Python wrapper (`src/FDq.py`) writes the input, calls the binary via
`subprocess`, and reads back the results.

---

## 3. Coordinate Mappings (src/grid.py)

The Fortran binary generates D¹, D² on a uniform reference grid η ∈ [−1, 1].
We need to transform these operators to a physical grid that is **stretched
near the boundaries** (boundary-layer clustering).

### Step 1: Boundary-Layer Map (`BL_Map`)

```
x = l (1 + ξ) / (1 + s − ξ)
```

This maps ξ ∈ [−1, 1] → x ∈ [0, ∞), with heavy clustering near x = 0.
Parameters:
- `half`: the ξ value at which x = domain_size/2 (controls stretching intensity)
- `l`, `s`: derived length and stretching parameters

**Why do we need the Jacobians J and J2?**
When we change variables from ξ to x, the chain rule gives:

```
d/dx = (dξ/dx) · d/dξ = J · d/dξ              (J = dξ/dx)
d²/dx² = J² · d²/dξ² + J₂ · d/dξ             (J₂ = d²ξ/dx²)
```

So the second-derivative operator in physical space is:

```
d²u/dx² ≈ [J² · D² + J₂ · D¹] u
```

The code computes J (called `Jx` in operators.py) and J₂ (called `Jxx`)
at every node, and multiplies the differentiation matrices accordingly.

### Step 2: Physical Scaling (`Map1D`)

After BL_Map, x lives in [0, ∞). We rescale linearly to [−1, 1]:

```
x_physical = scale · (x − x[0]) + (−1)
```

This adds a second Jacobian `Jx_s` (a single scalar = dξ_physical/dx) which
further scales the second-derivative operator.

### Combined second-derivative operator (in operators.py):

```python
dxx = (Jx_s**2) * (diag(Jx**2) @ D2x  +  diag(Jxx) @ D1x)
```

`dxx[i, j]` is the weight that the value at node j contributes to d²u/dx²
at node i, after both coordinate transforms.

---

## 4. Assembling the 3-D Operator (src/operators.py)

### The Kronecker product idea (conceptual)

If we had 1-D second-derivative matrices dxx, dyy, dzz, the 3-D Laplacian on
the full tensor-product grid would be:

```
A = kron(Iz, kron(Iy, dxx))   ← x-derivatives
  + kron(Iz, kron(dyy, Ix))   ← y-derivatives
  + kron(dzz, kron(Iy, Ix))   ← z-derivatives
```

where Ix, Iy, Iz are identity matrices of size Nx_n, Ny_n, Nz_n.

**kron(A, B)** means: replace each entry A[i,j] with the block A[i,j]·B.
So `kron(Iz, kron(Iy, dxx))` tiles dxx across Ny·Nz blocks, one for each
(iy, iz) combination. Conceptually, x-derivatives don't mix y or z nodes.

### Index mapping (Fortran column-major order)

We flatten the 3-D grid into a 1-D array with index:

```
i = ix  +  iy · (Nx+1)  +  iz · (Nx+1)(Ny+1)
```

x varies fastest (column-major). The inverse mapping is:

```
ix = i mod (Nx+1)
iy = floor(i / (Nx+1))  mod  (Ny+1)
iz = floor(i / (Nx+1)(Ny+1))
```

### How each row of the 3-D matrix is computed (without building the full matrix)

For row `i` (corresponding to point `(ix, iy, iz)`):

**x-contribution:** The Kronecker structure means we use row `ix` of `dxx`.
For each non-zero entry `dxx[ix, ix_col]`, the 3-D column index is:
```
j = ix_col  +  iy · Nx_n  +  iz · Nx_n · Ny_n
```
(only the x-index changes; y and z stay fixed).

**y-contribution:** Use row `iy` of `dyy`. For each `dyy[iy, iy_col]`:
```
j = ix  +  iy_col · Nx_n  +  iz · Nx_n · Ny_n
```
(only the y-index changes).

**z-contribution:** Use row `iz` of `dzz`. For each `dzz[iz, iz_col]`:
```
j = ix  +  iy · Nx_n  +  iz_col · Nx_n · Ny_n
```
(only the z-index changes).

The diagonal entry `j = i` appears in all three contributions and is summed
naturally by the `stencil` dictionary.

### Boundary conditions

Rows where `ix=0`, `ix=Nx`, `iy=0`, `iy=Ny`, `iz=0`, or `iz=Nz` are boundary nodes.
Dirichlet BCs are enforced by replacing those rows with identity:

```
A[i, i] = 1.0  (all other entries in that row = 0)
B[i, i] = 0.0  (this makes the boundary eigenvalue "infinite" → not physical)
```

Interior rows get the Laplacian stencil in A and 1.0 on the diagonal of B.

### The generalised eigenvalue problem structure

```
A x = λ B x
```

- **A**: Laplacian matrix with identity rows at boundaries
- **B**: Diagonal matrix: 1 at interior nodes, 0 at boundary nodes

For an interior node: A x = λ B x → [Laplacian] x = λ x → eigenvalue problem.
For a boundary node: A[i,:] x = 1·x_i = λ·0·x_i → 1 = 0 (inconsistent unless
treated specially by the solver → boundary modes have eigenvalue ∞, never extracted).

---

## 5. MPI Parallelism — A Complete Explanation

### What MPI does

When you run `mpiexec -n 8 python main.py`, MPI starts **8 identical copies**
of the Python process simultaneously. They all run the same code. Without
any rank-checking, they would all do exactly the same thing (wasteful or broken).

Each process is assigned a unique integer ID called its **rank**, from 0 to P−1
where P is the total number of processes. Rank 0 is conventionally the "master".

### Key MPI objects and calls

```python
comm = MPI.COMM_WORLD      # the communicator — the group of all P processes
rank = comm.Get_rank()     # this process's ID (0 to P-1)
size = comm.Get_size()     # total number of processes P
```

**`MPI.COMM_WORLD`**: the global communicator. Think of it as a phone directory
that includes all processes and allows them to talk to each other.

### Pattern 1: Rank guard (rank 0 does file I/O)

```python
if rank == 0:
    # only rank 0 executes this
    write file, call Fortran binary, read result
```

Without this guard, all 8 processes would simultaneously try to write the same
input file — a race condition. One process would overwrite another's write.

### Pattern 2: Broadcast (share data from rank 0 to all)

```python
result = comm.bcast(result, root=0)
```

`bcast` (broadcast) is a **collective operation** — all processes must call it.
Rank 0 sends its `result` to every other rank. All other ranks receive it.
Before `bcast`: result is populated only on rank 0, None on all others.
After `bcast`: result has the same value on all ranks.

### Pattern 3: Distributed matrix (each rank owns a row slice)

PETSc's `MPIAIJ` matrix format partitions rows across ranks automatically.
With 8 ranks and 29,791 rows, each rank owns approximately 29,791/8 ≈ 3,724 rows.

```python
rstart, rend = A_petsc.getOwnershipRange()
# Rank 0: rstart=0,    rend=3724
# Rank 1: rstart=3724, rend=7448
# ...
# Rank 7: rstart=26066, rend=29791
```

Each rank only assembles the rows in `[rstart, rend)`. No rank ever builds
the whole matrix.

### Pattern 4: Collective operations

Some PETSc/MPI operations require all ranks to participate together:

```python
A_petsc.assemblyBegin(); A_petsc.assemblyEnd()
```

This finalises the matrix. It allows PETSc to communicate off-rank entries
(if a rank inserted a value in a row owned by another rank). All ranks must
call this or the program deadlocks.

```python
info = A_petsc.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
```

GLOBAL_SUM means: each rank reports its local nnz (non-zero count), PETSc
sums them across all ranks and returns the total to every rank. All ranks
must call this, but only rank 0 uses the result.

### Pattern 5: Scatter (gather distributed vector to rank 0)

After solving, eigenvectors are split across all ranks (each rank holds
its rows). To write them to disk, rank 0 needs the full vector.

```python
scatter, v_seq = PETSc.Scatter.toZero(vr)
scatter.scatter(vr, v_seq, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
```

`toZero(vr)`: creates a scatter plan that will collect the full distributed
vector `vr` into the sequential vector `v_seq`, which only exists on rank 0.
`FORWARD`: send data from distributed → sequential (as opposed to reverse).
After this, `v_seq.getArray()` on rank 0 is the full eigenvector.

---

## 6. PETSc Matrix Format (MPIAIJ)

PETSc stores distributed sparse matrices in **MPIAIJ** format:

```
|  diagonal block  | off-diagonal block |
|  (rows owned by  | (columns owned by  |
|   this rank)     |  other ranks)      |
```

For each owned row, entries whose column index falls within the same rank's
row range are "diagonal" (d_nnz), others are "off-diagonal" (o_nnz).

The `setPreallocationNNZ((d_nnz, o_nnz))` call tells PETSc in advance how many
non-zeros per row to expect. This avoids expensive memory reallocations during
assembly. We use `3*q` as a conservative upper bound (q entries per direction).

`setOption(NEW_NONZERO_ALLOCATION_ERR, False)` turns off the error that PETSc
would raise if we exceed the allocation estimate. Safer for the first run;
you can turn it on once the preallocation is tuned.

---

## 7. The Eigensolver (src/solver.py)

### The mathematical problem

We want to find all (λ, x) satisfying:

```
A x = λ B x
```

where λ is the eigenvalue and x is the eigenvector (mode shape).

For our Helmholtz problem, the eigenvalues λ are negative numbers near
−π²(n²+m²+k²)/4. We look for modes near `target_sigma = −43·π²/4 ≈ −106.1`.

### Why we cannot use a standard eigensolver

We want **interior** eigenvalues (near a target), not the largest or smallest.
Standard Krylov methods (ARPACK, power iteration) naturally find extreme
eigenvalues. Finding interior ones is hard unless we transform the problem.

### Shift-and-invert transformation

Define the **shifted inverse** operator:

```
C = (A − σ B)⁻¹ B
```

Then C has eigenvalues θ = 1 / (λ − σ). If λ is close to σ, then θ is large.
So the eigenvalues of A nearest to σ become the **largest** eigenvalues of C,
which any Krylov method can find easily.

The code tells SLEPc to use this transformation:

```python
st.setType(SLEPc.ST.Type.SINVERT)   # use shift-and-invert
st.setShift(target_sigma)            # σ = −43·π²/4
```

### MUMPS: computing (A − σ B)⁻¹

Applying C to a vector v means solving (A − σ B) y = B v for y.
This requires factorising (A − σ B) once, then doing back-substitution for
each Krylov iteration. MUMPS (Multifrontal Massively Parallel Sparse Solver)
computes the **LU factorisation** of (A − σ B) in parallel across all MPI ranks.

```python
ksp.setType('preonly')          # no iterative solver — the preconditioner IS the solve
pc.setType('lu')                # use LU factorisation as preconditioner
pc.setFactorSolverType('mumps') # MUMPS does the parallel LU
```

`eps.setUp()` triggers this factorisation. This is the expensive one-time cost
(25.54s on 8 cores for the 30³ test case).

### Krylov-Schur iteration

Once MUMPS has factorised (A − σ B), SLEPc runs the **Krylov-Schur** algorithm:

1. Start with a random vector v₀
2. Build a Krylov subspace: v₀, C·v₀, C²·v₀, ..., C^(NCV)·v₀
3. Project the problem onto this subspace (now NCV × NCV, cheap)
4. Solve the small projected eigenvalue problem
5. Check which approximate eigenpairs have converged (residual < tolerance)
6. Deflate converged pairs and restart with the remaining subspace

`nev = num_modes = 50`: ask for 50 eigenpairs.
`ncv = krylov_size = 150`: build a Krylov subspace of size 150 before restarting.
Larger NCV = more work per restart but faster convergence. Must be > nev.

The solver converged **76 modes** (> 50 requested) because the Krylov space
was rich enough to resolve more modes near the target than requested.

### Post-processing

```python
lambda_sq = -np.real(eigenvalues_raw)
```

SLEPc returns the raw eigenvalues λ (negative). We negate to get positive
values λ² = π²(n²+m²+k²)/4.

```python
analytical_integers = lambda_sq / (np.pi**2 / 4)
```

Divides by π²/4 to recover n²+m²+k². For a perfect computation this is
exactly an integer. The error |ε| = |round(M) − M| measures discretisation error.

---

## 8. Execution Flow Summary

```
mpiexec -n 8 python main.py
         │
         ├── All 8 ranks enter main.py simultaneously
         │
         ├── [RANK 0 ONLY] Print header
         │
         ├── assemble_distributed()   ← operators.py
         │    ├── [RANK 0] Write input_size.dat
         │    ├── [RANK 0] Run Fortran binary → D1, D2, grid
         │    ├── [ALL RANKS] comm.bcast() → all ranks get D1, D2, grid
         │    ├── [ALL RANKS] Apply coordinate maps → dxx, dyy, dzz
         │    ├── [ALL RANKS] Create empty PETSc MPIAIJ matrices A, B
         │    ├── [RANK k] Fill rows [rstart_k, rend_k) of A and B
         │    └── [ALL RANKS] assemblyBegin/End → finalise matrices
         │
         ├── [ALL RANKS] getInfo(GLOBAL_SUM) → total nnz count
         ├── [RANK 0] Print matrix statistics
         │
         ├── solve_evp()   ← solver.py
         │    ├── [ALL RANKS] Create SLEPc EPS object with A, B
         │    ├── [ALL RANKS] Configure shift-and-invert + MUMPS
         │    ├── [ALL RANKS] eps.setUp() → MUMPS factorises (A − σB) in parallel
         │    ├── [ALL RANKS] eps.solve() → Krylov-Schur iterations
         │    ├── [ALL RANKS] Loop over converged modes:
         │    │    ├── eps.getEigenpair(i, vr, vi) → vr is distributed
         │    │    └── scatter.scatter() → collect vr onto rank 0
         │    └── [RANK 0] Sort, compute integer metrics, return arrays
         │
         └── [RANK 0] Write eigenvalues and eigenvectors to output/
```

---

## 9. Output Interpretation

From the test run (30³ grid, q=6, target_metric=43):

```
#    Int. Metric      λ²           |ε|
0    42.998928    106.095602   1.07e-03
1    42.998928    106.095602   1.07e-03
2    42.998928    106.095602   1.07e-03
```

- **Int. Metric ≈ 43.00**: the solver found modes with n²+m²+k² = 43. ✓
- **λ² ≈ 106.10**: λ² = 43·π²/4 = 106.13 analytically. Close. ✓
- **|ε| ≈ 1e-3**: the integer metric is off from exactly 43 by 0.001.
  This is numerical error from the 30-node grid. Increase Nx, Ny, Nz
  to drive this toward machine precision.
- **Modes 0, 1, 2 identical**: one mode has three degenerate copies
  (same λ but different mode shapes — a consequence of the cubic geometry).

The `|ε|` column is your primary convergence indicator for the scaling study.
As N increases, |ε| should decrease as O(h^q) where h ~ 1/N.

---

## 10. Scaling Test Strategy

For large-scale runs, the bottleneck progression is:

| N (per direction) | DOF       | Assembly cost | MUMPS cost | Memory per rank |
|---|---|---|---|---|
| 30  | 30K   | ~0.3s  | ~25s   | ~1 MB  |
| 60  | 226K  | ~2s    | ~300s  | ~8 MB  |
| 100 | 1M    | ~10s   | ~hours | ~30 MB |

With the distributed assembly, **memory scales as O(N_total / N_ranks)** —
doubling ranks halves memory per rank. MUMPS also scales well with more ranks
(parallel factorisation distributes the work).

To run a scaling study: fix N, vary P (MPI ranks), record MUMPS time and
Krylov-Schur time separately. Both should ideally scale as O(1/P) (perfect scaling).
