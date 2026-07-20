import sys
import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


def solve_evp(A_petsc, B_petsc, target_metric, num_modes, krylov_size):
    """
    Solve the generalised eigenvalue problem  A x = λ B x  using SLEPc.

    A_petsc and B_petsc are already distributed PETSc MPIAIJ matrices
    produced by assemble_distributed() — no conversion step is needed.

    Strategy: shift-and-invert with MUMPS
    -------------------------------------
    Standard Krylov methods find extreme eigenvalues (largest/smallest).
    We want INTERIOR eigenvalues near a target σ. The shift-and-invert
    transformation replaces the problem with:
        C x = θ x,   C = (A − σ B)⁻¹ B,   θ = 1/(λ − σ)
    Eigenvalues λ near σ → large θ → easy to find with Krylov methods.
    Each Krylov step applies C to a vector, which requires solving
    (A − σ B) y = v. MUMPS computes this LU factorisation once, then
    each subsequent Krylov step is a cheap triangular back-substitution.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Compute the spectral shift σ from the target integer metric M.
    # The analytical Helmholtz eigenvalues are λ = −π²·M/4, so σ = −M·π²/4.
    target_sigma = -target_metric * (np.pi**2 / 4)

    if rank == 0:
        print(f"Executing SLEPc MPI solver...")
        print(f"  Target Metric: {target_metric} | Target Shift: {target_sigma:.4f}")
        print(f"  Modes: {num_modes} | NCV: {krylov_size}")

    solver_start = time.time()

    # ------------------------------------------------------------------
    # 1. Configure the Krylov-Schur eigensolver
    # ------------------------------------------------------------------
    # EPS = Eigenvalue Problem Solver — SLEPc's main object.
    # All ranks create the same EPS object; it operates on the distributed matrices.
    eps = SLEPc.EPS().create(comm=comm)

    # Provide the operators. SLEPc knows A and B from this point on.
    eps.setOperators(A_petsc, B_petsc)

    # GNHEP = Generalised Non-Hermitian Eigenvalue Problem.
    # Our problem is real-symmetric, but B is singular (zeros at boundary rows),
    # which disqualifies the simpler GHEP type. GNHEP handles singular B correctly.
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

    # nev: number of eigenvalues to extract.
    # ncv: size of the Krylov subspace built before restarting.
    # More vectors in the subspace = better approximation per restart, but more memory.
    eps.setDimensions(nev=num_modes, ncv=krylov_size)

    # Tell SLEPc which eigenvalues to seek.
    # TARGET_MAGNITUDE: find eigenvalues whose magnitude |λ| is closest to |σ|.
    eps.setTarget(target_sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    # ------------------------------------------------------------------
    # 2. Configure shift-and-invert spectral transformation + MUMPS
    # ------------------------------------------------------------------
    # ST = Spectral Transformation — the object that implements the
    # shift-and-invert change of variables.
    st = eps.getST()

    # SINVERT = shift-and-invert: C = (A − σ B)⁻¹ B.
    st.setType(SLEPc.ST.Type.SINVERT)

    # σ is the spectral shift. The matrix (A − σ B) will be factorised by MUMPS.
    st.setShift(target_sigma)

    # KSP = Krylov Subspace "Solver" — here used as a direct solver wrapper.
    ksp = st.getKSP()

    # 'preonly': do not run any iterative Krylov iterations.
    # The preconditioner (MUMPS LU) IS the solution — one exact application per step.
    ksp.setType('preonly')

    # PC = Preconditioner — here it is the actual solver.
    pc = ksp.getPC()

    # Use LU factorisation as the preconditioner.
    pc.setType('lu')

    # Use MUMPS (Multifrontal Massively Parallel Sparse Solver) to compute
    # the LU factorisation in parallel across all MPI ranks.
    pc.setFactorSolverType('mumps')

    # ------------------------------------------------------------------
    # 3. Factorise (A − σ B) with MUMPS
    # ------------------------------------------------------------------
    # eps.setUp() triggers the MUMPS LU factorisation — the dominant one-time cost.
    # MUMPS distributes the sparse matrix across ranks and computes L and U factors.
    # After setUp(), each Krylov step only needs a cheap triangular solve.
    t0 = time.time()
    eps.setUp()
    if rank == 0:
        print(f"  MUMPS factorization      : {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 4. Run the Krylov-Schur iterations
    # ------------------------------------------------------------------
    # eps.solve() runs the iterative eigensolver:
    #   1. Build Krylov basis vectors v₀, C·v₀, C²·v₀, ..., C^(ncv)·v₀
    #   2. Project the problem onto the Krylov subspace (now ncv×ncv)
    #   3. Solve the small projected problem
    #   4. Check residuals; deflate converged pairs
    #   5. Restart with the un-converged portion of the subspace
    #   6. Repeat until nev pairs have converged
    t0 = time.time()
    eps.solve()
    solver_end = time.time()
    if rank == 0:
        print(f"  Krylov-Schur solve       : {time.time() - t0:.2f}s")
        print(f"  Total solver wall time   : {solver_end - solver_start:.2f}s")

    # ------------------------------------------------------------------
    # 5. Gather distributed eigenvectors back to Rank 0
    # ------------------------------------------------------------------
    # getConverged() returns how many eigenpairs actually converged.
    # This can be MORE than num_modes if the Krylov space was rich.
    nconv = eps.getConverged()
    if rank == 0:
        print(f"  Converged modes          : {nconv} / {num_modes} requested")

    eigenvalues_raw  = []   # will hold complex eigenvalues (rank 0 only)
    eigenvectors_raw = []   # will hold full eigenvectors as numpy arrays (rank 0 only)

    if nconv > 0:
        # createVecs() creates PETSc vectors with the same parallel layout as A.
        # vr = real part of eigenvector (distributed across ranks)
        # vi = imaginary part of eigenvector (near zero for real-symmetric problem)
        vr, vi = A_petsc.createVecs()

        # PETSc.Scatter.toZero() creates a scatter plan that, when executed,
        # collects the full distributed vector `vr` into the sequential vector
        # `v_seq` on rank 0. `v_seq` is empty on all other ranks.
        scatter, v_seq = PETSc.Scatter.toZero(vr)

        for i in range(min(num_modes, nconv)):
            # getEigenpair() fills `vr` and `vi` with the i-th eigenvector
            # (distributed across ranks) and returns the eigenvalue.
            val = eps.getEigenpair(i, vr, vi)

            # Execute the scatter: FORWARD = from distributed vr → sequential v_seq.
            # All ranks must call this (it's collective), but only rank 0 receives data.
            # INSERT mode: simply copy values (no addition).
            scatter.scatter(vr, v_seq, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

            if rank == 0:
                eigenvalues_raw.append(val)                  # complex scalar
                eigenvectors_raw.append(v_seq.getArray().copy())  # full numpy array

    # ------------------------------------------------------------------
    # 6. Post-processing on Rank 0
    # ------------------------------------------------------------------
    if rank == 0:
        # SLEPc returns λ which for the Laplacian is negative (∇² has negative spectrum).
        # We negate to get positive values: lambda_sq = π²(n²+m²+k²)/4.
        lambda_sq = -np.real(eigenvalues_raw)

        # Stack eigenvectors as columns of a 2-D array.
        eigenvectors = np.column_stack(eigenvectors_raw)

        # Sort modes by how close their eigenvalue is to the target.
        # np.argsort returns the indices that would sort the array.
        # We sort by |lambda_sq − target_eigenvalue|, putting the best matches first.
        # Note: -target_sigma = target_metric·π²/4 = the target eigenvalue.
        sort_idx = np.argsort(np.abs(lambda_sq - (-target_sigma)))
        lambda_sq    = lambda_sq[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Convert λ² to integer metric M = n² + m² + k².
        # For a perfect numerical result, this would be an exact integer.
        # Deviation from the nearest integer = discretisation error.
        analytical_integers = lambda_sq / (np.pi**2 / 4)
        return analytical_integers, lambda_sq, eigenvectors
    else:
        # Non-zero ranks return None — the caller (main.py) guards with `if rank == 0`.
        return None, None, None
