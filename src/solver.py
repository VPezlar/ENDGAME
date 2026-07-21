import sys
import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


def solve_evp(A_petsc, B_petsc, target_metric, num_modes, krylov_size):
    """
    Solve the generalised eigenvalue problem  A x = lambda B x  using SLEPc.

    Strategy: Krylov-Schur with shift-and-invert spectral transformation.
    (A - sigma*B) is factorised once by MUMPS in parallel across all MPI ranks.
    Each subsequent Krylov step is then a cheap triangular solve.

    Returns a 4-tuple on rank 0:
        (analytical_integers, lambda_sq, eigenvectors, timing_dict)
    Returns (None, None, None, {}) on all other ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Spectral shift: analytical Helmholtz eigenvalues are lambda = -pi^2*M/4,
    # so we shift to sigma = -target_metric*pi^2/4.
    target_sigma = -target_metric * (np.pi**2 / 4)

    if rank == 0:
        print(f"Executing SLEPc MPI solver...")
        print(f"  Target Metric: {target_metric} | Target Shift: {target_sigma:.4f}")
        print(f"  Modes: {num_modes} | NCV: {krylov_size}")

    solver_start = time.time()

    # ------------------------------------------------------------------
    # 1. Configure Krylov-Schur eigensolver
    # ------------------------------------------------------------------
    # EPS = Eigenvalue Problem Solver — SLEPc's main driver object.
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A_petsc, B_petsc)

    # GNHEP = Generalised Non-Hermitian EVP.
    # Used instead of GHEP because B is singular (zero rows at boundary nodes).
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

    # nev = number of eigenpairs requested.
    # ncv = Krylov subspace size; larger -> fewer restarts but more memory per step.
    eps.setDimensions(nev=num_modes, ncv=krylov_size)
    eps.setTarget(target_sigma)
    # TARGET_MAGNITUDE: find eigenvalues whose |lambda| is closest to |sigma|.
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    # ------------------------------------------------------------------
    # 2. Shift-and-invert via MUMPS direct solver
    # ------------------------------------------------------------------
    # ST = Spectral Transformation: C = (A - sigma*B)^{-1} B.
    # Eigenvalues lambda near sigma -> large theta = 1/(lambda-sigma) -> easy to find.
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target_sigma)

    ksp = st.getKSP()
    # 'preonly': no iterative refinement — the LU factorisation IS the solve.
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    # MUMPS: Multifrontal Massively Parallel Sparse Solver — distributed LU.
    pc.setFactorSolverType('mumps')

    # ------------------------------------------------------------------
    # 3. MUMPS factorisation of (A - sigma*B) — one-time expensive step
    # ------------------------------------------------------------------
    # ICNTL(14): extra workspace % above MUMPS's minimum estimate.
    # Default=20 can cause error -13 (workspace allocation failure).
    # 80 means MUMPS pre-allocates 80% extra — robust fix for error -13.
    PETSc.Options()["mat_mumps_icntl_14"] = 80
    t0 = time.time()
    eps.setUp()
    t_mumps = time.time() - t0
    if rank == 0:
        print(f"  MUMPS factorization      : {t_mumps:.2f}s")

    # ------------------------------------------------------------------
    # 4. Krylov-Schur iterations
    # ------------------------------------------------------------------
    t0 = time.time()
    eps.solve()
    t_krylov = time.time() - t0
    solver_end = time.time()
    if rank == 0:
        print(f"  Krylov-Schur solve       : {t_krylov:.2f}s")
        print(f"  Total solver wall time   : {solver_end - solver_start:.2f}s")

    # ------------------------------------------------------------------
    # 5. Gather distributed eigenvectors back to Rank 0
    # ------------------------------------------------------------------
    nconv = eps.getConverged()
    if rank == 0:
        print(f"  Converged modes          : {nconv} / {num_modes} requested")

    eigenvalues_raw  = []
    eigenvectors_raw = []

    if nconv > 0:
        # vr/vi: distributed vectors compatible with A (real/imaginary parts).
        vr, vi = A_petsc.createVecs()
        # Scatter: collects the full distributed vector onto rank 0 sequentially.
        scatter, v_seq = PETSc.Scatter.toZero(vr)

        for i in range(min(num_modes, nconv)):
            # getEigenpair fills vr/vi with the i-th eigenvector (distributed).
            val = eps.getEigenpair(i, vr, vi)
            # FORWARD scatter: distributed vr -> sequential v_seq on rank 0.
            scatter.scatter(vr, v_seq, PETSc.InsertMode.INSERT,
                            PETSc.ScatterMode.FORWARD)
            if rank == 0:
                eigenvalues_raw.append(val)
                eigenvectors_raw.append(v_seq.getArray().copy())

        # Explicit destroy to release PETSc memory immediately.
        # Python GC will eventually collect these, but in a long scaling study
        # it is better to be explicit to prevent accumulation.
        scatter.destroy()
        vr.destroy()
        vi.destroy()
        v_seq.destroy()

    # Destroy the eigensolver and free its internal Krylov vectors.
    eps.destroy()

    # ------------------------------------------------------------------
    # 6. Post-processing on Rank 0
    # ------------------------------------------------------------------
    if rank == 0:
        # Raw eigenvalues from SLEPc are negative (Laplacian spectrum).
        # Negate to get lambda^2 = pi^2*(n^2+m^2+k^2)/4 (positive).
        # In complex PETSc, eigenvalues are Python complex numbers.
        # np.real() extracts real parts — correct for both real and complex builds.
        # For Helmholtz (real symmetric) imaginary parts are ~machine precision.
        lambda_sq   = -np.real(eigenvalues_raw)
        # Print imaginary parts if non-trivial — proves complex arithmetic is active
        _imag = np.imag(np.array(eigenvalues_raw, dtype=complex))
        if np.any(np.abs(_imag) > 1e-10):
            print(f"  Im(eigenvalues)[:3] = {_imag[:3]}  << complex arithmetic confirmed >>", flush=True)
        # Eigenvectors from complex PETSc are complex numpy arrays.
        # np.column_stack works for both real and complex arrays.
        eigenvectors = np.column_stack(eigenvectors_raw)

        # Sort by closeness to the target eigenvalue.
        sort_idx     = np.argsort(np.abs(lambda_sq - (-target_sigma)))
        lambda_sq    = lambda_sq[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Integer metric M = n^2+m^2+k^2 (exact integer for the analytical answer).
        analytical_integers = lambda_sq / (np.pi**2 / 4)

        timing = {"t_mumps_s": round(t_mumps, 3),
                  "t_krylov_s": round(t_krylov, 3),
                  "nconv": nconv}
        return analytical_integers, lambda_sq, eigenvectors, timing
    else:
        return None, None, None, {}
