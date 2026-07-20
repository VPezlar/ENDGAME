import sys
import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


def solve_evp(A_petsc, B_petsc, target_metric, num_modes, krylov_size):
    """
    Solve the generalised eigenvalue problem A x = λ B x using SLEPc.

    A_petsc and B_petsc must already be distributed PETSc MPIAIJ matrices
    (produced by assemble_distributed — no SciPy conversion step needed).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    target_sigma = -target_metric * (np.pi**2 / 4)

    if rank == 0:
        print(f"Executing SLEPc MPI solver...")
        print(f"  Target Metric: {target_metric} | Target Shift: {target_sigma:.4f}")
        print(f"  Modes: {num_modes} | NCV: {krylov_size}")

    solver_start = time.time()

    # 1. Configure Krylov-Schur Solver
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A_petsc, B_petsc)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(nev=num_modes, ncv=krylov_size)
    eps.setTarget(target_sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    # 2. Configure Shift-and-Invert with MUMPS for singular B
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target_sigma)

    ksp = st.getKSP()
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')

    # 3. Factorize (setUp triggers MUMPS LU factorization)
    t0 = time.time()
    eps.setUp()
    if rank == 0:
        print(f"  MUMPS factorization      : {time.time() - t0:.2f}s")

    # 4. Execute Krylov-Schur eigensolver
    t0 = time.time()
    eps.solve()
    solver_end = time.time()
    if rank == 0:
        print(f"  Krylov-Schur solve       : {time.time() - t0:.2f}s")
        print(f"  Total solver wall time   : {solver_end - solver_start:.2f}s")

    # 5. Gather eigenvectors back to Rank 0
    nconv = eps.getConverged()
    if rank == 0:
        print(f"  Converged modes          : {nconv} / {num_modes} requested")

    eigenvalues_raw = []
    eigenvectors_raw = []

    if nconv > 0:
        vr, vi = A_petsc.createVecs()
        # Create a scatter context to funnel data back to the master core
        scatter, v_seq = PETSc.Scatter.toZero(vr)
        
        for i in range(min(num_modes, nconv)):
            val = eps.getEigenpair(i, vr, vi)
            scatter.scatter(vr, v_seq, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            
            if rank == 0:
                eigenvalues_raw.append(val)
                eigenvectors_raw.append(v_seq.getArray().copy())

    # 6. Post-Processing on Rank 0
    if rank == 0:
        lambda_sq = -np.real(eigenvalues_raw)
        eigenvectors = np.column_stack(eigenvectors_raw)
        
        # Sort by distance to target shift
        sort_idx = np.argsort(np.abs(lambda_sq - (-target_sigma)))
        lambda_sq = lambda_sq[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        analytical_integers = lambda_sq / (np.pi**2 / 4)
        return analytical_integers, lambda_sq, eigenvectors
    else:
        return None, None, None