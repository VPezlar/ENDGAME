import sys
import numpy as np
import time
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

def scipy_to_petsc(scipy_mat):
    """Converts a global SciPy CSR matrix into a distributed PETSc MPI matrix."""
    petsc_mat = PETSc.Mat().create(comm=MPI.COMM_WORLD)
    petsc_mat.setSizes(scipy_mat.shape)
    petsc_mat.setType(PETSc.Mat.Type.MPIAIJ)
    petsc_mat.setUp()
    
    # Each MPI rank extracts and assigns only its owned physical rows
    rstart, rend = petsc_mat.getOwnershipRange()
    csr_mat = scipy_mat.tocsr()
    
    for i in range(rstart, rend):
        cols = csr_mat.indices[csr_mat.indptr[i]:csr_mat.indptr[i+1]]
        vals = csr_mat.data[csr_mat.indptr[i]:csr_mat.indptr[i+1]]
        petsc_mat.setValues(i, cols, vals)
        
    petsc_mat.assemblyBegin()
    petsc_mat.assemblyEnd()
    return petsc_mat

def solve_evp(A_raw, B_raw, target_metric, num_modes, krylov_size):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    target_sigma = -target_metric * (np.pi**2 / 4)

    if rank == 0:
        print(f"Executing SLEPc MPI solver...")
        print(f"  Target Metric: {target_metric} | Target Shift: {target_sigma:.4f}")
        print(f"  Modes: {num_modes} | NCV: {krylov_size}")
    
    solver_start = time.time()

    # 1. Distribute Matrices across Cores
    t0 = time.time()
    A_petsc = scipy_to_petsc(A_raw)
    B_petsc = scipy_to_petsc(B_raw)
    if rank == 0:
        print(f"  SciPy → PETSc conversion : {time.time() - t0:.2f}s")

    # 2. Configure KrylovSchur Solver
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A_petsc, B_petsc)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(nev=num_modes, ncv=krylov_size)
    eps.setTarget(target_sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    # 3. Configure Shift-and-Invert with MUMPS for singular B
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target_sigma)

    ksp = st.getKSP()
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')

    # 4. Factorize (setUp triggers MUMPS LU factorization)
    t0 = time.time()
    eps.setUp()
    if rank == 0:
        print(f"  MUMPS factorization      : {time.time() - t0:.2f}s")

    # 5. Execute Krylov-Schur eigensolver
    t0 = time.time()
    eps.solve()
    solver_end = time.time()
    if rank == 0:
        print(f"  Krylov-Schur solve       : {time.time() - t0:.2f}s")
        print(f"  Total solver wall time   : {solver_end - solver_start:.2f}s")

    # 6. Gather Distributed Vectors Back to Rank 0
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

    # 7. Post-Processing on Rank 0
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