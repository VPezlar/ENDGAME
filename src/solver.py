import sys
import numpy as np
import time
import os
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

def solve_evp(A_petsc, B_petsc, target_sigma, num_modes, krylov_size):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print(f"Executing SLEPc MPI solver...")
        print(f"  Target Shift: {target_sigma:.4f}")
        print(f"  Modes: {num_modes} | NCV: {krylov_size}")

    solver_start = time.time()

    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A_petsc, B_petsc)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setDimensions(nev=num_modes, ncv=krylov_size)
    eps.setTarget(target_sigma)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target_sigma)
    
    ksp = st.getKSP()
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')

    pfx = ksp.getOptionsPrefix() or ""
    _opts = PETSc.Options()
    _opts[f"{pfx}mat_mumps_icntl_4"]  = 2
    _opts[f"{pfx}mat_mumps_icntl_14"] = 20
    _mem_mb = int(os.environ.get("ENDGAME_MUMPS_MEM_MB", 0))
    if _mem_mb > 0:
        _opts[f"{pfx}mat_mumps_icntl_23"] = _mem_mb
    _opts[f"{pfx}mat_mumps_icntl_35"] = 0

    t0 = time.time()
    eps.setUp()
    t_mumps = time.time() - t0

    mumps_mem_est_mb, mumps_mem_used_mb = -1, -1
    try:
        _F = st.getKSP().getPC().getFactorMatrix()
        mumps_mem_est_mb  = _F.getMumpsInfog(16)
        mumps_mem_used_mb = _F.getMumpsInfog(21)
    except Exception:
        pass

    if rank == 0:
        print(f"  MUMPS factorization      : {t_mumps:.2f}s")

    t0 = time.time()
    eps.solve()
    t_krylov = time.time() - t0

    if rank == 0:
        print(f"  Krylov-Schur solve       : {t_krylov:.2f}s")
        print(f"  Total solver wall time   : {time.time() - solver_start:.2f}s")

    try:
        st.getKSP().getPC().reset()
    except Exception:
        pass

    nconv = eps.getConverged()
    if rank == 0:
        print(f"  Converged modes          : {nconv} / {num_modes} requested")

    eigenvalues_raw  = []
    eigenvectors_raw = []

    if nconv > 0:
        vr, vi = A_petsc.createVecs()
        scatter, v_seq = PETSc.Scatter.toZero(vr)
        for i in range(min(num_modes, nconv)):
            val = eps.getEigenpair(i, vr, vi)
            scatter.scatter(vr, v_seq, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            if rank == 0:
                eigenvalues_raw.append(val)
                eigenvectors_raw.append(v_seq.getArray().copy())
        
        scatter.destroy()
        vr.destroy()
        vi.destroy()
        v_seq.destroy()

    eps.destroy()

    if rank == 0:
        evals = np.array(eigenvalues_raw, dtype=complex)
        eigenvectors = np.column_stack(eigenvectors_raw)
        
        sort_idx = np.argsort(np.abs(evals - target_sigma))
        evals = evals[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        timing = {"t_mumps_s": round(t_mumps, 3), "t_krylov_s": round(t_krylov, 3)}
        return None, evals, eigenvectors, timing
    else:
        return None, None, None, {}
