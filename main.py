import os
import sys
import time
import json
import numpy as np

import petsc4py
petsc4py.init(sys.argv)
from mpi4py import MPI
from petsc4py import PETSc

from src.operators import assemble_distributed
from src.solver import solve_evp
from src.baseflow import load_baseflow

Nx = int(os.environ.get("ENDGAME_NX", 30))
Ny = int(os.environ.get("ENDGAME_NY", Nx))
Nz = int(os.environ.get("ENDGAME_NZ", Nx))
q  = int(os.environ.get("ENDGAME_Q",  6))
xi_half   = float(os.environ.get("ENDGAME_XI_HALF",   0.501))
eta_half  = float(os.environ.get("ENDGAME_ETA_HALF",  0.501))
zeta_half = float(os.environ.get("ENDGAME_ZETA_HALF", 0.501))
target_sigma = float(os.environ.get("ENDGAME_TARGET", 0.1)) # Real part of NS eigenvalue
num_modes     = int(os.environ.get("ENDGAME_MODES", 10))
krylov_size   = int(os.environ.get("ENDGAME_NCV",   50))
imag_shift = float(os.environ.get("ENDGAME_IMAG_SHIFT", 0.0))
baseflow_cache = os.environ.get("ENDGAME_BASEFLOW", "")

def generate_dummy_baseflow(N_total):
    """
    Placeholder baseflow. In production, load this from your RANS/LES solver 
    and compute the spatial derivatives using the FDq operators.
    """
    bf = {}
    zeros = np.zeros(N_total)
    ones = np.ones(N_total)
    
    # Primitive Variables
    bf['RHO'] = ones
    bf['U'] = ones
    bf['V'] = zeros
    bf['W'] = zeros
    bf['T'] = ones
    
    # Properties
    bf['MU'] = ones * 1.716e-5
    bf['MU_x'], bf['MU_y'], bf['MU_z'] = zeros, zeros, zeros
    bf['MU_T'], bf['MU_TT'] = zeros, zeros
    bf['K'] = ones * 1.716e-5
    bf['K_T'], bf['K_TT'] = zeros, zeros
    
    # Spatial Derivatives (All zero for a uniform flow block)
    for var in ['RHO', 'U', 'V', 'W', 'T']:
        for drv in ['_x', '_y', '_z', '_xx', '_yy', '_zz', '_xy', '_xz', '_yz']:
            bf[var + drv] = zeros
            
    params = {'Re': 1e4, 'M': 0.5, 'gamma': 1.4, 'Pr': 0.72}
    return bf, params

if __name__ == "__main__":
    import os as _os
    _os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
    del _os

    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    size  = comm.Get_size()

    try:
        global_start = time.time()
        N_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
        N_sys   = N_nodes * 5  # TriGlobal DOF
        run_tag = f"NS_N{Nx}_q{q}_P{size}"
        out_dir = os.path.join("output", run_tag)
        
        if rank == 0:
            print("==========================================")
            print(" ENDGAME: TRIGLOBAL NAVIER-STOKES ENGINE")
            print("==========================================")
            print(f"  Grid         : {Nx}x{Ny}x{Nz}  (q={q})")
            print(f"  Total System : {N_sys:,} DOF")
            print(f"  Target sigma : {target_sigma:.4f}")
            print("==========================================")
            print(f"\n[1/3] Assembling block operators...")

        t0 = time.time()
        if baseflow_cache:
            params = {
                'Re': float(os.environ.get("ENDGAME_RE", 1e4)),
                'M': float(os.environ.get("ENDGAME_M", 0.5)),
                'gamma': float(os.environ.get("ENDGAME_GAMMA", 1.4)),
                'Pr': float(os.environ.get("ENDGAME_PR", 0.72)),
                'visc_law': os.environ.get("ENDGAME_VISC", "sutherland"),
                'visc_n': float(os.environ.get("ENDGAME_VISC_N", 0.666)),
                'S_ref': float(os.environ.get("ENDGAME_S_REF", 0.3676)),
                'T_reference': float(os.environ.get("ENDGAME_T_REF", 1.0)),
            }
            if "ENDGAME_S_DIM" in os.environ:
                params['S_dim'] = float(os.environ["ENDGAME_S_DIM"])
            baseflow, params = load_baseflow(
                baseflow_cache, Nx, Ny, Nz, q, params,
                xi_half=xi_half, eta_half=eta_half, zeta_half=zeta_half, strict=True
            )
            if rank == 0:
                print(f"  Baseflow     : cached ({baseflow_cache})")
        else:
            baseflow, params = generate_dummy_baseflow(N_nodes)
            if rank == 0:
                print("  Baseflow     : dummy uniform flow")

        A_petsc, B_petsc, x, y, z = assemble_distributed(
            Nx, Ny, Nz, q, baseflow, params, xi_half, eta_half, zeta_half, imag_shift=imag_shift
        )
        t_assemble = time.time() - t0
        info = A_petsc.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)
        
        if rank == 0:
            nnz_A = int(info["nz_used"])
            print(f"  Done in {t_assemble:.2f}s")
            print(f"  A: {N_sys}x{N_sys},  nnz={nnz_A:,}")
            print(f"\n[2/3] Running SLEPc eigensolver...")

        # We pass target_sigma directly now, not target_metric
        _, evals, evecs, timing = solve_evp(A_petsc, B_petsc, target_sigma, num_modes, krylov_size)
        
        A_petsc.destroy()
        B_petsc.destroy()

        if rank == 0 and evals is not None:
            print(f"\n[3/3] Exporting results to {out_dir}/...")
            os.makedirs(out_dir, exist_ok=True)
            
            # Save Eigenvalues (Real and Imaginary)
            eig_data = np.column_stack((np.real(evals), np.imag(evals)))
            np.savetxt(os.path.join(out_dir, "eigenvalues.txt"), eig_data, fmt="%.8e", header="Real Imaginary")
            
            # The eigenvector array is now 5x larger. It needs reshaping during post-processing.
            np.save(os.path.join(out_dir, "eigenvectors.npy"), evecs)
            
            print(f" TOTAL EXECUTION TIME : {time.time() - global_start:.2f}s")

    except Exception as exc:
        import traceback
        print(f"\n[RANK {rank}] FATAL EXCEPTION   aborting all ranks:\n", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        comm.Abort(1)
