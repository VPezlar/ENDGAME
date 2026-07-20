import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from petsc4py import PETSc
from src import FDq as FD
from src.grid import BL_Map, Map1D


def assemble_distributed(Nx, Ny, Nz, q, xi_half=0.501, eta_half=0.501, zeta_half=0.501):
    """
    Assemble the Laplacian (A) and mass (B) matrices as distributed PETSc
    MPIAIJ matrices without ever building a global SciPy matrix.

    Each MPI rank holds only the 1-D operators (small: Ni x Ni) and
    assembles directly into the rows it owns in the PETSc partition.
    Memory footprint during assembly is O(N_total / N_ranks) per rank,
    instead of O(N_total) on every rank.

    Index convention: Fortran column-major order (x varies fastest).
      i = ix + iy * Nx_n + iz * Nx_n * Ny_n
    """
    # Node counts per direction: Nx intervals → Nx+1 nodes (including both endpoints).
    Nx_n, Ny_n, Nz_n = Nx + 1, Ny + 1, Nz + 1
    # Total size of the linear system = total number of grid nodes.
    N_total = Nx_n * Ny_n * Nz_n

    # ------------------------------------------------------------------
    # 1.  1-D differentiation matrices (small; broadcast inside FDq_Mat)
    # ------------------------------------------------------------------
    # FDq_Mat returns:
    #   D1x : (Nx+1)×(Nx+1) first-derivative matrix in reference coordinate ξ
    #   D2x : (Nx+1)×(Nx+1) second-derivative matrix in reference coordinate ξ
    #   xi  : (Nx+1,) optimal node positions on [−1, 1]
    # All ranks receive the same result via broadcast (handled inside FDq_Mat).
    D1x, D2x, xi   = FD.FDq_Mat(Nx, q)
    D1y, D2y, eta  = FD.FDq_Mat(Ny, q)
    D1z, D2z, zeta = FD.FDq_Mat(Nz, q)

    # ------------------------------------------------------------------
    # 2.  Coordinate mappings
    # ------------------------------------------------------------------
    # BL_Map: reference grid ξ → stretched physical grid x, plus Jacobians.
    # The `1` argument is the domain half-size (inf parameter).
    # x_tmp lives in [0, 1] after BL_Map (one boundary-layer side).
    x_tmp, Jx, Jxx = BL_Map(xi,   1, xi_half)
    y_tmp, Jy, Jyy = BL_Map(eta,  1, eta_half)
    z_tmp, Jz, Jzz = BL_Map(zeta, 1, zeta_half)

    # Map1D: rescale x_tmp (arbitrary range) → [-1, 1] (our computational domain).
    # Jx_s is the scalar Jacobian of this linear rescaling (same at every node).
    x, Jx_s = Map1D(-1, 1, x_tmp)
    y, Jy_s = Map1D(-1, 1, y_tmp)
    z, Jz_s = Map1D(-1, 1, z_tmp)

    # Build the fully mapped 1-D second-derivative operators.
    # The chain rule for two successive coordinate changes gives:
    #   d²u/dx² = (Jx_s)² · [Jx² · D2x + Jxx · D1x]
    # where:
    #   D2x is d²/dξ² in reference coords (from Fortran binary)
    #   diag(Jx**2) @ D2x : scales each row i by Jx[i]² (pointwise)
    #   diag(Jxx)   @ D1x : adds the curvature correction term
    #   (Jx_s**2)         : scalar factor from the linear rescaling
    dxx = sp.csr_matrix((Jx_s**2) * (np.diag(Jx**2) @ D2x + np.diag(Jxx) @ D1x))
    dyy = sp.csr_matrix((Jy_s**2) * (np.diag(Jy**2) @ D2y + np.diag(Jyy) @ D1y))
    dzz = sp.csr_matrix((Jz_s**2) * (np.diag(Jz**2) @ D2z + np.diag(Jzz) @ D1z))

    # sp.csr_matrix() built from a dense array stores all entries including
    # near-zero values. eliminate_zeros() removes exact zeros, keeping only
    # the true stencil entries — critical for the row-loop below.
    dxx.eliminate_zeros()
    dyy.eliminate_zeros()
    dzz.eliminate_zeros()

    # ------------------------------------------------------------------
    # 3.  Create distributed PETSc matrices
    # ------------------------------------------------------------------
    # d_nnz: estimate of non-zeros per row in the diagonal block (columns
    #        owned by the same rank as the row). x- and y-stencil entries
    #        typically stay within the same rank's row range.
    # o_nnz: estimate for the off-diagonal block (columns owned by OTHER ranks).
    #        The z-stencil spans Nx_n*Ny_n rows between stencil nodes, so
    #        it can cross MPI partition boundaries.
    # Using 3*q as a conservative upper bound (q entries per direction).
    d_nnz = 3 * q
    o_nnz = 3 * q

    def _make_mat(diag_nnz, offdiag_nnz):
        m = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        # Set the global matrix dimensions. PETSc chooses how to distribute rows.
        m.setSizes((N_total, N_total))
        # MPIAIJ = distributed sparse matrix format (AIJ = CSR in PETSc notation).
        # MPI means rows are split across ranks; AIJ means CSR storage per rank.
        m.setType(PETSc.Mat.Type.MPIAIJ)
        # Tell PETSc how many non-zeros per row to pre-allocate.
        # Good preallocation avoids expensive reallocation during setValue() calls.
        m.setPreallocationNNZ((diag_nnz, offdiag_nnz))
        # Disable the error that fires when we exceed the nnz estimate.
        # With NEW_NONZERO_ALLOCATION_ERR=False, PETSc silently reallocates if needed.
        m.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        m.setUp()
        return m

    A_petsc = _make_mat(d_nnz, o_nnz)  # Laplacian operator + BC rows
    B_petsc = _make_mat(1, 0)           # Mass matrix (diagonal: 1 at interior, 0 at boundary)

    # Ask PETSc which rows THIS rank owns.
    # PETSc distributes rows contiguously: rank 0 gets [0, rend), rank 1 gets [rend, ...)
    # rend is exclusive (Python slice convention).
    rstart, rend = A_petsc.getOwnershipRange()

    # Verify 1-D operator sizes match the declared grid dimensions.
    # An assertion failure here means FDq returned wrong-sized matrices.
    assert dxx.shape == (Nx_n, Nx_n), \
        f'dxx shape {dxx.shape} != ({Nx_n},{Nx_n}) — FDq size mismatch'
    assert dyy.shape == (Ny_n, Ny_n), \
        f'dyy shape {dyy.shape} != ({Ny_n},{Ny_n}) — FDq size mismatch'
    assert dzz.shape == (Nz_n, Nz_n), \
        f'dzz shape {dzz.shape} != ({Nz_n},{Nz_n}) — FDq size mismatch'


    # ------------------------------------------------------------------
    # 4.  Row-local assembly — only owned rows are touched
    # ------------------------------------------------------------------
    for i in range(rstart, rend):     # iterate only over THIS rank's rows

        # Decode the 1-D global index i back into 3-D grid indices (ix, iy, iz).
        # Index convention: x varies fastest (Fortran column-major order):
        #   i = ix + iy * Nx_n + iz * Nx_n * Ny_n
        ix = int(i % Nx_n)               # x-index: remainder after dividing by Nx_n
        iy = int((i // Nx_n) % Ny_n)     # y-index: next digit
        iz = int(i // (Nx_n * Ny_n))     # z-index: most significant digit

        # Check whether this node lies on any face of the cube.
        # Boundary nodes have ix=0, ix=Nx (left/right x faces),
        # iy=0, iy=Ny (bottom/top y faces), iz=0, iz=Nz (front/back z faces).
        if ix == 0 or ix == Nx or iy == 0 or iy == Ny or iz == 0 or iz == Nz:
            # Dirichlet BC: enforce u=0 by making this an identity equation: 1·u_i = 0.
            # A[i, i] = 1 means the equation for node i is simply: u_i = 0 (eigenvalue).
            # B[i, i] = 0 (default, never set) means this equation is excluded
            # from the eigenvalue problem — the boundary eigenvalue is infinite.
            A_petsc.setValue(i, i, 1.0)
        else:
            # Interior node: assemble the Laplacian stencil row by row.
            # We exploit the Kronecker product structure:
            #   Row i of [kron(Iz, kron(Iy, dxx)) + kron(Iz, kron(dyy, Ix)) + kron(dzz, kron(Iy, Ix))]
            # can be computed from rows ix, iy, iz of the small 1-D matrices dxx, dyy, dzz.

            # Use a dict to accumulate column→value pairs.
            # The diagonal entry (j=i) will receive contributions from all three
            # directions and is summed naturally without special-casing.
            stencil = {}

            # --- x-direction contribution (from dxx) ---
            # For the Kronecker structure kron(Iz, kron(Iy, dxx)),
            # row i uses row ix of dxx. The 3-D column index for each
            # 1-D column ix_col is: j = ix_col + iy*Nx_n + iz*Nx_n*Ny_n
            # (only the x-index changes; y and z indices stay fixed).
            #
            # CSR format: indptr[ix] to indptr[ix+1] gives the range of
            # non-zero entries in row ix. indices[k] is the column, data[k] is the value.
            for k in range(int(dxx.indptr[ix]), int(dxx.indptr[ix + 1])):
                j = int(dxx.indices[k]) + iy * Nx_n + iz * Nx_n * Ny_n
                stencil[j] = stencil.get(j, 0.0) + float(dxx.data[k])

            # --- y-direction contribution (from dyy) ---
            # For kron(Iz, kron(dyy, Ix)), row i uses row iy of dyy.
            # 3-D column index: j = ix + iy_col*Nx_n + iz*Nx_n*Ny_n
            # (only the y-index changes).
            for k in range(int(dyy.indptr[iy]), int(dyy.indptr[iy + 1])):
                j = ix + int(dyy.indices[k]) * Nx_n + iz * Nx_n * Ny_n
                stencil[j] = stencil.get(j, 0.0) + float(dyy.data[k])

            # --- z-direction contribution (from dzz) ---
            # For kron(dzz, kron(Iy, Ix)), row i uses row iz of dzz.
            # 3-D column index: j = ix + iy*Nx_n + iz_col*Nx_n*Ny_n
            # (only the z-index changes).
            # NOTE: z-stencil columns are spaced Nx_n*Ny_n apart in global index,
            # so they may land in rows owned by a DIFFERENT MPI rank. PETSc handles
            # this correctly — we just insert the values and assemblyEnd() will
            # communicate them to the owning rank.
            for k in range(int(dzz.indptr[iz]), int(dzz.indptr[iz + 1])):
                j = ix + iy * Nx_n + int(dzz.indices[k]) * Nx_n * Ny_n
                stencil[j] = stencil.get(j, 0.0) + float(dzz.data[k])

            # Convert the accumulated stencil dict to numpy arrays and insert into PETSc.
            # setValues(row, cols, vals) inserts a full row at once.
            cols = np.array(list(stencil.keys()),   dtype=np.int32)
            vals = np.array(list(stencil.values()), dtype=np.float64)
            A_petsc.setValues(i, cols, vals)

            # B is diagonal: 1 at interior nodes, 0 at boundaries.
            # This makes the eigenvalue problem active only at interior nodes.
            B_petsc.setValue(i, i, 1.0)

    # assemblyBegin/End() is a collective operation — all ranks call it together.
    # It flushes any pending insertions, resolves off-rank entries
    # (e.g. z-stencil values that landed in another rank's rows), and
    # compresses the internal storage. Calling only Begin or only End would deadlock.
    A_petsc.assemblyBegin(); A_petsc.assemblyEnd()
    B_petsc.assemblyBegin(); B_petsc.assemblyEnd()

    return A_petsc, B_petsc, x, y, z
