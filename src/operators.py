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
    Nx_n, Ny_n, Nz_n = Nx + 1, Ny + 1, Nz + 1
    N_total = Nx_n * Ny_n * Nz_n

    # ------------------------------------------------------------------
    # 1.  1-D differentiation matrices (small; broadcast inside FDq_Mat)
    # ------------------------------------------------------------------
    D1x, D2x, xi   = FD.FDq_Mat(Nx, q)
    D1y, D2y, eta  = FD.FDq_Mat(Ny, q)
    D1z, D2z, zeta = FD.FDq_Mat(Nz, q)

    # ------------------------------------------------------------------
    # 2.  Coordinate mappings
    # ------------------------------------------------------------------
    x_tmp, Jx, Jxx = BL_Map(xi,   1, xi_half)
    y_tmp, Jy, Jyy = BL_Map(eta,  1, eta_half)
    z_tmp, Jz, Jzz = BL_Map(zeta, 1, zeta_half)

    x, Jx_s = Map1D(-1, 1, x_tmp)
    y, Jy_s = Map1D(-1, 1, y_tmp)
    z, Jz_s = Map1D(-1, 1, z_tmp)

    # Mapped 1-D second-derivative operators (Ni x Ni, CSR)
    dxx = sp.csr_matrix((Jx_s**2) * (np.diag(Jx**2) @ D2x + np.diag(Jxx) @ D1x))
    dyy = sp.csr_matrix((Jy_s**2) * (np.diag(Jy**2) @ D2y + np.diag(Jyy) @ D1y))
    dzz = sp.csr_matrix((Jz_s**2) * (np.diag(Jz**2) @ D2z + np.diag(Jzz) @ D1z))
    dxx.eliminate_zeros()
    dyy.eliminate_zeros()
    dzz.eliminate_zeros()

    # ------------------------------------------------------------------
    # 3.  Create distributed PETSc matrices
    # ------------------------------------------------------------------
    # Conservative nnz estimate per row: up to q nonzeros per direction.
    # Diagonal is shared so unique columns per interior row <= 3q - 2.
    d_nnz = 3 * q          # diagonal-block estimate
    o_nnz = 3 * q          # off-diagonal-block estimate (z-stencil may cross)

    def _make_mat(diag_nnz, offdiag_nnz):
        m = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        m.setSizes((N_total, N_total))
        m.setType(PETSc.Mat.Type.MPIAIJ)
        m.setPreallocationNNZ((diag_nnz, offdiag_nnz))
        m.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        m.setUp()
        return m

    A_petsc = _make_mat(d_nnz, o_nnz)
    B_petsc = _make_mat(1, 0)

    rstart, rend = A_petsc.getOwnershipRange()

    # ------------------------------------------------------------------
    # 4.  Row-local assembly — only owned rows are touched
    # ------------------------------------------------------------------
    for i in range(rstart, rend):
        ix = int(i % Nx_n)
        iy = int((i // Nx_n) % Ny_n)
        iz = int(i // (Nx_n * Ny_n))

        if ix == 0 or ix == Nx or iy == 0 or iy == Ny or iz == 0 or iz == Nz:
            # Dirichlet boundary: identity row in A, zero row in B
            A_petsc.setValue(i, i, 1.0)
        else:
            # Interior: accumulate Laplacian stencil from dxx + dyy + dzz.
            # The shared diagonal (all three directions contribute) is
            # handled naturally by the dict accumulation.
            stencil = {}

            # dxx contribution: row ix at fixed (iy, iz)
            for k in range(int(dxx.indptr[ix]), int(dxx.indptr[ix + 1])):
                j = int(dxx.indices[k]) + iy * Nx_n + iz * Nx_n * Ny_n
                stencil[j] = stencil.get(j, 0.0) + float(dxx.data[k])

            # dyy contribution: row iy at fixed (ix, iz)
            for k in range(int(dyy.indptr[iy]), int(dyy.indptr[iy + 1])):
                j = ix + int(dyy.indices[k]) * Nx_n + iz * Nx_n * Ny_n
                stencil[j] = stencil.get(j, 0.0) + float(dyy.data[k])

            # dzz contribution: row iz at fixed (ix, iy)
            for k in range(int(dzz.indptr[iz]), int(dzz.indptr[iz + 1])):
                j = ix + iy * Nx_n + int(dzz.indices[k]) * Nx_n * Ny_n
                stencil[j] = stencil.get(j, 0.0) + float(dzz.data[k])

            cols = np.array(list(stencil.keys()),   dtype=np.int32)
            vals = np.array(list(stencil.values()), dtype=np.float64)
            A_petsc.setValues(i, cols, vals)
            B_petsc.setValue(i, i, 1.0)

    A_petsc.assemblyBegin(); A_petsc.assemblyEnd()
    B_petsc.assemblyBegin(); B_petsc.assemblyEnd()

    return A_petsc, B_petsc, x, y, z
