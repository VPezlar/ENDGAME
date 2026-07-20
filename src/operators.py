import numpy as np
import scipy.sparse as sp
from src import FDq as FD
from src.grid import BL_Map, Map1D

def assemble_matrices(Nx, Ny, Nz, q, xi_half=0.501, eta_half=0.501, zeta_half=0.501):
    Nx_nodes, Ny_nodes, Nz_nodes = Nx + 1, Ny + 1, Nz + 1

    # 1. Generate 1D Operators
    dxi, d2xi, xi       = FD.FDq_Mat(Nx, q) 
    deta, d2eta, eta    = FD.FDq_Mat(Ny, q)
    dzeta, d2zeta, zeta = FD.FDq_Mat(Nz, q)

    dxi, d2xi       = sp.csr_matrix(dxi), sp.csr_matrix(d2xi)
    deta, d2eta     = sp.csr_matrix(deta), sp.csr_matrix(d2eta)
    dzeta, d2zeta   = sp.csr_matrix(dzeta), sp.csr_matrix(d2zeta)

    # 2. Apply Boundary Layer Mapping
    x_tmp, Jx, Jxx = BL_Map(xi, 1, xi_half)
    y_tmp, Jy, Jyy = BL_Map(eta, 1, eta_half)
    z_tmp, Jz, Jzz = BL_Map(zeta, 1, zeta_half)

    dx_tmp = sp.diags(Jx) @ dxi
    dy_tmp = sp.diags(Jy) @ deta
    dz_tmp = sp.diags(Jz) @ dzeta

    dxx_tmp = sp.diags(Jx**2) @ d2xi + sp.diags(Jxx) @ dxi
    dyy_tmp = sp.diags(Jy**2) @ d2eta + sp.diags(Jyy) @ deta
    dzz_tmp = sp.diags(Jz**2) @ d2zeta + sp.diags(Jzz) @ dzeta

    # 3. Apply Physical Space Mapping
    x, Jx_scalar = Map1D(-1, 1, x_tmp)
    y, Jy_scalar = Map1D(-1, 1, y_tmp)
    z, Jz_scalar = Map1D(-1, 1, z_tmp)

    dxx = (Jx_scalar**2) * dxx_tmp
    dyy = (Jy_scalar**2) * dyy_tmp
    dzz = (Jz_scalar**2) * dzz_tmp

    # 4. Global 3D Kronecker Assembly
    Ix = sp.eye(Nx_nodes, format="csr")
    Iy = sp.eye(Ny_nodes, format="csr")
    Iz = sp.eye(Nz_nodes, format="csr")

    Dxx = sp.kron(Iz, sp.kron(Iy, dxx, format="csr"), format="csr")
    Dyy = sp.kron(Iz, sp.kron(dyy, Ix, format="csr"), format="csr")
    Dzz = sp.kron(dzz, sp.kron(Iy, Ix, format="csr"), format="csr")

    A_raw = Dxx + Dyy + Dzz

    # 5. Generalized Mass Matrix Masking
    interior_mask_3d = np.zeros((Nx_nodes, Ny_nodes, Nz_nodes), dtype=bool)
    interior_mask_3d[1:-1, 1:-1, 1:-1] = True
    interior_mask_1d = interior_mask_3d.flatten(order='F')

    M_int = sp.diags(interior_mask_1d.astype(float), format="csr")
    M_bnd = sp.diags((~interior_mask_1d).astype(float), format="csr")

    A_global = M_int @ A_raw + M_bnd
    B_global = M_int

    return A_global, B_global, x, y, z