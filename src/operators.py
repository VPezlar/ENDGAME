import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
from petsc4py import PETSc
from src import FDq as FD
from src.grid import BL_Map, Map1D

class TriGlobalCoefficients:
    """
    Pointwise coefficients for the TriGlobal Navier-Stokes operators.
    Pre-computes all physical multipliers to keep the PETSc assembly loop O(N).
    """
    def __init__(self, baseflow, params):
        self.Re = params['Re']
        self.M = params['M']
        self.gamma = params['gamma']
        self.Pr = params['Pr']
        
        self.RHO, self.U, self.V, self.W, self.T = baseflow['RHO'], baseflow['U'], baseflow['V'], baseflow['W'], baseflow['T']
        self.RHO_x, self.RHO_y, self.RHO_z = baseflow['RHO_x'], baseflow['RHO_y'], baseflow['RHO_z']
        self.U_x, self.U_y, self.U_z = baseflow['U_x'], baseflow['U_y'], baseflow['U_z']
        self.U_xx, self.U_yy, self.U_zz = baseflow['U_xx'], baseflow['U_yy'], baseflow['U_zz']
        self.V_x, self.V_y, self.V_z = baseflow['V_x'], baseflow['V_y'], baseflow['V_z']
        self.V_xx, self.V_yy, self.V_zz = baseflow['V_xx'], baseflow['V_yy'], baseflow['V_zz']
        self.W_x, self.W_y, self.W_z = baseflow['W_x'], baseflow['W_y'], baseflow['W_z']
        self.W_xx, self.W_yy, self.W_zz = baseflow['W_xx'], baseflow['W_yy'], baseflow['W_zz']
        self.T_x, self.T_y, self.T_z = baseflow['T_x'], baseflow['T_y'], baseflow['T_z']
        self.T_xx, self.T_yy, self.T_zz = baseflow['T_xx'], baseflow['T_yy'], baseflow['T_zz']
        
        self.U_xy, self.U_xz = baseflow['U_xy'], baseflow['U_xz']
        self.V_xy, self.V_yz = baseflow['V_xy'], baseflow['V_yz']
        self.W_xz, self.W_yz = baseflow['W_xz'], baseflow['W_yz']

        self.MU, self.MU_x, self.MU_y, self.MU_z = baseflow['MU'], baseflow['MU_x'], baseflow['MU_y'], baseflow['MU_z']
        self.MU_T, self.MU_TT = baseflow['MU_T'], baseflow['MU_TT']
        self.K, self.K_T, self.K_TT = baseflow['K'], baseflow['K_T'], baseflow['K_TT']

        self.gM2 = self.gamma * (self.M**2)
        self.RePr = self.Re * self.Pr
        
        self.A = [[{} for _ in range(5)] for _ in range(5)]
        self.B = [[{} for _ in range(5)] for _ in range(5)]
        self._build_operators()

    def _build_operators(self):
        # ---------------------------------------------------------
        # ROW 1: CONTINUITY
        # ---------------------------------------------------------
        self.A[0][0] = {'I': self.U_x + self.V_y + self.W_z, 'Dx': self.U, 'Dy': self.V, 'Dz': self.W}
        self.A[0][1] = {'I': self.RHO_x, 'Dx': self.RHO}
        self.A[0][2] = {'I': self.RHO_y, 'Dy': self.RHO}
        self.A[0][3] = {'I': self.RHO_z, 'Dz': self.RHO}

        # ---------------------------------------------------------
        # ROW 2: X-MOMENTUM
        # ---------------------------------------------------------
        self.A[1][0] = {'I': self.U * self.U_x + self.V * self.U_y + self.W * self.U_z + (1.0 / self.gM2) * self.T_x, 'Dx': self.T / self.gM2}
        self.A[1][1] = {
            'I': self.RHO * self.U_x, 'Dx': self.RHO * self.U - (4.0 * self.MU_x) / (3.0 * self.Re),
            'Dy': self.RHO * self.V - self.MU_y / self.Re, 'Dz': self.RHO * self.W - self.MU_z / self.Re,
            'Dxx': - (4.0 * self.MU) / (3.0 * self.Re), 'Dyy': - self.MU / self.Re, 'Dzz': - self.MU / self.Re
        }
        self.A[1][2] = {'I': self.RHO * self.U_y, 'Dx': - self.MU_y / self.Re, 'Dy': - (2.0 * self.MU_x) / (3.0 * self.Re), 'Dxy': - self.MU / (3.0 * self.Re)}
        self.A[1][3] = {'I': self.RHO * self.U_z, 'Dx': - self.MU_z / self.Re, 'Dz': - (2.0 * self.MU_x) / (3.0 * self.Re), 'Dxz': - self.MU / (3.0 * self.Re)}
        self.A[1][4] = {
            'I': (1.0 / self.gM2) * self.RHO_x - (self.MU_T / self.Re) * ((4.0/3.0) * self.U_xx + self.U_yy + self.U_zz + (1.0/3.0)*(self.V_xy + self.W_xz)) 
                 - (self.MU_TT / self.Re) * ((2.0/3.0) * (2.0 * self.U_x - self.V_y - self.W_z) * self.T_x + (self.U_y + self.V_x) * self.T_y + (self.U_z + self.W_x) * self.T_z),
            'Dx': self.RHO / self.gM2 - (2.0 * self.MU_T / (3.0 * self.Re)) * (2.0 * self.U_x - self.V_y - self.W_z),
            'Dy': - (self.MU_T / self.Re) * (self.U_y + self.V_x), 'Dz': - (self.MU_T / self.Re) * (self.U_z + self.W_x)
        }

        # ---------------------------------------------------------
        # ROW 3: Y-MOMENTUM
        # ---------------------------------------------------------
        self.A[2][0] = {'I': self.U * self.V_x + self.V * self.V_y + self.W * self.V_z + (1.0 / self.gM2) * self.T_y, 'Dy': self.T / self.gM2}
        self.A[2][1] = {'I': self.RHO * self.V_x, 'Dx': - (2.0 * self.MU_y) / (3.0 * self.Re), 'Dy': - self.MU_x / self.Re, 'Dxy': - self.MU / (3.0 * self.Re)}
        self.A[2][2] = {
            'I': self.RHO * self.V_y, 'Dx': self.RHO * self.U - self.MU_x / self.Re,
            'Dy': self.RHO * self.V - (4.0 * self.MU_y) / (3.0 * self.Re), 'Dz': self.RHO * self.W - self.MU_z / self.Re,
            'Dxx': - self.MU / self.Re, 'Dyy': - (4.0 * self.MU) / (3.0 * self.Re), 'Dzz': - self.MU / self.Re
        }
        self.A[2][3] = {'I': self.RHO * self.V_z, 'Dy': - self.MU_z / self.Re, 'Dz': - (2.0 * self.MU_y) / (3.0 * self.Re), 'Dyz': - self.MU / (3.0 * self.Re)}
        self.A[2][4] = {
            'I': (1.0 / self.gM2) * self.RHO_y - (self.MU_T / self.Re) * ((4.0/3.0) * self.V_yy + self.V_xx + self.V_zz + (1.0/3.0)*(self.U_xy + self.W_yz))
                 - (self.MU_TT / self.Re) * ((2.0/3.0) * (2.0 * self.V_y - self.U_x - self.W_z) * self.T_y + (self.U_y + self.V_x) * self.T_x + (self.V_z + self.W_y) * self.T_z),
            'Dx': - (self.MU_T / self.Re) * (self.V_x + self.U_y),
            'Dy': self.RHO / self.gM2 - (2.0 * self.MU_T / (3.0 * self.Re)) * (2.0 * self.V_y - self.U_x - self.W_z),
            'Dz': - (self.MU_T / self.Re) * (self.V_z + self.W_y)
        }

        # ---------------------------------------------------------
        # ROW 4: Z-MOMENTUM
        # ---------------------------------------------------------
        self.A[3][0] = {'I': self.U * self.W_x + self.V * self.W_y + self.W * self.W_z + (1.0 / self.gM2) * self.T_z, 'Dz': self.T / self.gM2}
        self.A[3][1] = {'I': self.RHO * self.W_x, 'Dx': - (2.0 * self.MU_z) / (3.0 * self.Re), 'Dz': - self.MU_x / self.Re, 'Dxz': - self.MU / (3.0 * self.Re)}
        self.A[3][2] = {'I': self.RHO * self.W_y, 'Dy': - (2.0 * self.MU_z) / (3.0 * self.Re), 'Dz': - self.MU_y / self.Re, 'Dyz': - self.MU / (3.0 * self.Re)}
        self.A[3][3] = {
            'I': self.RHO * self.W_z, 'Dx': self.RHO * self.U - self.MU_x / self.Re,
            'Dy': self.RHO * self.V - self.MU_y / self.Re, 'Dz': self.RHO * self.W - (4.0 * self.MU_z) / (3.0 * self.Re),
            'Dxx': - self.MU / self.Re, 'Dyy': - self.MU / self.Re, 'Dzz': - (4.0 * self.MU) / (3.0 * self.Re)
        }
        self.A[3][4] = {
            'I': (1.0 / self.gM2) * self.RHO_z - (self.MU_T / self.Re) * ((4.0/3.0) * self.W_zz + self.W_yy + self.W_xx + (1.0/3.0)*(self.V_yz + self.U_xz))
                 - (self.MU_TT / self.Re) * ((2.0/3.0) * (2.0 * self.W_z - self.V_y - self.U_x) * self.T_z + (self.U_z + self.W_x) * self.T_x + (self.V_z + self.W_y) * self.T_y),
            'Dx': - (self.MU_T / self.Re) * (self.W_x + self.U_z), 'Dy': - (self.MU_T / self.Re) * (self.W_y + self.V_z),
            'Dz': self.RHO / self.gM2 - (2.0 * self.MU_T / (3.0 * self.Re)) * (2.0 * self.W_z - self.V_y - self.U_x)
        }

        # ---------------------------------------------------------
        # ROW 5: ENERGY
        # ---------------------------------------------------------
        gamma_1 = self.gamma - 1.0
        M2_term = 2.0 * self.gamma * gamma_1 * (self.M**2) * self.MU / self.Re
        self.A[4][0] = {'I': self.gamma * (self.T_x * self.U + self.T_y * self.V + self.T_z * self.W), 'Dx': -gamma_1 * self.T * self.U, 'Dy': -gamma_1 * self.T * self.V, 'Dz': -gamma_1 * self.T * self.W}
        self.A[4][1] = {'I': -gamma_1 * (self.T_x * self.RHO + self.T * self.RHO_x) + self.gamma * self.RHO * self.T_x, 'Dx': -M2_term * (2.0/3.0) * (2.0 * self.U_x - self.V_y - self.W_z), 'Dy': -M2_term * (self.U_y + self.V_x), 'Dz': -M2_term * (self.U_z + self.W_x)}
        self.A[4][2] = {'I': -gamma_1 * (self.T_y * self.RHO + self.T * self.RHO_y) + self.gamma * self.RHO * self.T_y, 'Dx': -M2_term * (self.V_x + self.U_y), 'Dy': -M2_term * (2.0/3.0) * (2.0 * self.V_y - self.U_x - self.W_z), 'Dz': -M2_term * (self.V_z + self.W_y)}
        self.A[4][3] = {'I': -gamma_1 * (self.T_z * self.RHO + self.T * self.RHO_z) + self.gamma * self.RHO * self.T_z, 'Dx': -M2_term * (self.W_x + self.U_z), 'Dy': -M2_term * (self.W_y + self.V_z), 'Dz': -M2_term * (2.0/3.0) * (2.0 * self.W_z - self.V_y - self.U_x)}
        dissipation = ((4.0/3.0)*(self.U_x**2 + self.V_y**2 + self.W_z**2 - self.U_x*self.V_y - self.U_x*self.W_z - self.V_y*self.W_z) 
                       + self.U_y**2 + self.V_x**2 + self.U_z**2 + self.W_x**2 + self.V_z**2 + self.W_y**2 + 2.0*self.U_y*self.V_x + 2.0*self.U_z*self.W_x + 2.0*self.V_z*self.W_y)
        self.A[4][4] = {
            'I': -gamma_1 * (self.RHO_x * self.U + self.RHO_y * self.V + self.RHO_z * self.W) - (self.gamma / self.RePr) * (self.K_T * (self.T_xx + self.T_yy + self.T_zz) + self.K_TT * (self.T_x**2 + self.T_y**2 + self.T_z**2)) - (self.gamma * gamma_1 * (self.M**2) * self.MU_T / self.Re) * dissipation,
            'Dx': self.RHO * self.U - (2.0 * self.gamma / self.RePr) * self.K_T * self.T_x,
            'Dy': self.RHO * self.V - (2.0 * self.gamma / self.RePr) * self.K_T * self.T_y,
            'Dz': self.RHO * self.W - (2.0 * self.gamma / self.RePr) * self.K_T * self.T_z,
            'Dxx': - (self.gamma / self.RePr) * self.K, 'Dyy': - (self.gamma / self.RePr) * self.K, 'Dzz': - (self.gamma / self.RePr) * self.K
        }

        # ---------------------------------------------------------
        # B OPERATOR (Mass Matrix)
        # ---------------------------------------------------------
        self.B[0][0] = {'I': np.ones_like(self.RHO)}
        self.B[1][1] = {'I': self.RHO}
        self.B[2][2] = {'I': self.RHO}
        self.B[3][3] = {'I': self.RHO}
        self.B[4][0] = {'I': -gamma_1 * self.T}
        self.B[4][4] = {'I': self.RHO}


def assemble_distributed(Nx, Ny, Nz, q, baseflow, params, xi_half=0.501, eta_half=0.501, zeta_half=0.501, imag_shift=0.0):
    """
    Assemble the TriGlobal Navier-Stokes operators directly into PETSc MPIAIJ format.
    Handles the 5x5 block system using physical dictionaries for pointwise insertion.
    """
    Nx_n, Ny_n, Nz_n = Nx + 1, Ny + 1, Nz + 1
    N_nodes = Nx_n * Ny_n * Nz_n
    N_sys = N_nodes * 5  # 5 equations: rho, u, v, w, T

    D1x, D2x, xi   = FD.FDq_Mat(Nx, q)
    D1y, D2y, eta  = FD.FDq_Mat(Ny, q)
    D1z, D2z, zeta = FD.FDq_Mat(Nz, q)

    x_tmp, Jx, Jxx = BL_Map(xi,   1, xi_half)
    y_tmp, Jy, Jyy = BL_Map(eta,  1, eta_half)
    z_tmp, Jz, Jzz = BL_Map(zeta, 1, zeta_half)

    x, Jx_s = Map1D(-1, 1, x_tmp)
    y, Jy_s = Map1D(-1, 1, y_tmp)
    z, Jz_s = Map1D(-1, 1, z_tmp)

    # 1D Physical First Derivatives
    dx = sp.csr_matrix(Jx_s * np.diag(Jx) @ D1x)
    dy = sp.csr_matrix(Jy_s * np.diag(Jy) @ D1y)
    dz = sp.csr_matrix(Jz_s * np.diag(Jz) @ D1z)

    # 1D Physical Second Derivatives
    dxx = sp.csr_matrix((Jx_s**2) * (np.diag(Jx**2) @ D2x + np.diag(Jxx) @ D1x))
    dyy = sp.csr_matrix((Jy_s**2) * (np.diag(Jy**2) @ D2y + np.diag(Jyy) @ D1y))
    dzz = sp.csr_matrix((Jz_s**2) * (np.diag(Jz**2) @ D2z + np.diag(Jzz) @ D1z))

    for mat in [dx, dy, dz, dxx, dyy, dzz]:
        mat.eliminate_zeros()

    coeff = TriGlobalCoefficients(baseflow, params)

    # Preallocation estimate: 5 equations * approx 3 stencils * (q+1) entries
    d_nnz = 15 * (q + 1) + 1
    o_nnz = 15 * (q + 1) + 1

    def _make_mat(diag_nnz, offdiag_nnz):
        m = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        m.setSizes((N_sys, N_sys))
        m.setType(PETSc.Mat.Type.MPIAIJ)
        m.setPreallocationNNZ((diag_nnz, offdiag_nnz))
        m.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        m.setUp()
        return m

    A_petsc = _make_mat(d_nnz, o_nnz)
    B_petsc = _make_mat(1, 0)

    rstart, rend = A_petsc.getOwnershipRange()

    for I_global in range(rstart, rend):
        eq_idx = I_global // N_nodes
        i = I_global % N_nodes

        ix = int(i % Nx_n)
        iy = int((i // Nx_n) % Ny_n)
        iz = int(i // (Nx_n * Ny_n))

        # DIRICHLET BCs - Enforced strongly at inflow and all other physical boundaries.
        if ix == 0 or ix == Nx or iy == 0 or iy == Ny or iz == 0 or iz == Nz:
            A_petsc.setValue(I_global, I_global, 1.0)
            continue # Leave B_petsc at 0.0 to exclude boundary modes from the spectrum

        for var_idx in range(5):
            block_A = coeff.A[eq_idx][var_idx]
            block_B = coeff.B[eq_idx][var_idx]
            
            # Assembly for B Operator
            if 'I' in block_B:
                J_global = var_idx * N_nodes + i
                B_petsc.setValue(I_global, J_global, block_B['I'][i], PETSc.InsertMode.ADD_VALUES)

            if not block_A: continue

            # Identity
            if 'I' in block_A:
                J_global = var_idx * N_nodes + i
                A_petsc.setValue(I_global, J_global, block_A['I'][i], PETSc.InsertMode.ADD_VALUES)

            # First Derivatives
            if 'Dx' in block_A:
                for k in range(int(dx.indptr[ix]), int(dx.indptr[ix + 1])):
                    J_global = var_idx * N_nodes + (int(dx.indices[k]) + iy * Nx_n + iz * Nx_n * Ny_n)
                    A_petsc.setValue(I_global, J_global, block_A['Dx'][i] * float(dx.data[k]), PETSc.InsertMode.ADD_VALUES)
            if 'Dy' in block_A:
                for k in range(int(dy.indptr[iy]), int(dy.indptr[iy + 1])):
                    J_global = var_idx * N_nodes + (ix + int(dy.indices[k]) * Nx_n + iz * Nx_n * Ny_n)
                    A_petsc.setValue(I_global, J_global, block_A['Dy'][i] * float(dy.data[k]), PETSc.InsertMode.ADD_VALUES)
            if 'Dz' in block_A:
                for k in range(int(dz.indptr[iz]), int(dz.indptr[iz + 1])):
                    J_global = var_idx * N_nodes + (ix + iy * Nx_n + int(dz.indices[k]) * Nx_n * Ny_n)
                    A_petsc.setValue(I_global, J_global, block_A['Dz'][i] * float(dz.data[k]), PETSc.InsertMode.ADD_VALUES)

            # Second Derivatives
            if 'Dxx' in block_A:
                for k in range(int(dxx.indptr[ix]), int(dxx.indptr[ix + 1])):
                    J_global = var_idx * N_nodes + (int(dxx.indices[k]) + iy * Nx_n + iz * Nx_n * Ny_n)
                    A_petsc.setValue(I_global, J_global, block_A['Dxx'][i] * float(dxx.data[k]), PETSc.InsertMode.ADD_VALUES)
            if 'Dyy' in block_A:
                for k in range(int(dyy.indptr[iy]), int(dyy.indptr[iy + 1])):
                    J_global = var_idx * N_nodes + (ix + int(dyy.indices[k]) * Nx_n + iz * Nx_n * Ny_n)
                    A_petsc.setValue(I_global, J_global, block_A['Dyy'][i] * float(dyy.data[k]), PETSc.InsertMode.ADD_VALUES)
            if 'Dzz' in block_A:
                for k in range(int(dzz.indptr[iz]), int(dzz.indptr[iz + 1])):
                    J_global = var_idx * N_nodes + (ix + iy * Nx_n + int(dzz.indices[k]) * Nx_n * Ny_n)
                    A_petsc.setValue(I_global, J_global, block_A['Dzz'][i] * float(dzz.data[k]), PETSc.InsertMode.ADD_VALUES)

            # Cross Derivatives (Nested Kronecker mapping)
            if 'Dxy' in block_A:
                for k_x in range(int(dx.indptr[ix]), int(dx.indptr[ix + 1])):
                    for k_y in range(int(dy.indptr[iy]), int(dy.indptr[iy + 1])):
                        J_global = var_idx * N_nodes + (int(dx.indices[k_x]) + int(dy.indices[k_y]) * Nx_n + iz * Nx_n * Ny_n)
                        val = block_A['Dxy'][i] * float(dx.data[k_x]) * float(dy.data[k_y])
                        A_petsc.setValue(I_global, J_global, val, PETSc.InsertMode.ADD_VALUES)
            if 'Dxz' in block_A:
                for k_x in range(int(dx.indptr[ix]), int(dx.indptr[ix + 1])):
                    for k_z in range(int(dz.indptr[iz]), int(dz.indptr[iz + 1])):
                        J_global = var_idx * N_nodes + (int(dx.indices[k_x]) + iy * Nx_n + int(dz.indices[k_z]) * Nx_n * Ny_n)
                        val = block_A['Dxz'][i] * float(dx.data[k_x]) * float(dz.data[k_z])
                        A_petsc.setValue(I_global, J_global, val, PETSc.InsertMode.ADD_VALUES)
            if 'Dyz' in block_A:
                for k_y in range(int(dy.indptr[iy]), int(dy.indptr[iy + 1])):
                    for k_z in range(int(dz.indptr[iz]), int(dz.indptr[iz + 1])):
                        J_global = var_idx * N_nodes + (ix + int(dy.indices[k_y]) * Nx_n + int(dz.indices[k_z]) * Nx_n * Ny_n)
                        val = block_A['Dyz'][i] * float(dy.data[k_y]) * float(dz.data[k_z])
                        A_petsc.setValue(I_global, J_global, val, PETSc.InsertMode.ADD_VALUES)

    A_petsc.assemblyBegin(); A_petsc.assemblyEnd()
    B_petsc.assemblyBegin(); B_petsc.assemblyEnd()

    # Apply global frequency shift to the spectrum if requested
    if imag_shift != 0.0:
        A_petsc.axpy(1j * imag_shift, B_petsc)
        A_petsc.assemblyBegin(); A_petsc.assemblyEnd()

    return A_petsc, B_petsc, x, y, z
