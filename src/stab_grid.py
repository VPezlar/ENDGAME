"""
Shared construction of the stability grid and its 1-D physical derivative
operators.
"""

import hashlib
import numpy as np
import scipy.sparse as sp

from src import FDq as FD
from src.grid import BL_Map, Map1D


def build_axis(N, q, half):
    D1, D2, ref = FD.FDq_Mat(N, q)
    x_tmp, J, Jz = BL_Map(ref, 1, half)
    coord, Js = Map1D(-1, 1, x_tmp)

    d1 = sp.csr_matrix(Js * np.diag(J) @ D1)
    d2 = sp.csr_matrix((Js**2) * (np.diag(J**2) @ D2 + np.diag(Jz) @ D1))
    d1.eliminate_zeros()
    d2.eliminate_zeros()
    return coord, d1, d2


def build_grid(Nx, Ny, Nz, q, xi_half, eta_half, zeta_half):
    x, dx, dxx = build_axis(Nx, q, xi_half)
    y, dy, dyy = build_axis(Ny, q, eta_half)
    z, dz, dzz = build_axis(Nz, q, zeta_half)
    return {
        "x": x, "y": y, "z": z,
        "dx": dx, "dy": dy, "dz": dz,
        "dxx": dxx, "dyy": dyy, "dzz": dzz,
    }


def grid_fingerprint(Nx, Ny, Nz, q, xi_half, eta_half, zeta_half, x, y, z):
    h = hashlib.sha1()
    for v in (Nx, Ny, Nz, q):
        h.update(np.int64(v).tobytes())
    for v in (xi_half, eta_half, zeta_half):
        h.update(np.float64(round(float(v), 12)).tobytes())
    for arr in (x, y, z):
        h.update(np.round(np.asarray(arr, dtype=np.float64), 12).tobytes())
    return h.hexdigest()
