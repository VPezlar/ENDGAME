"""
Runtime baseflow loading for the TriGlobal NS operator.
"""

import os
import numpy as np

from src.stab_grid import build_grid, grid_fingerprint
from src.transport import compute_transport


def _to_cube(f, Nx_n, Ny_n, Nz_n):
    return f.reshape(Nz_n, Ny_n, Nx_n)


def _from_cube(c):
    return c.ravel()


def _apply_1d(op, cube, axis):
    moved = np.moveaxis(cube, axis, -1)
    shp = moved.shape
    flat = moved.reshape(-1, shp[-1])
    out = flat @ op.T
    out = out.reshape(shp)
    return np.moveaxis(out, -1, axis)


class _Differentiator:
    def __init__(self, grid, Nx_n, Ny_n, Nz_n):
        self.dx, self.dy, self.dz = grid["dx"], grid["dy"], grid["dz"]
        self.dxx, self.dyy, self.dzz = grid["dxx"], grid["dyy"], grid["dzz"]
        self.dims = (Nx_n, Ny_n, Nz_n)

    def _c(self, f):
        return _to_cube(f, *self.dims)

    def d(self, f, which):
        c = self._c(f)
        if which == "x":   return _from_cube(_apply_1d(self.dx,  c, 2))
        if which == "y":   return _from_cube(_apply_1d(self.dy,  c, 1))
        if which == "z":   return _from_cube(_apply_1d(self.dz,  c, 0))
        if which == "xx":  return _from_cube(_apply_1d(self.dxx, c, 2))
        if which == "yy":  return _from_cube(_apply_1d(self.dyy, c, 1))
        if which == "zz":  return _from_cube(_apply_1d(self.dzz, c, 0))
        raise ValueError(which)

    def dd(self, f, a, b):
        c = self._c(f)
        ax = {"x": 2, "y": 1, "z": 0}
        op = {"x": self.dx, "y": self.dy, "z": self.dz}
        c = _apply_1d(op[a], c, ax[a])
        c = _apply_1d(op[b], c, ax[b])
        return _from_cube(c)


def load_baseflow(cache_path, Nx, Ny, Nz, q, params,
                  xi_half=0.501, eta_half=0.501, zeta_half=0.501,
                  strict=True):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"baseflow cache not found: {cache_path}\n"
            f"Run: python interpolate_baseflow.py <baseflow.csv> {Nx} {Ny} {Nz} --q {q} first."
        )

    d = np.load(cache_path, allow_pickle=False)

    grid = build_grid(Nx, Ny, Nz, q, xi_half, eta_half, zeta_half)
    x1, y1, z1 = grid["x"], grid["y"], grid["z"]
    fp_now = grid_fingerprint(Nx, Ny, Nz, q, xi_half, eta_half, zeta_half, x1, y1, z1)
    fp_cache = str(d["fingerprint"])
    if fp_now != fp_cache:
        msg = (
            "STABILITY GRID MISMATCH — cached baseflow was built for a different grid.\n"
            f"cache fingerprint: {fp_cache}\n"
            f"current fingerprint: {fp_now}\n"
            f"Re-run: python interpolate_baseflow.py <csv> {Nx} {Ny} {Nz} --q {q} "
            f"--xi-half {xi_half} --eta-half {eta_half} --zeta-half {zeta_half}"
        )
        if strict:
            raise ValueError(msg)
        print("WARNING: " + msg)

    prim = {"RHO": d["RHO"], "U": d["U"], "V": d["V"], "W": d["W"], "T": d["T"]}
    for k, v in prim.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"Cached primitive {k} contains non-finite values. Re-run interpolation with nearest fill enabled.")

    Nx_n, Ny_n, Nz_n = Nx + 1, Ny + 1, Nz + 1
    diff = _Differentiator(grid, Nx_n, Ny_n, Nz_n)

    bf = {k: np.asarray(v, dtype=float) for k, v in prim.items()}

    for k in ("RHO", "U", "V", "W", "T"):
        bf[f"{k}_x"] = diff.d(bf[k], "x")
        bf[f"{k}_y"] = diff.d(bf[k], "y")
        bf[f"{k}_z"] = diff.d(bf[k], "z")

    for k in ("U", "V", "W", "T"):
        bf[f"{k}_xx"] = diff.d(bf[k], "xx")
        bf[f"{k}_yy"] = diff.d(bf[k], "yy")
        bf[f"{k}_zz"] = diff.d(bf[k], "zz")

    for comp in ("xx", "yy", "zz"):
        bf[f"RHO_{comp}"] = np.zeros_like(bf["RHO"])

    bf["U_xy"] = diff.dd(bf["U"], "x", "y")
    bf["U_xz"] = diff.dd(bf["U"], "x", "z")
    bf["V_xy"] = diff.dd(bf["V"], "x", "y")
    bf["V_yz"] = diff.dd(bf["V"], "y", "z")
    bf["W_xz"] = diff.dd(bf["W"], "x", "z")
    bf["W_yz"] = diff.dd(bf["W"], "y", "z")

    trans = compute_transport(bf["T"], params)
    bf.update(trans)
    bf["MU_x"] = bf["MU_T"] * bf["T_x"]
    bf["MU_y"] = bf["MU_T"] * bf["T_y"]
    bf["MU_z"] = bf["MU_T"] * bf["T_z"]

    return bf, params
