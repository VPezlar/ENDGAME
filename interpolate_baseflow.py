#!/usr/bin/env python3
"""
Interpolate FVM baseflow primitives onto the ENDGAME stability grid and cache them.

Key behavior:
- Header-driven column lookup (order-independent CSV)
- Linear interpolation first
- NaN/out-of-hull points are filled with nearest-neighbor interpolation
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from src.stab_grid import build_grid, grid_fingerprint


def _normalize_header(h):
    return h.strip().lower().replace(' ', '').replace('_', '').replace('-', '')


def _find_col(headers, aliases):
    norm = {_normalize_header(h): i for i, h in enumerate(headers)}
    for a in aliases:
        k = _normalize_header(a)
        if k in norm:
            return norm[k]
    raise KeyError(f"Missing required column; tried aliases: {aliases}")


def _read_csv_columns(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    headers = [h.strip() for h in header_line.split(',')]
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data[None, :]

    aliases = {
        'x': ['x', 'points:0', 'point:0', 'coordx'],
        'y': ['y', 'points:1', 'point:1', 'coordy'],
        'z': ['z', 'points:2', 'point:2', 'coordz'],
        'RHO': ['rho', 'density'],
        'U': ['u', 'velocity:0', 'velx'],
        'V': ['v', 'velocity:1', 'vely'],
        'W': ['w', 'velocity:2', 'velz'],
        'T': ['t', 'temperature', 'temp'],
    }

    cols = {k: data[:, _find_col(headers, v)] for k, v in aliases.items()}

    valid = np.ones(len(data), dtype=bool)
    for k in cols:
        valid &= np.isfinite(cols[k])
    for k in cols:
        cols[k] = cols[k][valid]
    return cols


def _query_points(x, y, z):
    Nx_n, Ny_n, Nz_n = len(x), len(y), len(z)
    X = np.tile(x, Ny_n * Nz_n)
    Y = np.tile(np.repeat(y, Nx_n), Nz_n)
    Z = np.repeat(z, Nx_n * Ny_n)
    return np.column_stack((X, Y, Z))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv')
    p.add_argument('Nx', type=int)
    p.add_argument('Ny', type=int, nargs='?', default=None)
    p.add_argument('Nz', type=int, nargs='?', default=None)
    p.add_argument('--q', type=int, default=6)
    p.add_argument('--xi-half', type=float, default=0.501)
    p.add_argument('--eta-half', type=float, default=0.501)
    p.add_argument('--zeta-half', type=float, default=0.501)
    p.add_argument('--out', type=str, default='')
    args = p.parse_args()

    Ny = args.Ny if args.Ny is not None else args.Nx
    Nz = args.Nz if args.Nz is not None else args.Nx

    cols = _read_csv_columns(args.csv)
    points = np.column_stack((cols['x'], cols['y'], cols['z']))

    grid = build_grid(args.Nx, Ny, Nz, args.q, args.xi_half, args.eta_half, args.zeta_half)
    x, y, z = grid['x'], grid['y'], grid['z']
    qpts = _query_points(x, y, z)

    tri = Delaunay(points)

    out_fields = {}
    total_filled = 0
    for key in ('RHO', 'U', 'V', 'W', 'T'):
        lin = LinearNDInterpolator(tri, cols[key], fill_value=np.nan)
        near = NearestNDInterpolator(points, cols[key])
        vals = np.asarray(lin(qpts), dtype=float)
        mask = ~np.isfinite(vals)
        if np.any(mask):
            vals[mask] = near(qpts[mask])
            total_filled += int(mask.sum())
        if not np.all(np.isfinite(vals)):
            raise RuntimeError(f"Interpolation for {key} still contains non-finite values after nearest fill.")
        out_fields[key] = vals

    fp = grid_fingerprint(args.Nx, Ny, Nz, args.q, args.xi_half, args.eta_half, args.zeta_half, x, y, z)

    if args.out:
        out_path = Path(args.out)
    else:
        tag = f"N{args.Nx}_Ny{Ny}_Nz{Nz}_q{args.q}" if not (args.Nx == Ny == Nz) else f"N{args.Nx}_q{args.q}"
        out_path = Path('baseflow_cache') / f'{tag}.npz'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        RHO=out_fields['RHO'], U=out_fields['U'], V=out_fields['V'], W=out_fields['W'], T=out_fields['T'],
        x=x, y=y, z=z,
        fingerprint=np.array(fp),
        Nx=np.array(args.Nx), Ny=np.array(Ny), Nz=np.array(Nz), q=np.array(args.q),
        xi_half=np.array(args.xi_half), eta_half=np.array(args.eta_half), zeta_half=np.array(args.zeta_half),
    )

    print(f"Wrote cache: {out_path}")
    print(f"Grid nodes: {(args.Nx+1)*(Ny+1)*(Nz+1):,}")
    print(f"Nearest-filled points across primitives: {total_filled}")


if __name__ == '__main__':
    main()
