import numpy as np


def BL_Map(xi, inf, half):
    """
    Algebraic boundary-layer coordinate transformation.

    Maps the reference coordinate ξ ∈ [−1, 1] to a physical coordinate
    x ∈ [0, inf] with node clustering near x = 0 (the boundary layer).

    The mapping formula is:
        x = l · (1 + ξ) / (1 + s − ξ)

    Parameters
    ----------
    xi   : array — reference coordinates in [−1, 1] (output of FDq_Mat)
    inf  : float — physical domain size (the value x approaches as ξ → 1)
    half : float — the reference coordinate ξ at which x = inf/2;
                   smaller half → stronger clustering near the boundary

    Returns
    -------
    x  : array — physical node positions
    J  : array — first Jacobian dξ/dx at each node (needed for chain rule)
    J2 : array — second Jacobian d²ξ/dx² at each node (needed for chain rule)
    """
    # l is an algebraic length scale derived from the domain size and half-domain
    # parameter. Together with s, it controls the shape of the stretching.
    l = inf * half / (inf - 2 * half)
    s = 2 * l / inf

    # Physical node positions via the algebraic map.
    x = l * (1 + xi) / (1 + s - xi)

    # First Jacobian: J = dξ/dx  (NOT dx/dξ — we need the inverse direction
    # for the chain rule when transforming derivative operators).
    # Analytically: J = l·(2+s) / (l+x)²
    J = l * (2 + s) / ((l + x)**2)

    # Second Jacobian: J2 = d²ξ/dx²
    # This appears in the second-order chain rule:
    #   d²u/dx² = J² · d²u/dξ² + J2 · du/dξ
    # Analytically: J2 = −2·l·(2+s) / (l+x)³
    J2 = -(2 * l * (2 + s)) / ((l + x)**3)

    return x, J, J2


def Map1D(min_val, max_val, grid):
    """
    Linearly rescale an arbitrary grid to the interval [min_val, max_val].

    After BL_Map, the physical grid x lives in [0, inf]. This function
    rescales it to the desired computational domain (e.g. [−1, 1]).

    Parameters
    ----------
    min_val : float — lower bound of the target interval
    max_val : float — upper bound of the target interval
    grid    : array — grid points to rescale

    Returns
    -------
    x_1D   : array — rescaled grid in [min_val, max_val]
    1/xscale: float — the inverse scaling factor dξ/dx for this linear map;
                      used as an additional scalar Jacobian in operators.py
    """
    # Physical and reference domain lengths.
    Lx  = max_val - min_val        # length of the target interval
    Lxi = grid[-1] - grid[0]       # length of the input grid span

    # Scale factor: how much we stretch or compress the grid.
    xscale = Lx / Lxi

    # Apply the linear map: shift grid so it starts at 0, scale, then shift to min_val.
    x_1D = xscale * (grid - grid[0]) + min_val

    # Return the inverse scale factor (1/xscale = dξ/dx for this linear map).
    # This is the Jacobian of the inverse map, needed in the chain rule
    # when applying the coordinate-mapped second-derivative operator.
    return x_1D, (1.0 / xscale)
