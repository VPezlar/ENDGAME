import numpy as np

def BL_Map(xi, inf, half):
    l = inf * half / (inf - 2 * half)
    s = 2 * l / inf
    x = l * (1 + xi) / (1 + s - xi)
    
    J = l * (2 + s) / ((l + x)**2)
    J2 = -(2 * l * (2 + s)) / ((l + x)**3)
    
    return x, J, J2

def Map1D(min_val, max_val, grid):
    Lx = max_val - min_val
    Lxi = grid[-1] - grid[0]
    
    xscale = Lx / Lxi
    x_1D = xscale * (grid - grid[0]) + min_val
    
    return x_1D, (1.0 / xscale)