"""
Temperature-dependent transport properties for compressible LNSE.
"""

import numpy as np


def _mu_sutherland_theta(theta, S):
    c = 1.0 + S
    mu = c * theta**1.5 / (theta + S)

    num = theta**1.5
    den = theta + S
    dnum = 1.5 * theta**0.5
    d2num = 0.75 * theta**-0.5

    dmu_dtheta = c * (dnum * den - num) / den**2
    d2mu_dtheta2 = c * ((d2num * den - dnum) * den**2 - (dnum * den - num) * 2.0 * den) / den**4
    return mu, dmu_dtheta, d2mu_dtheta2


def _mu_power_theta(theta, n):
    mu = theta**n
    dmu_dtheta = n * theta**(n - 1.0)
    d2mu_dtheta2 = n * (n - 1.0) * theta**(n - 2.0)
    return mu, dmu_dtheta, d2mu_dtheta2


def compute_transport(T, params):
    T = np.asarray(T, dtype=float)
    law = str(params.get("visc_law", "sutherland")).lower()

    T_ref = float(params.get("T_reference", 1.0))
    if T_ref <= 0.0:
        raise ValueError("T_reference must be positive")
    theta = T / T_ref

    if law == "power":
        n = float(params.get("visc_n", 0.666))
        mu, dmu_dtheta, d2mu_dtheta2 = _mu_power_theta(theta, n)
    elif law == "sutherland":
        if "S_ref" in params:
            S = float(params["S_ref"])
        elif "S_dim" in params:
            S = float(params["S_dim"]) / T_ref
        else:
            S = 0.3676 / T_ref
        mu, dmu_dtheta, d2mu_dtheta2 = _mu_sutherland_theta(theta, S)
    else:
        raise ValueError(f"unknown visc_law '{law}' (use 'sutherland' or 'power')")

    dmu = dmu_dtheta / T_ref
    d2mu = d2mu_dtheta2 / (T_ref**2)

    return {
        "MU": mu, "MU_T": dmu, "MU_TT": d2mu,
        "K": mu.copy(), "K_T": dmu.copy(), "K_TT": d2mu.copy(),
    }
