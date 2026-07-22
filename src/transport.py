"""
Temperature-dependent transport properties for compressible LNSE.

Convention used here (matching operator/baseflow chain-rule usage):
- Input T is NON-DIMENSIONAL.
- MU_T and MU_TT are derivatives with respect to NON-DIMENSIONAL T.
- T_reference is used only to define the Sutherland constant scale when
  dimensional constants are provided (CS/T_reference), not to rescale T.
"""

import numpy as np


def _mu_sutherland(T, S):
    """Non-dimensional Sutherland law and first two T-derivatives."""
    c = 1.0 + S
    den = T + S

    mu = c * T**1.5 / den

    # d/dT [ c*T^(3/2)/(T+S) ]
    dmu = c * ((1.5 * T**0.5) * den - T**1.5) / (den**2)

    # d2/dT2 [ c*T^(3/2)/(T+S) ]
    # Let num=T^(3/2), den=(T+S):
    # f'' = (num''*den^2 - 2*num'*den + 2*num)/den^3
    num = T**1.5
    dnum = 1.5 * T**0.5
    d2num = 0.75 * T**-0.5
    d2mu = c * (d2num * den**2 - 2.0 * dnum * den + 2.0 * num) / (den**3)

    return mu, dmu, d2mu


def _mu_power(T, n):
    """Power law mu=T^n and first two T-derivatives."""
    mu = T**n
    dmu = n * T**(n - 1.0)
    d2mu = n * (n - 1.0) * T**(n - 2.0)
    return mu, dmu, d2mu


def compute_transport(T, params):
    """
    Return MU, MU_T, MU_TT, K, K_T, K_TT where derivatives are w.r.t. non-dim T.

    Parameter precedence:
      1) If S_ref provided: use directly as non-dimensional Sutherland constant S.
      2) Else if S_dim provided: S = S_dim / T_reference.
      3) Else use gas default with T_reference:
           air -> S_dim=110.4, n2 -> S_dim=107.0

    Other keys:
      visc_law: 'sutherland' (default) or 'power'
      visc_n: exponent for power law (default 0.666)
      T_reference: dimensional reference temperature for S scaling (default 288.0)
      gas: 'air' (default) or 'n2'
    """
    T = np.asarray(T, dtype=float)
    law = str(params.get("visc_law", "sutherland")).lower()

    if law == "power":
        n = float(params.get("visc_n", 0.666))
        mu, dmu, d2mu = _mu_power(T, n)

    elif law == "sutherland":
        if "S_ref" in params:
            S = float(params["S_ref"])
        else:
            T_ref = float(params.get("T_reference", 288.0))
            if T_ref <= 0.0:
                raise ValueError("T_reference must be positive")
            if "S_dim" in params:
                S_dim = float(params["S_dim"])
            else:
                gas = str(params.get("gas", "air")).lower()
                if gas in ("air", "a"):
                    S_dim = 110.4
                elif gas in ("n2", "nitrogen"):
                    S_dim = 107.0
                else:
                    raise ValueError(f"unknown gas '{gas}' (use 'air' or 'n2')")
            S = S_dim / T_ref

        mu, dmu, d2mu = _mu_sutherland(T, S)

    else:
        raise ValueError(f"unknown visc_law '{law}' (use 'sutherland' or 'power')")

    return {
        "MU": mu,
        "MU_T": dmu,
        "MU_TT": d2mu,
        "K": mu.copy(),
        "K_T": dmu.copy(),
        "K_TT": d2mu.copy(),
    }
