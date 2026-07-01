"""
option_pricing/models/gbm.py
=============================
Exact GBM path simulation.

Uses the closed-form solution S_t = S0 * exp((r - sigma^2/2)*t + sigma*W_t)
so no Euler-Maruyama discretisation error is introduced.  This is the
primary reason GBM is used as the Stage 2 baseline before introducing
Heston dynamics in Stage 3.

The function accepts scalars OR per-path arrays for (r, S0, sigma) via
NumPy broadcasting, so the same call works for:
  - single-theta evaluation:  simulate_gbm(n, 0.02, 100.0, 0.20, rng)
  - training-data generation: simulate_gbm(n, r_arr, S0_arr, sigma_arr, rng)
"""

import numpy as np
from option_pricing.config import ND, DT, TGRID


def simulate_gbm(n, r, S0, sigma, rng, antithetic=False):
    """
    Simulate n exact GBM paths at the ND monitoring dates.

    Parameters
    ----------
    n          : int         Number of paths.
    r          : float | (n,) array   Risk-free rate.
    S0         : float | (n,) array   Initial asset price.
    sigma      : float | (n,) array   Volatility.
    rng        : numpy.random.Generator
    antithetic : bool        If True, generate n//2 normals and mirror them.
                             Free variance-reduction baseline, stackable with PEMC.

    Returns
    -------
    S  : (n, ND)   Asset values at each monitoring date.
    W  : (n, ND)   Cumulative Brownian path W_{t_i}.
    dW : (n, ND)   Brownian increments.

    Notes
    -----
    W and dW are returned because every auxiliary variable phi(Y) in
    aux_variables.py is a deterministic function of them, so we simulate
    randomness once and reuse it across all constructions.
    """
    if antithetic:
        half = (n + 1) // 2
        z = rng.standard_normal((half, ND))
        z = np.concatenate([z, -z], axis=0)[:n]
    else:
        z = rng.standard_normal((n, ND))

    dW = z * np.sqrt(DT)
    W  = np.cumsum(dW, axis=1)

    # broadcast scalars → (n, 1) for vectorised path computation
    r     = np.broadcast_to(np.asarray(r,     float), (n,)).reshape(n, 1)
    S0    = np.broadcast_to(np.asarray(S0,    float), (n,)).reshape(n, 1)
    sigma = np.broadcast_to(np.asarray(sigma, float), (n,)).reshape(n, 1)

    S = S0 * np.exp((r - 0.5 * sigma**2) * TGRID[None, :] + sigma * W)
    return S, W, dW
