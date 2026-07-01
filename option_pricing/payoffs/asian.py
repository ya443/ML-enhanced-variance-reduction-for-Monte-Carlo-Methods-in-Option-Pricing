"""
option_pricing/payoffs/asian.py
================================
Payoff functions for Asian options.

Three functions:
  arithmetic_asian_payoff  — the target f(Y); no closed form
  geometric_asian_payoff   — textbook control variate payoff
  geometric_asian_price    — its exact Black-Scholes-type price under GBM

Derivation of the closed form
------------------------------
log(geometric average) = (1/n_D) * sum log(S_{t_i})
is a linear combination of jointly Gaussian rvs, so:
    log(G_bar) ~ N(mu_G, var_G)
where
    mu_G  = log(S0) + (r - sigma^2/2) * mean(TGRID)
    var_G = (sigma^2 / n_D^2) * sum_{i,j} min(t_i, t_j)
using cov(W_{t_i}, W_{t_j}) = min(t_i, t_j).
"""

import numpy as np
from scipy.stats import norm
from option_pricing.config import T, ND, TGRID


def arithmetic_asian_payoff(S, K, r):
    """
    Discounted arithmetic-average Asian call payoff.
    This is f(Y) — the expensive quantity PEMC estimates.

    Parameters
    ----------
    S : (n, ND) asset paths at monitoring dates
    K : float   strike price
    r : float   risk-free rate

    Returns
    -------
    (n,) discounted payoffs: e^{-rT} * (S_bar - K)^+
    """
    return np.exp(-r * T) * np.maximum(S.mean(axis=1) - K, 0.0)


def geometric_asian_payoff(S, K, r):
    """
    Discounted geometric-average Asian call payoff.
    Used as the classical control variate — its expectation is known in
    closed form via geometric_asian_price().

    Returns
    -------
    (n,) discounted geometric payoffs: e^{-rT} * (G_bar - K)^+
    """
    gavg = np.exp(np.mean(np.log(S), axis=1))
    return np.exp(-r * T) * np.maximum(gavg - K, 0.0)


def geometric_asian_price(r, S0, sigma, K):
    """
    Closed-form price of a discretely-monitored geometric-average Asian call
    under GBM.  Used as the analytical benchmark (ground-truth CV).

    Parameters
    ----------
    r, S0, sigma, K : floats — evaluation point parameters

    Returns
    -------
    float — exact geometric Asian call price
    """
    mu_G  = np.log(S0) + (r - 0.5 * sigma**2) * TGRID.mean()
    tt    = np.minimum.outer(TGRID, TGRID)
    var_G = sigma**2 / ND**2 * tt.sum()
    sig_G = np.sqrt(var_G)
    d1    = (mu_G - np.log(K) + var_G) / sig_G
    d2    = d1 - sig_G
    return np.exp(-r * T) * (
        np.exp(mu_G + 0.5 * var_G) * norm.cdf(d1) - K * norm.cdf(d2)
    )
