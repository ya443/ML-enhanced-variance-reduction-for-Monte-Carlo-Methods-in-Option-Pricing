"""
option_pricing/pemc/aux_variables.py
=====================================
Auxiliary variable constructions X = phi(Y).

Each function takes simulated Brownian information and maps it into a lower
dimensional feature array X. These X variables are used by PEMC to learn or
condition on useful summaries of the randomness driving the payoff.

The functions here do not simulate anything themselves. They only transform
already-simulated paths from simulate_gbm or simulate_heston. This keeps the
auxiliary variables separate from the model simulation code.

The registry dictionaries store:
    name -> (function, has_exact_gaussian_marginal)

The Boolean flag tells marginals.py how to cheaply sample X-tilde:
    True  = X has an exact Gaussian marginal, so sample it directly.
    False = X is more complicated, so use Brownian-path simulation instead.
"""

import numpy as np
from option_pricing.config import DT, T, ND

# Split the ND Brownian increments into 14 roughly equal time blocks.
# Each block will later be summed to form a coarse summary of the path.
_PART_IDX = np.array_split(np.arange(ND), 14)


def _terminal(W, dW):
    """
    Terminal Brownian value W_T.

    This is the simplest auxiliary variable. It only keeps the final value of
    the Brownian path and discards the shape of the path in between.

    Shape:
        input  W: (n, ND)
        output  : (n, 1)

    Marginal:
        W_T ~ N(0, T), so this can be sampled exactly.
    """
    return W[:, -1:]


def _partition14(W, dW):
    """
    Fourteen block sums of Brownian increments.

    Instead of keeping only W_T, this keeps a rough time-profile of the
    Brownian path by splitting the increments into 14 groups and summing each
    group separately.

    Each block sum is Gaussian because it is a sum of Gaussian increments.
    Since the blocks use disjoint increments, the block sums are independent.

    Shape:
        output: (n, 14)
    """
    return np.stack([dW[:, idx].sum(axis=1) for idx in _PART_IDX], axis=1)


def _time_average(W, dW):
    """
    Discretised time-average of the Brownian path.

        bar_W = (1 / T) * integral_0^T W_t dt

    In discrete form, using the monitoring grid:

        bar_W approx (1 / T) * sum_i W_{t_i} * DT

    This is payoff-aware because an Asian option depends on an average over
    time, not just the terminal price. Since GBM depends on W_t, the time
    average of W_t is naturally related to the arithmetic average of S_t.

    Marginal:
        bar_W is Gaussian because it is a linear combination of Gaussian W_t.
    """
    return (W * DT).sum(axis=1, keepdims=True) / T


def _time_avg_plus_WT(W, dW):
    """
    Pair of Brownian summaries: (bar_W, W_T).

    This keeps both:
        1. the time-average of the Brownian path
        2. the terminal Brownian value

    This can be useful because Asian payoffs care about the whole path, while
    W_T captures the final cumulative shock.

    Marginal:
        This is bivariate Gaussian because both components are linear
        combinations of the Brownian path.
    """
    ta = (W * DT).sum(axis=1, keepdims=True) / T
    return np.concatenate([ta, W[:, -1:]], axis=1)


def _extrema(W, dW):
    """
    Terminal value, running maximum and running minimum of Brownian motion.

        X = (W_T, max W_t, min W_t)

    These features capture the range and extremes of the path. This can be
    useful because option payoffs often react strongly to path shape.

    This is marked as a hard marginal because max(W_t) and min(W_t) are not
    simple Gaussian variables.
    """
    return np.stack([W[:, -1], W.max(axis=1), W.min(axis=1)], axis=1)


def _moments(W, dW):
    """
    Terminal value plus quadratic and cubic variation-style summaries.

        X = (W_T, sum dW^2, sum dW^3)

    sum dW^2 captures realised path variation. For Brownian motion this is
    connected to quadratic variation.

    sum dW^3 captures asymmetry in the realised increments. It is not Gaussian,
    so this construction needs path simulation for its marginal.
    """
    return np.stack([
        W[:, -1],
        (dW**2).sum(axis=1),
        (dW**3).sum(axis=1)
    ], axis=1)


def _combined(W, dW):
    """
    Concatenate all GBM auxiliary variables into one feature vector.

    This gives a richer summary of the Brownian path by combining terminal,
    block-sum, time-average, extrema and moment information.

    It is useful as an empirical upper benchmark because it gives PEMC more
    information than any single construction.

    Dimension:
        terminal      1
        partition14   14
        time_average  1
        extrema       3
        moments       3
        total         22
    """
    return np.concatenate([
        _terminal(W, dW),
        _partition14(W, dW),
        _time_average(W, dW),
        _extrema(W, dW),
        _moments(W, dW),
    ], axis=1)


# Registry for GBM auxiliary variables.
# True means the marginal distribution of X can be sampled exactly as Gaussian.
# False means the marginal is harder, so marginals.py should simulate paths.
AUX_GBM = {
    "terminal"         : (_terminal,         True),
    "partition14"      : (_partition14,      True),
    "time_average"     : (_time_average,     True),
    "time_avg_plus_WT" : (_time_avg_plus_WT, True),
    "extrema"          : (_extrema,          False),
    "moments"          : (_moments,          False),
    "combined"         : (_combined,         False),
}


def _terminal_heston(W_S, W_v, dW_S, dW_v, v_path):
    """
    Terminal values of the two Heston Brownian drivers.

        X = (W_T^S, W_T^v)

    W_S drives the asset-price noise and W_v drives the variance-process noise.

    This is the Heston version of the terminal Brownian baseline. The pair has
    a bivariate Gaussian marginal, so it can be sampled directly.
    """
    return np.stack([W_S[:, -1], W_v[:, -1]], axis=1)


def _levy_area(W_S, W_v, dW_S, dW_v, v_path):
    """
    Levy-area-style feature for the two Brownian drivers.

        L = sum_j (W_S_j * dW_v_j - W_v_j * dW_S_j)

    This captures interaction between the two Brownian paths over time. Terminal
    values only tell us where each path ended, while the Levy area also carries
    information about how the two paths moved around each other.

    The returned feature is:

        X = (W_T^S, W_T^v, L)

    This has a hard marginal because L is nonlinear in the Brownian path.
    """
    L = (W_S[:, :-1] * dW_v[:, 1:] - W_v[:, :-1] * dW_S[:, 1:]).sum(axis=1)
    return np.stack([W_S[:, -1], W_v[:, -1], L], axis=1)


def _integrated_var(W_S, W_v, dW_S, dW_v, v_path):
    """
    Terminal Brownian values plus an integrated variance proxy.

        X = (W_T^S, W_T^v, average variance)

    The average variance is approximated by:

        (1 / T) * integral_0^T v_t dt
        approx (DT / T) * sum_j v_{t_j}

    This is useful under Heston because the asset path depends heavily on the
    realised variance path, not just the Brownian shocks.

    This is marked as a hard marginal because it depends on v_path.
    """
    int_var = (v_path * DT).sum(axis=1) / T
    return np.stack([W_S[:, -1], W_v[:, -1], int_var], axis=1)


def _combined_heston(W_S, W_v, dW_S, dW_v, v_path):
    """
    Concatenate all Heston auxiliary variables.

    This combines:
        terminal_heston  -> 2 features
        levy_area        -> 3 features
        integrated_var   -> 3 features

    Total dimension:
        8 features

    This gives a richer Heston path summary than using terminal Brownian values
    alone.
    """
    return np.concatenate([
        _terminal_heston(W_S, W_v, dW_S, dW_v, v_path),
        _levy_area(W_S, W_v, dW_S, dW_v, v_path),
        _integrated_var(W_S, W_v, dW_S, dW_v, v_path),
    ], axis=1)


# Registry for Heston auxiliary variables.
# The exact flag is only True for terminal_heston because it is Gaussian.
# Levy area and integrated variance need path-based marginal simulation.
AUX_HESTON = {
    "terminal_heston"  : (_terminal_heston, True),
    "levy_area"        : (_levy_area,       False),
    "integrated_var"   : (_integrated_var,  False),
    "combined_heston"  : (_combined_heston, False),
}