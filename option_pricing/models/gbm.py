"""
option_pricing/models/gbm.py
=============================
Exact GBM path simulation.

This module simulates paths for a stock price following Geometric Brownian Motion:

    dS_t = r S_t dt + sigma S_t dW_t

Instead of using an Euler approximation, we use the closed-form GBM solution:

    S_t = S0 * exp((r - sigma^2 / 2) * t + sigma * W_t)

This is important because it means the simulated stock paths are exact at the
chosen monitoring dates. Therefore, any Monte Carlo error comes from simulation
noise, not from discretisation error.

This is why GBM is used as the Stage 2 baseline before moving to Heston dynamics
in Stage 3, where the volatility itself becomes stochastic and exact simulation
is more difficult.

The function is designed to work with both:
    - scalar parameters, where every simulated path uses the same theta
    - array parameters, where each simulated path can have its own theta

For example:

    simulate_gbm(n, 0.02, 100.0, 0.20, rng)

means all paths use:

    r = 0.02
    S0 = 100
    sigma = 0.20

Whereas:

    simulate_gbm(n, r_arr, S0_arr, sigma_arr, rng)

allows every path to have a different interest rate, initial price and volatility.
This is useful when generating training data over many sampled parameter values.
"""

import numpy as np
from option_pricing.config import ND, DT, TGRID
# ND     = number of monitoring dates
# DT     = time step size between monitoring dates
# TGRID  = array of monitoring times

# all stored centrally in config.py so all pricing code uses the same time grid
# avoids accidentally simulating paths on one grid and pricing payoffs on another


def simulate_gbm(n, r, S0, sigma, rng, antithetic=False):
    """
    Simulate n exact GBM paths at the ND monitoring dates.

    Parameters
    ----------
    n : int
        Number of Monte Carlo paths to simulate.

    r : float or (n,) array
        Risk-free interest rate.

        If r is a scalar, every path uses the same interest rate.
        If r is an array of length n, each path uses its own interest rate.

    S0 : float or (n,) array
        Initial asset price.

        As with r, this can either be one value shared by all paths,
        or one value per simulated path.

    sigma : float or (n,) array
        Volatility of the asset.

        This controls how strongly the stock price reacts to Brownian noise.

    rng : numpy.random.Generator
        Random number generator used to generate the Brownian increments.

        Passing rng in from outside the function is better than using
        np.random directly because it makes simulations reproducible.

    antithetic : bool, default False
        If True, use antithetic variates.

        This means we generate a set of random normal draws z, then also use -z.
        The idea is that paths generated from z and -z partially cancel out
        random noise, which can reduce Monte Carlo variance.

        This is a simple variance-reduction method and can be combined with
        PEMC or other control-variate-style methods later.

    Returns
    -------
    S : ndarray, shape (n, ND)
        Simulated asset prices at each monitoring date.

        Each row is one simulated path.
        Each column is one monitoring date.

    W : ndarray, shape (n, ND)
        Cumulative Brownian motion values W_{t_i}.

        These are needed because the exact GBM formula depends on W_t.

    dW : ndarray, shape (n, ND)
        Brownian increments between monitoring dates.

        These are returned so that auxiliary variables can be constructed from
        the same underlying randomness as the payoff. This keeps comparisons
        fair because all methods use the same simulated Brownian paths.

    Notes
    -----
    W and dW are returned because every auxiliary variable phi(Y) in
    aux_variables.py is a deterministic function of them.

    So the workflow is:

        1. Simulate randomness once.
        2. Build the stock path S from that randomness.
        3. Build all auxiliary variables from the same randomness.

    This avoids giving one method an unfair advantage due to different random
    samples.
    """
    # Step 1: Generate standard normal random variables
    # A Brownian increment over a small time step dt is: dW ~ Normal(0, dt)
    # We create this by first drawing: z ~ Normal(0, 1)
    # and then scaling by sqrt(dt): dW = z * sqrt(dt)
    # The array has shape (n, ND): rows are simulated paths, columns are monitoring dates / time steps
    # So z[i, j] is the standard normal shock for path i at time step j
    if antithetic:
        # Instead of generating all n paths independently, we generate roughly half of them and then create matching "opposite" paths by multiplying the shocks by -1
        # This often reduces variance because positive and negative random shocks balance each other out
        half = (n + 1) // 2                               # Use (n + 1) // 2 rather than n // 2 so odd n values are handled
        z = rng.standard_normal((half, ND))               # Generate standard normal shocks for half the paths
        z = np.concatenate([z, -z], axis=0)[:n]           # Stack the original shocks and their negatives, and trim back to exactly n paths
    else:
        # Standard Monte Carlo case
        z = rng.standard_normal((n, ND))
    
    # Step 2: Convert standard normal shocks into Brownian increments.
    dW = z * np.sqrt(DT)
    # Step 3: Build the cumulative Brownian path W_t.
    W  = np.cumsum(dW, axis=1)

    # Step 4: Prepare parameters for vectorised computation.
    # We want the same code to work whether r, S0 and sigma are scalars, shared by all paths, or arrays of shape (n,), with one value per path
    # To do this, we convert each input to a NumPy array and broadcast it to shape (n,). Then we reshape it to (n, 1).
    r     = np.broadcast_to(np.asarray(r,     float), (n,)).reshape(n, 1)
    S0    = np.broadcast_to(np.asarray(S0,    float), (n,)).reshape(n, 1)
    sigma = np.broadcast_to(np.asarray(sigma, float), (n,)).reshape(n, 1)

    # Step 5: Apply the exact GBM solution.
    S = S0 * np.exp((r - 0.5 * sigma**2) * TGRID[None, :] + sigma * W)
    # Step 6: Return the simulated paths and the randomness used to create them.
    return S, W, dW
