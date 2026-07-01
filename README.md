# ML enhanced variance reduction for Monte Carlo Methods in Option Pricing

This repo contains the code for my Master's thesis, where I investigate auxiliary variable design, parameter sensitivity, and predictor complexity as levers for improving variance reduction, network convergence, and computational efficiency in Prediction-Enhanced Monte Carlo (PEMC) pricing of path-dependent or high-dimensional derivatives.

MSc Financial Mathematics with Data Science, University of Bath (2026).

## Overview

[Prediction-Enhanced Monte Carlo](https://arxiv.org/pdf/2412.11257) (Li et al., 2024) reduces the
variance of Monte Carlo option pricing estimators by training a neural network predictor
`g(θ, X)` on a low-dimensional auxiliary variable `X = φ(Y)` summarising each simulated path `Y`,
and using it as a control variate whose mean is estimated by cheap parallel simulation rather
than computed in closed form. The variance reduction achieved is governed by the correlation
`ρ = corr(f(Y), g(θ, X))` between the payoff and the predictor, and by the cost ratio
`c = c_g / c_f` between evaluating the predictor and evaluating the true payoff.

The original PEMC implementation makes a fixed, generic choice for the auxiliary variable
(terminal sums of Brownian increments), a fixed predictor architecture, and a fixed sample
allocation across every derivative class. This project investigates whether departing from these
defaults can improve variance reduction, predictor convergence, and computational efficiency,
along three axes, examined consistently for every derivative setting:

- **Auxiliary variable design** — payoff-aware constructions of `X = φ(Y)` in place of the
  generic terminal Brownian baseline.
- **Parameter sensitivity** — how variance reduction responds to the financial parameter regime
  (volatility, mean reversion, correlation, etc.) at which the estimator is evaluated.
- **Predictor complexity** — how the architecture of `g(θ, X)` (depth, width) affects convergence
  and realised variance reduction, and whether richer auxiliary variables demand a correspondingly
  different network to avoid overfitting.

Each axis is evaluated across derivative settings of increasing complexity: arithmetic Asian,
barrier, and lookback options under GBM and Heston dynamics; variance swaps under stochastic
local volatility (stretch); and swaptions under a Heath–Jarrow–Morton framework (stretch).

## Repository structure

```
.
├── option_pricing/            # Core module — all reusable logic lives here
│   ├── config.py               # Shared constants: time grid, evaluation point θ*, θ ranges
│   ├── models/
│   │   ├── gbm.py            
│   │   ├── heston.py        
│   │   ├── slv.py         
│   │   └── hjm.py   
│   ├── payoffs/
│   │   ├── asian.py
│   │   ├── barrier.py
│   │   ├── lookback.py
│   │   ├── variance_swap.py
│   │   └── swaption.py 
│   └── pemc/
│       ├── aux_variables.py     # All φ(Y) constructions, keyed by setting
│       ├── marginals.py         # Marginal samplers for each auxiliary variable
│       ├── predictor.py         # MLP, FlexibleMLP, Standardiser
│       ├── training.py          # train_predictor(), train_predictor_arch()
│       ├── estimator.py         # one_experiment(), experiment_at_theta(), generate_training_pool()
│       ├── evaluation.py        # rmse(), bootstrap_rmse_ci(), r_rho_c(), measure_costs()
│       └── plotting.py          # plot_convergence(), plot_sensitivity(), plot_complexity(), etc.
│
└── notebooks/                  # One folder per derivative setting
    ├── 01_asian_gbm/
    │   ├── 01_replication_aux_variables.ipynb   # PEMC baseline + auxiliary variable comparison
    │   ├── 02_parameter_sensitivity.ipynb       # Variance reduction across the θ parameter space
    │   └── 03_predictor_complexity.ipynb        # Variance reduction across network architectures
    ├── 02_asian_heston/
    │   └── ... (same three notebooks)
    ├── 03_barrier_heston/
    │   └── ...
    ├── 04_lookback_heston/
    │   └── ...
    ├── 05_variance_swaps_slv/   
    │   └── ...
    └── 06_swaptions_hjm/ 
        └── ...
```