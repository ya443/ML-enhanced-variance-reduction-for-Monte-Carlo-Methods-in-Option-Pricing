# Research Workflow, AI Use and Version Control

## Purpose

This document explains how the codebase for the dissertation was developed, checked and version-controlled. It is intended to provide a clear record of the computational workflow behind the results, with particular attention to:

- how the repository was organised;
- how experiments were developed and reproduced;
- how Git was used to preserve the history of the project;
- how generative AI supported the research;
- how AI-generated suggestions were checked before being used.

The project investigates prediction-enhanced Monte Carlo (PEMC) for pricing path-dependent options under geometric Brownian motion and the Heston stochastic-volatility model. The contracts considered include Asian, barrier and lookback options.

The overall workflow was:

```text
Research question
→ model and payoff implementation
→ auxiliary-variable design
→ predictor training
→ repeated Monte Carlo evaluation
→ figures and tables
→ critical interpretation
→ dissertation write-up
```

The aim was not only to obtain a lower RMSE, but to understand why a method worked, whether the result was computationally efficient, and whether the conclusion remained valid under different parameters and model choices.

---

## 1. Repository design

The repository was organised so that the core mathematical components were separated from the experimental notebooks. A representative structure is:

```text
option-pricing/
├── README.md
├── requirements.txt
├── config.py
├── option_pricing/
│   ├── models/
│   │   ├── gbm.py
│   │   └── heston.py
│   ├── payoffs/
│   │   ├── asian.py
│   │   ├── barrier.py
│   │   └── lookback.py
│   └── pemc/
│       ├── aux_variables.py
│       ├── marginals.py
│       ├── estimator.py
│       └── evaluation.py
├── notebooks/
├── figures/
└── tests/
```

The main reason for this separation was traceability. The notebooks were used to configure and run experiments, while the reusable model, payoff and estimator logic was kept in Python modules.

This reduced duplication and made it easier to answer questions such as:

- whether two notebooks were using the same Heston simulator;
- whether a payoff function had changed between experiments;
- whether the marginal sampler was independent of the expensive path sample;
- whether a difference in results came from the feature, the model, or the experiment configuration.

---

## 2. Mathematical workflow

### 2.1 Stochastic models

Under geometric Brownian motion, the asset follows

$$
dS_t = rS_t\,dt + \sigma S_t\,dW_t.
$$

Under the Heston model,

$$
dS_t = rS_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S,
$$

$$
dv_t = \kappa(\eta-v_t)\,dt
+\delta\sqrt{v_t}\,dW_t^v,
$$

with

$$
d\langle W^S,W^v\rangle_t = \rho_H\,dt.
$$

The Heston implementation uses a full-truncation Euler scheme. This was chosen because an ordinary Euler approximation can produce negative variance values, which are incompatible with the square-root diffusion term.

The Feller condition was also checked:

$$
2\kappa\eta \geq \delta^2.
$$

Where this condition was violated, the result was reported explicitly because it affects how frequently the variance process approaches zero and how important the truncation rule becomes.

### 2.2 Prediction-enhanced Monte Carlo

For an expensive simulated path $Y$, payoff $f(Y)$ and auxiliary variable $X=\phi(Y)$, the PEMC estimator is

$$
\widehat{\mu}_{\mathrm{PEMC}}
=
\frac{1}{n}
\sum_{i=1}^{n}
\left[
f(Y_i)-g(\theta,X_i)
\right]
+
\frac{1}{N}
\sum_{j=1}^{N}
g(\theta,\widetilde X_j).
$$

Here:

- $n$ is the number of expensive full-path observations;
- $N$ is the number of auxiliary-only observations;
- $g(\theta,X)$ is the trained predictor;
- $\widetilde X_j$ is an independent draw from the marginal distribution of the auxiliary variable.

The estimator variance is

$$
\operatorname{Var}
\left(
\widehat{\mu}_{\mathrm{PEMC}}
\right)
=
\frac{\operatorname{Var}(f-g)}{n}
+
\frac{\operatorname{Var}(g)}{N}.
$$

This formula was central to the interpretation of the results. It shows that high correlation between $f$ and $g$ is helpful, but not sufficient on its own. The final estimator also depends on the scale of the predictions and the variance of the auxiliary correction term.

The predictor was trained to approximate

$$
g^*(\theta,x)
=
\mathbb{E}_{\theta}
\left[
f(Y)\mid X=x
\right].
$$

This conditional-expectation interpretation guided the design of the auxiliary variables. Features were chosen according to the structure of each payoff rather than selected purely through trial and error.

---

## 3. Experimental workflow

Each option and model combination was studied using three main experiments.

### 3.1 Auxiliary-variable comparison

The first experiment asked:

> Which low-dimensional path summaries are most informative for the payoff, and are they cheap enough to be useful?

For each auxiliary variable, the analysis recorded:

- feature dimension;
- financial interpretation;
- validation MSE;
- payoff-prediction correlation;
- empirical PEMC variance ratio;
- percentage RMSE reduction;
- full-path cost $c_f$;
- auxiliary cost $c_g$;
- cost ratio $c=c_g/c_f$.

The cost ratio was defined as

$$
c=\frac{c_g}{c_f}.
$$

The total PEMC cost is approximately

$$
C_{\mathrm{PEMC}}
=
nc_f+Nc_g.
$$

Since the main experiments used $N=10n$,

$$
C_{\mathrm{PEMC}}
=
nc_f(1+10c).
$$

This was important because a richer auxiliary variable could produce a large RMSE reduction at fixed sample sizes while still being unattractive under an equal computational budget.

### 3.2 Parameter sensitivity

The second experiment asked:

> Does a predictor trained once over a parameter domain remain useful away from the central calibration?

The trained network was held fixed while selected model or option parameters were varied. This tested the intended “train once, price many times” use case.

The analysis considered:

- whether correlation rose or fell;
- whether the ranking of the auxiliary variables changed;
- whether the effect agreed with the financial structure of the model;
- whether the central parameter value was included in the sensitivity grid;
- whether the test represented interpolation or extrapolation.

The one-factor-at-a-time design was useful for interpretation, but its limitations were acknowledged because it does not capture interactions between parameters.

### 3.3 Predictor complexity

The third experiment asked:

> Does increasing neural-network capacity materially improve the final PEMC estimator?

Linear, shallow, medium, deep and wide models were compared while holding the feature and training data fixed.

The experiment distinguished between:

- **capacity limitation**, where the model is too simple;
- **information limitation**, where the auxiliary variable omits important path information.

A plateau in validation MSE was interpreted as diminishing returns from additional capacity. A decline in estimator RMSE performance was not automatically labelled overfitting unless the training and validation evidence supported that conclusion.

---

## 4. Development and paper-scale configurations

The notebooks use a Boolean switch to separate rapid development runs from more expensive final runs.

The development configuration is:

```python
PAPER_SCALE = False
N_TRAIN = 150_000
EPOCHS = 25
N_SEEDS = 30
```

The larger configuration is:

```python
PAPER_SCALE = True
N_TRAIN = 400_000
EPOCHS = 30
N_SEEDS = 60
```

The purpose of this switch was to keep the code path unchanged while allowing smaller experiments during development. This avoided maintaining separate “test” and “final” notebooks that could gradually diverge.

Every reported result should state which configuration was used.

---

## 5. Version control using Git

Git was used throughout the project to preserve the development history and make changes reversible.

### 5.1 Basic workflow

The standard workflow was:

```bash
git status
git add .
git commit -m "Describe the change clearly"
git push
```

Before committing, `git status` was used to check which files had changed. This reduced the risk of uploading temporary files, generated outputs or unrelated edits.

### 5.2 Commit strategy

Commits were made around meaningful changes rather than arbitrary time intervals. Examples include:

```text
Implement Heston full-truncation simulator
Add floating-strike lookback payoff
Add exact Gaussian marginal for terminal Heston feature
Add parameter sensitivity experiment
Fix independent auxiliary sampling in PEMC estimator
Add cost-adjusted comparison figures
Clarify fixed-sample versus fixed-budget interpretation
```

This made the repository history easier to read and helped isolate the source of any unexpected numerical change.

### 5.3 Why version control mattered

Version control was particularly useful for this project because changes to simulation code can alter every downstream figure and conclusion.

Git allowed me to:

- identify when a model or payoff function changed;
- restore an earlier working implementation;
- compare two versions of the estimator;
- separate code refactoring from methodological changes;
- preserve the exact code used to generate a result;
- avoid overwriting a stable implementation while testing a new idea.

Before finalising a result, the current commit can be recorded using:

```bash
git rev-parse HEAD
```

This provides a direct link between the dissertation result and the repository state that produced it.

### 5.4 Branching for experimental changes

Where a change was uncertain or potentially disruptive, it could be developed on a separate branch:

```bash
git checkout -b feature/new-marginal-sampler
```

After testing, the branch could be merged into the main branch:

```bash
git checkout main
git merge feature/new-marginal-sampler
```

This was useful for changes such as:

- introducing a new auxiliary variable;
- replacing a marginal sampler;
- restructuring the repository;
- changing the Heston discretisation;
- testing a new architecture.

The project did not require an elaborate branching system. The main principle was simply to avoid mixing unstable experimental changes with the most recent verified version.

---

## 6. Reproducing the experiments

A new user should be able to reproduce the workflow using the following steps.

### 6.1 Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

### 6.2 Run basic checks

```bash
python -m pytest
```

### 6.3 Run the notebooks

The recommended order is:

1. auxiliary-variable comparison;
2. parameter sensitivity;
3. predictor complexity.

The configuration cell should be run first so that the seed, sample sizes and `PAPER_SCALE` setting are visible in the output.

### 6.4 Reproduction checks

A successful reproduction should confirm that:

- the same parameter vector is used;
- the same master seed is used;
- the auxiliary sample is independent of the expensive sample;
- $N=N_{\mathrm{RATIO}}n$;
- the benchmark estimate is close within Monte Carlo uncertainty;
- figures are generated directly from executed results;
- the reported RMSE reduction is calculated from the repeated estimates rather than copied manually.

---

## 7. Use of generative AI

Generative AI was used as a research-support tool. Its main roles were:

- explaining mathematical derivations;
- reviewing code for conceptual errors;
- suggesting more precise experimental comparisons;
- challenging interpretations that were too strong;
- helping structure Markdown and LaTeX;
- improving the clarity of research prompts;
- helping identify limitations that required further checking.

AI was not treated as a numerical source. It did not replace executed code, mathematical verification or judgement about whether a conclusion was supported.

### 7.1 Conceptual assistance

One important use was clarifying the difference between predictive quality and estimator quality.

A representative prompt was:

> Derive the PEMC variance formula from the estimator and explain why high correlation between the payoff and predictor is not sufficient to guarantee the lowest final RMSE. Include the role of predictor scale and the finite auxiliary-sample term.

The resulting explanation was checked directly from

$$
\operatorname{Var}(f-g)
=
\operatorname{Var}(f)
+
\operatorname{Var}(g)
-
2\operatorname{Cov}(f,g).
$$

Another prompt was:

> Explain precisely what $c_f$, $c_g$ and $c=c_g/c_f$ measure in the implementation. Show how the total cost changes when $N=10n$, and distinguish a fixed-sample comparison from an equal-budget comparison.

This was checked against the timing code and

$$
C_{\mathrm{PEMC}}=nc_f+Nc_g.
$$

### 7.2 Critical review

AI was also used as a simulated critical reviewer.

A representative prompt was:

> Audit this experiment as if you were reviewing a dissertation. Identify any claim that is stronger than the evidence supports. Check benchmark uncertainty, discretisation bias, unequal runtime, the use of one training seed, the number of repeated pricing seeds and whether the sensitivity grid includes the central parameter value.

This was useful because it forced the interpretation to address weaknesses rather than only reporting the best result.

Another prompt was:

> The combined feature has the best validation MSE and the highest correlation, but another feature has the lowest empirical RMSE. Give mathematically plausible explanations without assuming that the code must be wrong.

This helped separate regression quality from PEMC estimator quality.

### 7.3 Experimental design

Representative prompts included:

> Design an architecture comparison that distinguishes underfitting from an information ceiling. State which quantities should be recorded in addition to validation MSE.

> Propose a parameter-sensitivity experiment that tests “train once, price many times” without retraining the network at every parameter point. Explain the limitations of a one-factor-at-a-time design.

> For a floating-strike lookback call under Heston, compare the information contained in terminal Brownian values, the running minimum, the running minimum plus integrated variance, and the running minimum and maximum.

These suggestions were used only after checking that they were consistent with the payoff definition and the stochastic model.

### 7.4 Code review

Representative code-review prompts included:

> Check whether the auxiliary marginal sample is independent of the expensive path sample. Identify any reuse of random draws that would invalidate the estimator.

> Check whether the timing functions for $c_f$ and $c_g$ measure comparable per-observation quantities. List exactly which operations are included in each timing block.

> Refactor repeated notebook logic into reusable model, payoff, auxiliary, marginal and estimator modules without changing the numerical experiment.

AI-generated code suggestions were reviewed line by line before being used. Where a refactor changed numerical outputs unexpectedly, the change was treated as a potential behavioural change rather than assumed to be harmless.

### 7.5 Writing support

AI was used to help structure technical explanations and improve consistency. Representative prompts included:

> Rewrite this results section so that it separates the numerical result, the theoretical interpretation and the limitation.

> Convert the internal headings into LaTeX paragraph headings while preserving equations, tables, figure labels and cross-references.

> Write a conclusion that states both the fixed-sample winner and the cost-adjusted winner without presenting them as contradictory.

The final text was checked against the notebooks. Generated prose was edited or rejected where it overstated the evidence.

---

## 8. How AI outputs were verified

AI output was checked using four main methods.

### 8.1 Mathematical verification

Equations and derivations were checked manually against the estimator definitions. This was particularly important for:

- unbiasedness of PEMC;
- the PEMC variance formula;
- the predictor-scale interpretation;
- the cost ratio;
- the Feller condition;
- homogeneity of the floating-strike lookback payoff.

### 8.2 Code verification

Suggestions about implementation were checked against:

- array shapes;
- random-number independence;
- feature definitions;
- marginal distributions;
- repeated-run loops;
- timing blocks;
- output tables.

### 8.3 Numerical verification

Any claim involving a value, ranking or percentage was checked against executed notebook output.

Examples include:

- benchmark estimates and standard errors;
- payoff-prediction correlations;
- validation MSE;
- empirical variance ratios;
- RMSE reductions;
- cost ratios;
- training times.

### 8.4 Interpretation verification

Interpretations were accepted only when they were consistent with both the mathematics and the experiment.

For example:

- high correlation was not treated as sufficient evidence of the lowest RMSE;
- a path-aware feature was not automatically described as computationally efficient;
- a deep network performing worse was not automatically described as overfitting;
- a one-million-path Heston estimate was described as a discretised reference value, not an exact price;
- a validation-MSE curve was not described as a formal bias-variance decomposition.

---

## 9. Evidence of independent judgement

The use of AI did not remove the need for independent decisions. The main evidence of research judgement lies in the choices made after reviewing the outputs.

Examples include:

- retaining both fixed-sample and cost-adjusted comparisons;
- reporting benchmark uncertainty;
- distinguishing Monte Carlo error from Heston discretisation error;
- identifying when a sensitivity grid omitted the central calibration;
- rejecting the claim that the highest-correlation feature must be the best PEMC estimator;
- recognising when increased architecture size had reached an information ceiling;
- retaining negative or inconclusive results instead of presenting only favourable findings;
- selecting payoff-aware auxiliary variables from the mathematical structure of each contract.

AI was most useful when it was given a precise question and asked to challenge a result. The final responsibility for deciding what entered the codebase and dissertation remained with the researcher.

---

## 10. Limitations of the workflow

The workflow still has limitations:

- some experiments were run using `PAPER_SCALE=False`;
- most networks were trained once per architecture;
- one-factor-at-a-time sensitivity does not test parameter interactions;
- Heston benchmarks inherit discretisation error;
- some path-based auxiliary marginals are expensive to generate;
- close RMSE rankings may be sensitive to the number of repeated seeds;
- the fixed ratio $N=10n$ may not be optimal for every auxiliary variable.

These limitations were reported because reproducibility is not only the ability to rerun code. It also requires clarity about what the experiment does and does not establish.

---

## 11. Summary

The project used a modular repository, repeated experiments, explicit configurations and Git version control to maintain traceability from code to dissertation conclusions.

Generative AI supported explanation, criticism, code review and writing, but it was not treated as evidence. Mathematical statements were checked from first principles, code suggestions were reviewed, and numerical claims were verified against executed outputs.

The central principle was:

> A result was not accepted because it sounded plausible. It was accepted only when the code, mathematics and numerical evidence agreed.
