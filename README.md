# Goal-Oriented Certified ROM for Forced-Convection Heat Sinks

Reproducibility code for the paper

> M. A. Khan, *Goal-Oriented Error Estimation and Adaptive Model Reduction for
> Engineering Outputs in Forced-Convection Heat Sinks.*

It implements a coupled **Stokes / SUPG convection–diffusion** truth model on a
parametrized wavy-channel cell and a **goal-oriented (dual-weighted-residual)
certified reduced-order model** for two engineering outputs — a Nusselt-like
wall-to-bulk thermal output and the pumping power — together with an
output-adaptive greedy, an anchored-norm a posteriori certificate, a
supremizer-stabilized reduced Stokes model, and the full numerical study
(effectivity, rigor, online cost, affine defects, verification).

Parameter vector: `mu = (A, lam, chi, Pe)` — amplitude, wavelength, aspect ratio,
Péclet number.

## Installation

Python 3.9+ with:

```bash
pip install -r requirements.txt      # numpy, scipy, scikit-fem, matplotlib
```

No compiled extensions or external meshers are required (scikit-fem is pure Python).

## Running

By default the package runs only a lightweight **self-test** that verifies the solver
and certificate are wired correctly **without writing any results or figures**:

```bash
python run_all.py            # self-test only (writes nothing)
```

The full numerical study and figures are **gated** behind an explicit flag and are
*not* produced by default:

```bash
python run_all.py --reproduce            # FULL study; reproduces the manuscript numbers
python run_all.py --reproduce --quick    # fast coarse-mesh check
python run_all.py --reproduce --no-figures
```

When `--reproduce` is used, outputs are written to:

* `results/` — machine-readable `.csv` summaries and `.pkl/.npy` data
* `figures/` — `.pdf` and `.png` figures

Reproducibility is deterministic: all random samples are seeded in `config.py`
(`seed_train=7`, `seed_test=101`, `flow_seed=3`).

> **Confidentiality note.** This archive contains only source code (the methods
> described in the paper); it ships **no** results or figures. The full study is
> reproducible by running `--reproduce`, so the public deposit (with the DOI) is
> intended for **publication time**. Until then the generated results and figures
> should be kept confidential.

## What each stage produces (and where it appears in the paper)

| Stage (`run_all.py`) | Output files | Paper item |
|---|---|---|
| truth cache | (in-memory) | Sec. 2 |
| goal-oriented vs field greedy | `greedy.csv`, `greedy_hist.pkl`, `fig_greedy` | Fig. 3, Table 3 |
| effectivity + rigor | `effectivity.csv`, `effectivity.pkl`, `fig_effectivity`, `fig_rigor` | Figs. 4–5, Table 4 |
| pumping-power reduced Stokes | `pumping.csv`, `flow.pkl`, `fig_flow` | Fig. 6, Table 5 |
| affine reduced-flow defects | `defects.csv`, `defects.pkl` | Table 6 |
| online cost | `timing.csv`, `timing.pkl`, `fig_timing` | Fig. 7, Table 7 |
| verification | `verification_corners.csv`, `verification.pkl` | Tables 1–2 |
| Pareto trade-off | `pareto.pkl`, `fig_pareto` | Fig. 8 |
| fields (velocity/temperature/adjoint) | `fig_fields` | Fig. 2 |

## Module layout

```
params.py         parameter domain, feasibility, sampling
fem.py            truth model: Stokes (P2/P1) + skew-symmetric SUPG energy (P1); outputs
reduction.py      anchored-norm Reductor, stability constant, DWR bounds, greedy
flow.py           supremizer reduced Stokes; pumping certificate; affine defects
verification.py   mesh convergence, heat balance, backflow, tau sensitivity, inf-sup
timing.py         truth vs affine online query timing
figures.py        all figures
config.py         QUICK / FULL run configurations
run_all.py        orchestrator
```

## Method notes

* Flow: vector-Laplacian Stokes, traction-free (do-nothing) outflow; the
  straight-channel dissipation reproduces `12/chi` to machine precision.
* Energy: P1 on the piecewise-affine deformed mesh (so the mapped diffusion
  tensor is element-wise constant). The convective term is in **skew-symmetric
  (conservative)** form with a backflow-stabilized `(beta.n)_+` outlet term, so
  the bilinear form is coercive for any (weakly divergence-free) velocity.
* Certification uses a fixed **anchored** energy norm and an offline-computed
  stability constant `alpha(mu)` (rigorous to the eigensolver tolerance); a
  successive-constraint lower bound is the fully-online alternative.

## License

MIT (see `LICENSE`).
