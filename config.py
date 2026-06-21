"""Run configurations.  FULL reproduces the manuscript; QUICK is a fast smoke test."""

FULL = dict(
    name="full",
    n=32,                     # truth mesh (n x n cells)
    n_train=90, n_test=60,    # thermal parameter samples
    seed_train=7, seed_test=101,
    Nmax=26,                  # greedy iterations
    eff_Ns=[4, 8, 12, 16, 20, 24],
    flow_train=70, flow_test=25, flow_seed=3,
    flow_Nf=[1, 2, 4, 6, 8, 10, 14, 18, 24],
    defect_N=20, defect_Nf=8,
    timing_ns=[24, 32, 40, 48, 56],
    corners=[(0.0, 1.0, 1.0, 10.0), (0.1, 1.0, 1.5, 500.0),
             (0.2, 1.5, 2.0, 1000.0), (0.2, 1.5, 0.5, 1000.0)],
)

QUICK = dict(
    name="quick",
    n=16,
    n_train=18, n_test=12,
    seed_train=7, seed_test=101,
    Nmax=12,
    eff_Ns=[2, 4, 6, 8, 10],
    flow_train=25, flow_test=10, flow_seed=3,
    flow_Nf=[1, 2, 4, 6, 8],
    defect_N=8, defect_Nf=6,
    timing_ns=[12, 16, 20, 24],
    corners=[(0.0, 1.0, 1.0, 10.0), (0.1, 1.0, 1.5, 500.0),
             (0.2, 1.5, 2.0, 1000.0), (0.2, 1.5, 0.5, 1000.0)],
)
