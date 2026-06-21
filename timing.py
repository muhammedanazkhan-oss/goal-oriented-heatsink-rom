"""Online cost vs truth dimension: truth solve time grows with the mesh while the
affine reduced query (assemble + primal/dual solve + residual + correction) is
truth-dimension independent."""
import time
import numpy as np
import fem


def truth_times(ns, mu=(0.1, 1.0, 1.1, 300.0), reps=3):
    rows = []
    for n in ns:
        ref = fem.RefMesh(n=n)
        t = min(_time(lambda: fem.solve_truth(ref, mu)) for _ in range(reps))
        rows.append((ref.N_t, ref.N_u, t * 1e3))
    return rows


def online_query_time(cache_tr, V, W, Q=15, reps=4000):
    """Time one affine reduced certified query (fixed reduced size, N_h-independent)."""
    n = V.shape[1]
    anch = cache_tr[:Q]
    Aq = [V.T @ (c['Af'] @ V) for c in anch]
    AqT = [W.T @ (c['Af'].T @ W) for c in anch]
    fN = [V.T @ c['ff'] for c in anch]
    lN = [W.T @ c['lf'] for c in anch]
    Gpr = np.random.default_rng(0).standard_normal((Q * n, Q * n)); Gpr = Gpr @ Gpr.T
    rng = np.random.default_rng(1)
    def query(th):
        AN = sum(th[q] * Aq[q] for q in range(Q)); ANT = sum(th[q] * AqT[q] for q in range(Q))
        fn = sum(th[q] * fN[q] for q in range(Q)); ln = sum(th[q] * lN[q] for q in range(Q))
        c = np.linalg.solve(AN, fn); e = np.linalg.solve(ANT, ln)
        dNc = float(ln @ c) + float(fn @ e - c @ (AN.T @ e))
        z = np.concatenate([th[q] * c for q in range(Q)])
        return dNc, float(np.sqrt(abs(z @ (Gpr @ z))))
    ths = [rng.random(Q) for _ in range(reps)]
    t0 = time.perf_counter()
    for th in ths:
        query(th)
    return (time.perf_counter() - t0) / reps * 1e3


def _time(fn):
    t0 = time.perf_counter(); fn(); return time.perf_counter() - t0
