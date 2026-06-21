"""Parameter domain, feasibility, and sampling for the wavy heat-sink benchmark.

Parameter vector mu = (A, lam, chi, Pe):
    A    wave amplitude          in [0, 0.2]
    lam  wavelength              in [0.8, 1.5]
    chi  channel aspect ratio    in [0.5, 2.0]
    Pe   thermal Peclet number   in [10, 1000]   (log-uniform)
with the curvature-feasibility constraint  lam^2 / (4 pi^2 A) >= cD * Dh   (A>0).
"""
import numpy as np

BOX = ((0.0, 0.2), (0.8, 1.5), (0.5, 2.0), (10.0, 1000.0))


def feasible(mu, Dh=0.25, cD=1.0):
    A, lam, chi, Pe = mu
    return (A == 0.0) or (lam * lam / (4.0 * np.pi**2 * A) >= cD * Dh)


def sample_params(n, seed, box=BOX):
    """Latin-hypercube-style i.i.d. feasible sample (Pe log-uniform), reproducible by seed."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        A = rng.uniform(*box[0]); lam = rng.uniform(*box[1]); chi = rng.uniform(*box[2])
        Pe = 10.0**rng.uniform(np.log10(box[3][0]), np.log10(box[3][1]))
        if feasible((A, lam, chi, Pe)):
            out.append((A, lam, chi, Pe))
    return out


def sample_geometries(n, seed, box=BOX):
    """Feasible geometry sample (A, lam, chi) for the flow reduced-order model."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        A = rng.uniform(*box[0]); lam = rng.uniform(*box[1]); chi = rng.uniform(*box[2])
        if feasible((A, lam, chi, 1.0)):
            out.append((A, lam, chi))
    return out
