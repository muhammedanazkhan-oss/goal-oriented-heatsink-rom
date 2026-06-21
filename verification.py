"""Truth-model verification: straight-channel dissipation, mesh convergence (incl. at
parameter corners), steady convective heat-recovery deficit, outlet backflow check,
SUPG-parameter sensitivity, weak-divergence measure, and the Taylor-Hood inf-sup constant.
"""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import Basis, FacetBasis, LinearForm, BilinearForm, asm
from skfem.helpers import grad, div, ddot
import fem


def straight_dissipation(n=40):
    ref = fem.RefMesh(n=n)
    return {chi: fem.solve_stokes(ref, 0.0, 1.0, chi)['D_geo'] for chi in (0.5, 1.0, 2.0)}


def mesh_convergence(mu, ns):
    return [(n, fem.solve_truth(fem.RefMesh(n=n), mu)['d'],
             fem.solve_truth(fem.RefMesh(n=n), mu)['Nu']) for n in ns]


def mesh_convergence_corners(corners, ns=(32, 48, 64)):
    rows = []
    for mu in corners:
        ds = [fem.solve_truth(fem.RefMesh(n=n), mu)['d'] for n in ns]
        order = (np.log(abs(ds[1] - ds[0]) / abs(ds[2] - ds[1])) / np.log(ns[1] / ns[0])
                 if abs(ds[2] - ds[1]) > 1e-13 else np.nan)
        rows.append(dict(mu=mu, d=ds, order=order, fe_err=abs(ds[2] - ds[0]) / abs(ds[2])))
    return rows


def heat_balance_and_backflow(mu, n=48):
    """Convective heat-recovery deficit |wall_in - conv_out|/wall_in and min outlet velocity."""
    ref = fem.RefMesh(n=n); r = fem.solve_truth(ref, mu); md = r['stk']['md']; Pe = mu[3]
    Lh = asm(LinearForm(lambda v, w: v), FacetBasis(md, ref.et, facets=md.boundaries['bot'])).sum()
    fbo = FacetBasis(md, ref.et, facets=md.boundaries['outlet'])
    un = fbo.with_element(r['stk']['bu'].elem).interpolate(r['stk']['vu'])[0]
    conv_out = asm(LinearForm(lambda v, w: w['un'] * v), fbo, un=un) @ r['T']
    return dict(mu=mu, deficit=abs(Lh / Pe - conv_out) / abs(Lh / Pe),
                min_bn=float(un.min()), d=r['d'], Nu=r['Nu'],
                Dgeo=r['D_geo'], Ppump=r['Ppump'])


def tau_sensitivity(mu, deltas=(0.25, 0.5, 1.0, 2.0), n=40):
    ref = fem.RefMesh(n=n)
    return [(de, fem.solve_truth(ref, mu, delta=de)['d']) for de in deltas]


def divergence_measure(geoms, n=40):
    """||div beta_h|| / ||grad beta_h|| for the Taylor-Hood velocity."""
    ref = fem.RefMesh(n=n); out = []
    for g in geoms:
        o = fem.solve_stokes(ref, *g, return_ops=True); bu = o['bu']; vu = o['vu']
        dd = asm(BilinearForm(lambda u, v, w: div(u) * div(v)), bu)
        num=np.sqrt(max(vu @ (dd @ vu), 0.0)); den=np.sqrt(max(vu @ (o['K'] @ vu), 0.0))
        out.append((g, num/den if den>0 else 0.0))
    return out


def stokes_infsup(geoms, n=24):
    """Truth Taylor-Hood discrete inf-sup constant gamma_h(mu)."""
    ref = fem.RefMesh(n=n); out = []
    for g in geoms:
        o = fem.solve_stokes(ref, *g, return_ops=True); bu = o['bu']; bp = bu.with_element(ref.ep)
        di = bu.get_dofs('inlet'); dbo = bu.get_dofs('bot'); dto = bu.get_dofs('top')
        Dd = np.unique(np.concatenate([di.all('u^1'), di.all('u^2'), dbo.all(), dto.all()]))
        fr = np.setdiff1d(np.arange(bu.N), Dd)
        K = o['K'][fr][:, fr].tocsc(); B = o['B'][:, fr]
        Mp = asm(BilinearForm(lambda p, q, w: p * q), bp).tocsc()
        Klu = spla.splu(K)
        S = spla.LinearOperator((B.shape[0],) * 2, matvec=lambda x: B @ Klu.solve(B.T @ x))
        vals = spla.eigsh(S, k=3, M=Mp, sigma=1e-6, which='LM', return_eigenvectors=False)
        vals = np.sort(vals[vals > 1e-9])
        out.append((g, float(np.sqrt(vals[0])) if len(vals) else float('nan')))
    return out
