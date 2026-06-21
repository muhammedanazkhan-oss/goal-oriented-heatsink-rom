"""High-fidelity (truth) model: coupled Stokes flow + skew-symmetric SUPG
convection-diffusion energy transport for a parametrized forced-convection
wavy-channel heat sink.  mu = (A, lam, chi, Pe).

Reference cell (0,1)^2 deformed (piecewise-affine, on the mesh vertices) by
    Phi(xh, yh) = ( xh ,  chi*yh + A sin(2 pi xh / lam) ).
Flow:  P2/P1 Taylor-Hood, vector-Laplacian form, traction-free (do-nothing) outflow.
Energy: P1 elements; the convective term is in skew-symmetric (conservative) form
        with the (beta.n)_+ outlet (backflow-stabilized) term, so coercivity holds
        for any (weakly divergence-free) velocity.
Outputs: d = Tw - Tb (linear, certified), Nu_ref = 2 chi / d, pumping P = Pe^2 * D_geo.
"""
import numpy as np
import dataclasses
import scipy.sparse as sp
from skfem import (MeshTri, Basis, FacetBasis, ElementTriP2, ElementTriP1,
                   ElementVector, BilinearForm, LinearForm, asm, condense, solve)
from skfem.helpers import grad, dot, ddot


def tri_area(md):
    t, p = md.t, md.p
    x, y = p[0, t], p[1, t]
    return 0.5 * np.abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))


class RefMesh:
    """Fixed-topology reference triangulation of (0,1)^2 with tagged boundaries."""
    def __init__(self, n=32):
        xs = np.linspace(0, 1, n + 1)
        m = MeshTri.init_tensor(xs, xs).with_boundaries({
            'inlet':  lambda p: p[0] < 1e-9,
            'outlet': lambda p: p[0] > 1 - 1e-9,
            'bot':    lambda p: p[1] < 1e-9,
            'top':    lambda p: p[1] > 1 - 1e-9})
        self.m = m
        self.eu = ElementVector(ElementTriP2())   # velocity
        self.ep = ElementTriP1()                   # pressure
        self.et = ElementTriP1()                   # temperature
        self.bu = Basis(m, self.eu)
        self.bt = Basis(m, self.et)
        self.N_u, self.N_t = self.bu.N, self.bt.N

    def deformed(self, A, lam, chi):
        p = self.m.p
        p2 = np.vstack([p[0], chi * p[1] + A * np.sin(2 * np.pi * p[0] / lam)])
        return dataclasses.replace(self.m, doflocs=p2)


@BilinearForm
def _a_stokes(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def _b_div(u, q, w):
    from skfem.helpers import div
    return div(u) * q


def solve_stokes(ref, A, lam, chi, return_ops=False):
    """Unit-mean-inlet Stokes flow; returns velocity/pressure and dissipation D_geo."""
    md = ref.deformed(A, lam, chi)
    bu = Basis(md, ref.eu); bp = bu.with_element(ref.ep)
    K = asm(_a_stokes, bu); B = asm(_b_div, bu, bp)
    Z = sp.csr_matrix((bp.N, bp.N))
    Kmat = sp.bmat([[K, B.T], [B, Z]], format='csr')
    di = bu.get_dofs('inlet'); dbo = bu.get_dofs('bot'); dto = bu.get_dofs('top')
    ix = di.all('u^1'); iy = di.all('u^2')
    wall = np.concatenate([dbo.all(), dto.all()])
    Ntot = bu.N + bp.N; x = np.zeros(Ntot)
    Yc = bu.doflocs[1, ix]; yh = Yc / chi
    x[ix] = 6.0 * yh * (1.0 - yh)                  # parabolic inlet, unit mean
    D = np.unique(np.concatenate([ix, iy, wall]))
    sol = solve(*condense(Kmat, np.zeros(Ntot), x=x, D=D))
    vu, vp = sol[:bu.N], sol[bu.N:]
    D_geo = float(vu @ (K @ vu))                   # viscous dissipation (= flow x pressure drop)
    out = dict(md=md, bu=bu, bp=bp, vu=vu, vp=vp, D_geo=D_geo)
    if return_ops:
        out.update(K=K, B=B)
    return out


def energy_operators(ref, stk, Pe, qflux=1.0, delta=0.5):
    """Assemble the skew-symmetric SUPG energy operator and the fixed-flux load."""
    qflux = qflux / Pe
    md = stk['md']; bt = Basis(md, ref.et, intorder=4); bu = stk['bu']
    bvel = bt.with_element(ref.eu)
    wfield = bvel.interpolate(stk['vu'])
    h = np.sqrt(2.0 * tri_area(md)); nqp = bt.X.shape[1]
    hq = np.repeat(h[:, None], nqp, axis=1)
    speed = np.sqrt(wfield[0]**2 + wfield[1]**2)
    tau = delta * hq * speed / (2.0 * speed**2 + 1e-9)

    @BilinearForm
    def a_energy(T, v, w):
        wx, wy = w['vel'][0], w['vel'][1]
        bgT = wx * grad(T)[0] + wy * grad(T)[1]
        bgv = wx * grad(v)[0] + wy * grad(v)[1]
        conv = 0.5 * bgT * v - 0.5 * bgv * T       # skew-symmetric convective form
        diff = (1.0 / Pe) * dot(grad(T), grad(v))
        supg = w['tau'] * bgT * bgv
        return conv + diff + supg

    Ae = asm(a_energy, bt, vel=wfield, tau=tau)
    # backflow-stabilized skew outlet term  1/2 int_{Gamma_out} (beta.n)_+ T v
    fbo = FacetBasis(md, ref.et, facets=md.boundaries['outlet'])
    un_o = np.maximum(fbo.with_element(bu.elem).interpolate(stk['vu'])[0], 0.0)
    Bout = asm(BilinearForm(lambda T, v, w: 0.5 * w['un'] * T * v), fbo, un=un_o)
    Ae = (Ae + Bout).tocsr()
    fb = FacetBasis(md, ref.et, facets=md.boundaries['bot'])
    fe = asm(LinearForm(lambda v, w: qflux * v), fb)
    Dt = bt.get_dofs('inlet').all()
    return dict(bt=bt, Ae=Ae, fe=fe, Dt=Dt, tau=tau, wfield=wfield)


def solve_energy(ref, stk, Pe, qflux=1.0, eops=None, delta=0.5):
    if eops is None:
        eops = energy_operators(ref, stk, Pe, qflux, delta=delta)
    T = solve(*condense(eops['Ae'], eops['fe'], x=np.zeros(ref.N_t), D=eops['Dt']))
    eops['T'] = T
    return T, eops


def output_functionals(ref, stk):
    """Wall-temperature, bulk-temperature and wall-to-bulk (d) functionals."""
    md = stk['md']; bu = stk['bu']
    bot = md.boundaries['bot']
    midx = md.p[0, md.facets[:, bot]].mean(axis=0)
    dh = bot[midx >= 0.5]                           # downstream half of the heated wall
    fbw = FacetBasis(md, ref.et, facets=dh)
    Lw = asm(LinearForm(lambda v, w: v), fbw)
    aw = asm(LinearForm(lambda v, w: v), fbw).sum(); Lw = Lw / aw
    out = md.boundaries['outlet']
    fbo_t = FacetBasis(md, ref.et, facets=out); fbo_u = fbo_t.with_element(bu.elem)
    un = fbo_u.interpolate(stk['vu'])[0]
    Lb = asm(LinearForm(lambda v, w: w['un'] * v), fbo_t, un=un)
    W = asm(LinearForm(lambda v, w: w['un'] * v), fbo_t, un=un).sum(); Lb = Lb / W
    return dict(Lw=Lw, Lb=Lb, LdT=Lw - Lb)


def solve_truth(ref, mu, qflux=1.0, delta=0.5):
    """Full high-fidelity solve; returns operators, outputs d, Nu_ref, dissipation, pumping."""
    A, lam, chi, Pe = mu
    stk = solve_stokes(ref, A, lam, chi, return_ops=True)
    T, eops = solve_energy(ref, stk, Pe, qflux, delta=delta)
    funcs = output_functionals(ref, stk)
    d = float(funcs['LdT'] @ T)
    Nu = 2.0 * chi / d if d > 0 else np.nan
    return dict(stk=stk, eops=eops, funcs=funcs, T=T, d=d, Nu=Nu,
                Ppump=Pe**2 * stk['D_geo'], D_geo=stk['D_geo'])


if __name__ == "__main__":
    ref = RefMesh(n=32)
    o = solve_stokes(ref, 0.0, 1.0, 1.0)
    print(f"straight-channel dissipation D_geo = {o['D_geo']:.5f} (exact 12)")
    for mu in [(0.0, 1.0, 1.0, 50.0), (0.15, 0.9, 1.5, 800.0)]:
        r = solve_truth(ref, mu)
        print(f"mu={mu}: d={r['d']:.5f}  Nu_ref={r['Nu']:.3f}  P={r['Ppump']:.3e}")
