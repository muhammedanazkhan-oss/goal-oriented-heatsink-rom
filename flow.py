"""Supremizer-stabilized reduced Stokes for the pumping-power (compliance) output,
and the affine reduced-flow defects (eps_beta, eps_ell) entering the fully-affine
thermal online certificate.
"""
import numpy as np
import scipy.sparse.linalg as spla
from skfem import Basis, FacetBasis, LinearForm, BilinearForm, asm
import fem


class FlowROM:
    """Energy-POD velocity + pressure-POD + supremizer-enriched reduced Stokes saddle."""
    def __init__(self, ref, train_geoms):
        self.ref = ref
        o0 = fem.solve_stokes(ref, 0.1, 1.0, 1.0, return_ops=True)
        bu = o0['bu']
        di = bu.get_dofs('inlet'); dbo = bu.get_dofs('bot'); dto = bu.get_dofs('top')
        self.Ddir = np.unique(np.concatenate([di.all('u^1'), di.all('u^2'),
                                              dbo.all(), dto.all()]))
        self.Nu = bu.N
        self.freeu = np.setdiff1d(np.arange(self.Nu), self.Ddir)
        snaps = [fem.solve_stokes(ref, *g, return_ops=True) for g in train_geoms]
        self.b0 = snaps[0]['vu']
        Kff = snaps[0]['K'][self.freeu][:, self.freeu].tocsc()
        self.Kff = Kff
        self.Bf = snaps[0]['B'][:, self.freeu]
        FL = np.array([(s['vu'] - self.b0)[self.freeu] for s in snaps[1:]]).T
        G = FL.T @ (Kff @ FL)
        w, Vg = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1]; w = np.clip(w[idx], 1e-30, None); Vg = Vg[:, idx]
        self.Umodes = FL @ Vg / np.sqrt(w)          # energy-orthonormal velocity modes
        self.sv = np.sqrt(w)
        P = np.array([s['vp'] for s in snaps[1:]]).T
        self.Pu = np.linalg.svd(P - P.mean(1, keepdims=True), full_matrices=False)[0]

    def reduced_velocity(self, g, nu_modes, np_modes=6):
        """Return reduced velocity (full vector) and the truth Stokes ops at geometry g."""
        Uf = self.Umodes[:, :nu_modes]; Qp = self.Pu[:, :np_modes]
        Sup = np.zeros((len(self.freeu), np_modes))
        for j in range(np_modes):
            Sup[:, j] = spla.spsolve(self.Kff, self.Bf.T @ Qp[:, j])
        Wm, _ = np.linalg.qr(np.concatenate([Uf, Sup], axis=1))
        Wfull = np.zeros((self.Nu, Wm.shape[1])); Wfull[self.freeu, :] = Wm
        o = fem.solve_stokes(self.ref, *g, return_ops=True); K = o['K']; B = o['B']
        Kr = Wfull.T @ (K @ Wfull); Br = Qp.T @ (B @ Wfull)
        Z = np.zeros((np_modes, np_modes))
        rhs = np.concatenate([-Wfull.T @ (K @ self.b0), -Qp.T @ (B @ self.b0)])
        a = np.linalg.solve(np.block([[Kr, Br.T], [Br, Z]]), rhs)[:Wfull.shape[1]]
        return self.b0 + Wfull @ a, o


def pumping_study(ref, flow, test_geoms, Pe_ref, nf_list):
    """Pumping-power relative error and residual certificate vs flow dimension N_f."""
    te = [(g,) + (lambda r: (r['K'], r['vu']))(fem.solve_stokes(ref, *g, return_ops=True))
          for g in test_geoms]
    rows = []
    for nf in nf_list:
        errs, certs = [], []
        for g, K, vut in te:
            bN, _ = flow.reduced_velocity(g, nf)
            DT = float(vut @ (K @ vut)); DN = float(bN @ (K @ bN))
            db = float(np.sqrt(max((vut - bN) @ (K @ (vut - bN)), 0.0)))   # ||beta_h-beta_Nf||_aS
            bNn = float(np.sqrt(bN @ (K @ bN)))
            errs.append(abs(DN - DT) / DT)
            Dcert = (2 * bNn + db) * db                                    # certificate on |D_h-D_Nf|
            certs.append(Dcert / DT)
        rows.append(dict(Nf=nf, rel_err_med=float(np.median(errs)),
                         rel_err_max=float(np.max(errs)),
                         cert_rel_med=float(np.median(certs)),
                         cert_eff_med=float(np.median([c / e for c, e in zip(certs, errs)
                                                       if e > 1e-12]))))
    return rows


def _energy_with(ref, vu, md, Pe):
    """Energy operator (free-dof) assembled with a prescribed velocity vector vu."""
    stk = {'md': md, 'bu': Basis(md, ref.eu), 'vu': vu}
    eo = fem.energy_operators(ref, stk, Pe)
    L = fem.output_functionals(ref, stk)['LdT']
    f = np.setdiff1d(np.arange(ref.N_t), eo['bt'].get_dofs('inlet').all())
    return eo['Ae'][f][:, f].tocsc(), eo['fe'][f], L[f]


def defect_study(ref, red, flow, cache_te, Vgo, Wgo, N, nf):
    """Reduced-flow operator/functional defects and fully-affine vs truth-operator bound."""
    from reduction import reduced_primal, reduced_dual
    rows = []
    for c in cache_te:
        A, lam, chi, Pe = c['mu']
        bN, ostk = flow.reduced_velocity((A, lam, chi), nf)
        AfN, ffN, lfN = _energy_with(ref, bN, ostk['md'], Pe)
        Af, ff, lf, al = c['Af'], c['ff'], c['lf'], c['alpha']
        uN = reduced_primal(Af, ff, Vgo[:, :N]); zM = reduced_dual(Af, lf, Wgo[:, :N])
        dpr = red.dnorm((Af - AfN) @ uN)            # eps_beta * ||u_N||
        ddu = red.dnorm((Af - AfN).T @ zM)
        eell = red.dnorm(lf - lfN)                  # eps_ell
        Dtruth = red.dnorm(ff - Af @ uN) * red.dnorm(lf - Af.T @ zM) / al
        Daff = (red.dnorm(ffN - AfN @ uN) + dpr) * (red.dnorm(lfN - AfN.T @ zM) + ddu + eell) / al
        rows.append(dict(eps_beta_uN=dpr, dual_defect=ddu, eps_ell=eell,
                         Dtruth=Dtruth, Daff=Daff, ratio=Daff / Dtruth))
    return rows
