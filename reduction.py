"""Goal-oriented certified reduced-basis machinery for the thermal output.

Certification is in a FIXED anchored energy norm  X0 = a_h^sym(.,.;mu0)  (mu0 at the
convective corner), with a stability constant alpha(mu) computed offline as the smallest
generalized eigenvalue of (a_h^sym(mu), X0) (rigorous to the eigensolver tolerance; a
successive-constraint lower bound alpha_SCM <= alpha is the fully-online alternative).

Certified output bounds (anchored norm, stability alpha):
    field    : |d_h - d_N|       <= ||ell||_0' ||r^pr||_0' / alpha
    goal-or. : |d_h - d_N^c|     <= ||r^pr||_0' ||r^du||_0' / alpha     (DWR-corrected)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import fem

MU0 = (0.1, 1.0, 1.0, 1000.0)   # anchor parameter (convective corner)


class Reductor:
    def __init__(self, ref):
        self.ref = ref
        r0 = fem.solve_truth(ref, MU0)
        bt0 = r0['eops']['bt']
        self.Dt = bt0.get_dofs('inlet').all()
        self.free = np.setdiff1d(np.arange(ref.N_t), self.Dt)
        A0 = r0['eops']['Ae'][self.free][:, self.free]
        self.X0 = (0.5 * (A0 + A0.T)).tocsc()       # anchored inner product
        self.X0lu = spla.splu(self.X0)

    def dnorm(self, r):
        """Dual norm ||r||_{X0'} (fixed, online-efficient)."""
        return np.sqrt(max(r @ self.X0lu.solve(r), 0.0))

    def alpha(self, Af):
        """Stability constant alpha(mu) = lambda_min(a_h^sym(mu), X0) > 0 (offline, rigorous)."""
        As = (0.5 * (Af + Af.T)).tocsc()
        try:
            lam = spla.eigsh(As, k=1, M=self.X0, sigma=0, which='LM',
                             return_eigenvectors=False)
            return float(abs(lam[0]))
        except Exception:
            return float('nan')


def build_cache(ref, red, params, verbose=False):
    """Per-parameter truth operators/outputs (free-dof restriction) for the RB study."""
    C = []
    f = red.free
    for i, mu in enumerate(params):
        r = fem.solve_truth(ref, mu)
        Ae = r['eops']['Ae']; fe = r['eops']['fe']; L = r['funcs']['LdT']
        Af = Ae[f][:, f].tocsc(); ff = fe[f]; lf = L[f]; Tf = r['T'][f]
        zf = spla.spsolve(Af.T.tocsc(), lf); d = float(lf @ Tf)
        C.append(dict(mu=mu, Af=Af, ff=ff, lf=lf, Tf=Tf, zf=zf, d=d,
                      Nu=2 * mu[2] / d, alpha=red.alpha(Af), lnorm=red.dnorm(lf),
                      Ppump=r['Ppump'], Dgeo=r['D_geo']))
        if verbose and (i + 1) % 20 == 0:
            print(f"    cached {i+1}/{len(params)}")
    return C


def xorth_append(V, vecs, red, tol=1e-10):
    """Append columns, X0-orthonormalized by modified Gram-Schmidt."""
    cols = [] if V is None else [V[:, j] for j in range(V.shape[1])]
    for w in vecs:
        w = w.copy()
        for q in cols:
            w = w - (q @ red.X0.dot(w)) * q
        nrm = np.sqrt(max(w @ red.X0.dot(w), 0.0))
        if nrm > tol:
            cols.append(w / nrm)
    return np.column_stack(cols)


def reduced_primal(Af, ff, V):
    return V @ np.linalg.solve(V.T @ (Af @ V), V.T @ ff)


def reduced_dual(Af, lf, W):
    return W @ np.linalg.solve(W.T @ (Af.T @ W), W.T @ lf)


def eval_point(c, V, W, red):
    """Errors and certified bounds at one cached parameter."""
    Af, ff, lf, al = c['Af'], c['ff'], c['lf'], c['alpha']
    Tn = reduced_primal(Af, ff, V); dN = float(lf @ Tn)
    rpr = ff - Af @ Tn; rpr_n = red.dnorm(rpr)
    out = dict(d=c['d'], dN=dN, err_uncorr=abs(c['d'] - dN),
               D_field=c['lnorm'] * rpr_n / al, rpr_n=rpr_n, alpha=al)
    if W is not None:
        zM = reduced_dual(Af, lf, W); rdu = lf - Af.T @ zM; rdu_n = red.dnorm(rdu)
        eta = float(rpr @ zM); dNc = dN + eta
        out.update(dNc=dNc, err_corr=abs(c['d'] - dNc), eta=eta,
                   D_go=rpr_n * rdu_n / al, rdu_n=rdu_n,
                   theta=(eta / (c['d'] - dN) if abs(c['d'] - dN) > 1e-12 else np.nan))
    return out


def greedy(cache_tr, cache_te, red, Nmax=26, mode='go'):
    """Output-adaptive weak greedy.  mode='go' (primal+dual, corrected output) or
    'field' (primal-only, uncorrected output).  Center initialization."""
    ctr = np.array([0.1, 1.15, 1.25, np.sqrt(10 * 1000)])
    def dist(mu):
        m = np.array(mu, float); m[3] = np.log10(m[3])
        cc = ctr.copy(); cc[3] = np.log10(cc[3])
        return np.sum(((m - cc) / np.array([0.2, 0.7, 1.5, 3.0]))**2)
    seed = int(np.argmin([dist(c['mu']) for c in cache_tr]))
    V = xorth_append(None, [cache_tr[seed]['Tf']], red)
    W = xorth_append(None, [cache_tr[seed]['zf']], red) if mode == 'go' else None
    hist = []
    for _ in range(Nmax):
        key = 'D_go' if mode == 'go' else 'D_field'
        ind = [eval_point(c, V, W if mode == 'go' else None, red)[key] for c in cache_tr]
        k = int(np.argmax(ind))
        errs, bnds, effs = [], [], []
        for c in cache_te:
            e = eval_point(c, V, W if mode == 'go' else None, red)
            if mode == 'go':
                errs.append(e['err_corr']); bnds.append(e['D_go'])
                if e['err_corr'] > 1e-14:
                    effs.append(e['D_go'] / e['err_corr'])
            else:
                errs.append(e['err_uncorr']); bnds.append(e['D_field'])
                if e['err_uncorr'] > 1e-14:
                    effs.append(e['D_field'] / e['err_uncorr'])
        hist.append(dict(N=V.shape[1], max_err=max(errs), mean_err=float(np.mean(errs)),
                         max_bnd=max(bnds),
                         med_eff=float(np.median(effs)) if effs else np.nan,
                         sel_mu=cache_tr[k]['mu']))
        V = xorth_append(V, [cache_tr[k]['Tf']], red)
        if mode == 'go':
            W = xorth_append(W, [cache_tr[k]['zf']], red)
        if ind[k] < 1e-13:
            break
    return hist, V, (W if mode == 'go' else None)
