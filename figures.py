"""Generate all figures from the saved results (and one representative truth solve)."""
import pickle
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fem

plt.rcParams.update({"font.family": "serif", "font.size": 10, "mathtext.fontset": "cm",
                     "axes.linewidth": 0.7, "savefig.bbox": "tight", "savefig.dpi": 200,
                     "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linewidth": 0.5})


def _save(fig, figdir, name):
    fig.savefig(f"{figdir}/{name}.pdf"); fig.savefig(f"{figdir}/{name}.png"); plt.close(fig)


def make_all(ref, resdir, figdir):
    _fields(ref, figdir)
    _greedy(resdir, figdir)
    _effectivity(resdir, figdir)
    _rigor(resdir, figdir)
    _flow(resdir, figdir)
    _timing(resdir, figdir)
    _pareto(resdir, figdir)


def _fields(ref, figdir):
    mu = (0.15, 1.0, 1.0, 200.0)
    r = fem.solve_truth(ref, mu); md = r["stk"]["md"]; bt = r["eops"]["bt"]
    verts = bt.nodal_dofs[0]; x, y, tri = md.p[0], md.p[1], md.t.T
    bu = r["stk"]["bu"]; vu = r["stk"]["vu"]
    speed = np.sqrt(vu[bu.nodal_dofs[0]]**2 + vu[bu.nodal_dofs[1]]**2)
    free = np.setdiff1d(np.arange(ref.N_t), ref.bt.get_dofs("inlet").all())
    Af = r["eops"]["Ae"][free][:, free].tocsc(); lf = r["funcs"]["LdT"][free]
    z = np.zeros(ref.N_t); z[free] = spla.spsolve(Af.T.tocsc(), lf)
    fig, axs = plt.subplots(3, 1, figsize=(5.4, 6.0))
    for ax, val, ttl, cm in [(axs[0], speed, "(a) Stokes velocity magnitude $|\\beta|$", "viridis"),
                             (axs[1], r["T"][verts], "(b) primal temperature $T(\\mu)$", "inferno"),
                             (axs[2], z[verts], "(c) adjoint $\\psi(\\mu)$ for the thermal output $d$", "coolwarm")]:
        tcf = ax.tricontourf(x, y, tri, val, levels=24, cmap=cm)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=9, loc="left"); plt.colorbar(tcf, ax=ax, shrink=0.85, pad=0.01)
    fig.tight_layout(h_pad=0.6); _save(fig, figdir, "fig_fields")


def _greedy(resdir, figdir):
    H = pickle.load(open(f"{resdir}/greedy_hist.pkl", "rb")); go, fd = H["hist_go"], H["hist_fd"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogy([h["N"] for h in fd], [h["max_err"] for h in fd], "s--", color="tab:orange",
                ms=4, label="field-greedy, uncorrected output")
    ax.semilogy([h["N"] for h in go], [h["max_err"] for h in go], "o-", color="tab:blue",
                ms=4, label="goal-oriented greedy, DWR-corrected")
    ax.semilogy([h["N"] for h in go], [h["max_bnd"] for h in go], ":", color="tab:blue",
                lw=1.3, label="certified bound $\\Delta^{d}_N$ (anchored norm)")
    ax.set_xlabel("reduced dimension $N$"); ax.set_ylabel("max. thermal-output error $|d-d_N|$")
    ax.set_title("Goal-oriented vs. field-based greedy (thermal output $d$)", fontsize=9.5)
    ax.legend(); _save(fig, figdir, "fig_greedy")


def _effectivity(resdir, figdir):
    E = pickle.load(open(f"{resdir}/effectivity.pkl", "rb")); Ns = sorted(E)
    thq = np.array([[np.percentile(E[N]["theta"], p) for p in (10, 50, 90)] for N in Ns])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axs[0].fill_between(Ns, thq[:, 0], thq[:, 2], alpha=0.25, color="tab:blue", label="10-90th pct")
    axs[0].plot(Ns, thq[:, 1], "o-", color="tab:blue", ms=4, label="median")
    axs[0].axhline(1.0, color="k", lw=0.8, ls="--"); axs[0].set_ylim(0, 2.2)
    axs[0].set_xlabel("reduced dimension $N$"); axs[0].set_ylabel("DWR estimate effectivity")
    axs[0].set_title("(a) sharpness of the DWR estimate", fontsize=9); axs[0].legend()
    axs[1].plot(Ns, [E[N]["bound_eff"] for N in Ns], "o-", color="tab:blue", ms=4)
    axs[1].set_xlabel("reduced dimension $N$"); axs[1].set_ylabel("certified-bound effectivity")
    axs[1].set_title("(b) rigorous bound effectivity (anchored norm)", fontsize=9)
    fig.tight_layout(); _save(fig, figdir, "fig_effectivity")


def _rigor(resdir, figdir):
    E = pickle.load(open(f"{resdir}/effectivity.pkl", "rb"))
    N0 = sorted(E)[len(E) // 2]
    err = np.array(E[N0]["err_co"]); bnd = np.array(E[N0]["Dgo"])
    o = np.argsort(err); pc = np.linspace(0, 100, len(err))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogy(pc, bnd[o], "-", color="tab:blue", lw=1.5, label="certified bound $\\Delta^{d}_N$")
    ax.semilogy(pc, err[o], "o", color="tab:blue", ms=3, label="true corrected error $|d-d_N^{\\,c}|$")
    ax.set_xlabel("test-parameter percentile"); ax.set_ylabel("thermal output error / bound")
    ax.set_title(f"Rigor of the certified bound ($N={N0}$)", fontsize=9.5); ax.legend()
    _save(fig, figdir, "fig_rigor")


def _flow(resdir, figdir):
    F = pickle.load(open(f"{resdir}/flow.pkl", "rb")); rows = F["rows"]; sv = F["sv"]
    Nf = [r["Nf"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.semilogy(Nf, [r["rel_err_med"] for r in rows], "o-", color="tab:green", ms=4, label="median rel. error")
    ax.semilogy(Nf, [r["rel_err_max"] for r in rows], "^--", color="tab:green", ms=4, alpha=0.6, label="max rel. error")
    ax2 = ax.twinx(); ax2.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], ".", color="0.5", ms=4)
    ax2.set_ylabel("norm. POD energy", color="0.4"); ax2.grid(False)
    ax.set_xlabel("flow reduced dimension $N_f$"); ax.set_ylabel("pumping-power rel. error")
    ax.set_title("Reduced Stokes: pumping-power certification", fontsize=9.5)
    ax.legend(loc="upper right"); _save(fig, figdir, "fig_flow")


def _timing(resdir, figdir):
    T = pickle.load(open(f"{resdir}/timing.pkl", "rb")); tr = np.array(T["truth"]); to = T["t_online"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axs[0].loglog(tr[:, 0], tr[:, 2], "o-", color="tab:red", ms=5, label="truth solve")
    axs[0].loglog(tr[:, 0], [to] * len(tr), "s--", color="tab:blue", ms=5, label="certified online")
    axs[0].set_xlabel("truth thermal dofs $\\mathcal{N}_h$"); axs[0].set_ylabel("wall time/query [ms]")
    axs[0].set_title("(a) online cost is $\\mathcal{N}_h$-independent", fontsize=9); axs[0].legend()
    axs[1].semilogx(tr[:, 0], tr[:, 2] / to, "o-", color="k", ms=5)
    axs[1].set_xlabel("truth thermal dofs $\\mathcal{N}_h$"); axs[1].set_ylabel("speed-up")
    axs[1].set_title("(b) speed-up vs. truth size", fontsize=9)
    fig.tight_layout(); _save(fig, figdir, "fig_timing")


def _pareto(resdir, figdir):
    PA = pickle.load(open(f"{resdir}/pareto.pkl", "rb"))
    Nu, Pp, mu, front = PA["Nu"], PA["Ppump"], PA["mu"], PA["front"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sc = ax.scatter(Pp, Nu, c=np.log10(mu[:, 3]), cmap="viridis", s=18, alpha=0.8)
    fo = front[np.argsort(Pp[front])]
    ax.plot(Pp[fo], Nu[fo], "r-o", ms=4, lw=1.3, label="nominal Pareto frontier")
    ax.set_xscale("log"); ax.set_xlabel("pumping power $\\mathcal{P}(\\mu)$")
    ax.set_ylabel("Nusselt-like number $Nu_{\\mathrm{ref}}(\\mu)$")
    plt.colorbar(sc, ax=ax).set_label("$\\log_{10} Pe$")
    ax.set_title("Certified $Nu_{\\mathrm{ref}}$ vs. pumping-power trade-off", fontsize=9.5)
    ax.legend(loc="lower right"); _save(fig, figdir, "fig_pareto")
