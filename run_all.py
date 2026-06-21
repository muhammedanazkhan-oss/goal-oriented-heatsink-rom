#!/usr/bin/env python3
"""Reproduce the numerical study of the goal-oriented certified ROM for a
forced-convection wavy heat sink.

Usage:
    python run_all.py            # FULL configuration (reproduces the manuscript; ~minutes)
    python run_all.py --quick    # fast smoke test (~1 minute)
    python run_all.py --no-figures   # skip figure generation

Outputs are written to  results/  (CSV + .pkl/.npy) and  figures/  (PDF + PNG).
"""
import os
import sys
import time
import pickle
import csv
import numpy as np

import fem
import params as P
import reduction as R
import flow as FL
import timing as TM
import verification as VER

RESDIR = "results"
FIGDIR = "figures"


def _save_csv(path, rows, header):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for r in rows:
            w.writerow(r)


def reproduce():
    """Full numerical study: writes results/ and figures/.  Intended for the public
    deposit at publication time (NOT run by default, to keep results confidential)."""
    from config import FULL, QUICK
    cfg = QUICK if "--quick" in sys.argv else FULL
    make_figs = "--no-figures" not in sys.argv
    os.makedirs(RESDIR, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
    t_start = time.time()
    print(f"=== configuration: {cfg['name']}  (truth mesh n={cfg['n']}) ===")

    ref = fem.RefMesh(n=cfg["n"])
    red = R.Reductor(ref)
    print(f"[reductor] free thermal dofs Nf = {len(red.free)}")

    # ---- 1. parameter samples + truth cache ----
    train = P.sample_params(cfg["n_train"], cfg["seed_train"])
    test = P.sample_params(cfg["n_test"], cfg["seed_test"])
    print(f"[cache] building truth cache: {len(train)} train + {len(test)} test ...")
    Ctr = R.build_cache(ref, red, train, verbose=True)
    Cte = R.build_cache(ref, red, test, verbose=True)

    # ---- 2. goal-oriented vs field greedy ----
    print("[greedy] goal-oriented and field-based ...")
    hist_go, Vgo, Wgo = R.greedy(Ctr, Cte, red, Nmax=cfg["Nmax"], mode="go")
    hist_fd, Vfd, _ = R.greedy(Ctr, Cte, red, Nmax=cfg["Nmax"], mode="field")
    np.save(f"{RESDIR}/Vgo.npy", Vgo); np.save(f"{RESDIR}/Wgo.npy", Wgo)
    np.save(f"{RESDIR}/Vfd.npy", Vfd)
    pickle.dump(dict(hist_go=hist_go, hist_fd=hist_fd), open(f"{RESDIR}/greedy_hist.pkl", "wb"))
    _save_csv(f"{RESDIR}/greedy.csv",
              [(a["N"], a["max_err"], a["max_bnd"], b["max_err"], b["max_bnd"])
               for a, b in zip(hist_go, hist_fd)],
              ["N", "GO_corrected_max_err", "GO_bound", "field_uncorr_max_err", "field_bound"])
    print(f"        N={hist_go[-1]['N']}: GO max_err={hist_go[-1]['max_err']:.2e}  "
          f"field max_err={hist_fd[-1]['max_err']:.2e}")

    # ---- 3. effectivity + rigor ----
    print("[effectivity] DWR estimate effectivity and certified-bound rigor ...")
    eff = {}
    for N in cfg["eff_Ns"]:
        th, ec, dgo = [], [], []
        rig = True
        for c in Cte:
            e = R.eval_point(c, Vgo[:, :N], Wgo[:, :N], red)
            ec.append(e["err_corr"]); dgo.append(e["D_go"])
            if not np.isnan(e["theta"]):
                th.append(e["theta"])
            if e["D_go"] < e["err_corr"] - 1e-13:
                rig = False
        th = np.array(th); ec = np.array(ec); dgo = np.array(dgo)
        msk = ec > 1e-13
        eff[N] = dict(theta=th, err_co=ec, Dgo=dgo,
                      theta_signed=float(np.median(th)),
                      theta_abs=float(np.median(np.abs(th))),
                      bound_eff=float(np.median(dgo[msk] / ec[msk])),
                      rigorous=bool(rig),
                      min_alpha=float(min(c["alpha"] for c in Cte)))
    pickle.dump(eff, open(f"{RESDIR}/effectivity.pkl", "wb"))
    _save_csv(f"{RESDIR}/effectivity.csv",
              [(N, eff[N]["theta_signed"], eff[N]["theta_abs"], eff[N]["bound_eff"],
                eff[N]["rigorous"], eff[N]["min_alpha"]) for N in cfg["eff_Ns"]],
              ["N", "DWR_eff_signed", "DWR_eff_abs", "bound_eff", "rigorous", "min_alpha"])

    # ---- 4. Pareto data (from the cache) ----
    allC = Ctr + Cte
    mu = np.array([c["mu"] for c in allC]); Nu = np.array([c["Nu"] for c in allC])
    Pp = np.array([c["Ppump"] for c in allC]); Dg = np.array([c["Dgeo"] for c in allC])
    order = np.argsort(Pp); front = []; best = -np.inf
    for i in order:
        if Nu[i] > best:
            front.append(i); best = Nu[i]
    pickle.dump(dict(mu=mu, Nu=Nu, Ppump=Pp, Dgeo=Dg, front=np.array(front)),
                open(f"{RESDIR}/pareto.pkl", "wb"))

    # ---- 5. reduced Stokes: pumping-power convergence + certificate ----
    print("[flow] supremizer-stabilized reduced Stokes ...")
    flow = FL.FlowROM(ref, P.sample_geometries(cfg["flow_train"], cfg["flow_seed"]))
    test_geoms = P.sample_geometries(cfg["flow_test"], cfg["flow_seed"] + 1)
    prows = FL.pumping_study(ref, flow, test_geoms, 1.0, cfg["flow_Nf"])
    pickle.dump(dict(rows=prows, sv=flow.sv), open(f"{RESDIR}/flow.pkl", "wb"))
    _save_csv(f"{RESDIR}/pumping.csv",
              [(r["Nf"], r["rel_err_med"], r["rel_err_max"], r["cert_rel_med"], r["cert_eff_med"])
               for r in prows],
              ["Nf", "rel_err_med", "rel_err_max", "cert_rel_bound_med", "cert_effectivity_med"])

    # ---- 6. affine reduced-flow defects (fully-affine vs truth-operator bound) ----
    print("[defects] reduced-flow defects eps_beta, eps_ell; affine vs truth bound ...")
    drows = FL.defect_study(ref, red, flow, Cte, Vgo, Wgo, cfg["defect_N"], cfg["defect_Nf"])
    dr = {k: np.array([row[k] for row in drows]) for k in drows[0]}
    pickle.dump(drows, open(f"{RESDIR}/defects.pkl", "wb"))
    _save_csv(f"{RESDIR}/defects.csv",
              [(cfg["defect_Nf"], float(np.median(dr["eps_beta_uN"])),
                float(np.median(dr["eps_ell"])), float(np.median(dr["dual_defect"])),
                float(np.median(dr["ratio"])), float(np.max(dr["ratio"])))],
              ["Nf", "eps_beta_uN_med", "eps_ell_med", "dual_defect_med",
               "Daff_over_Dtruth_med", "Daff_over_Dtruth_max"])

    # ---- 7. online cost vs truth dimension ----
    print("[timing] truth vs affine online query ...")
    truth = TM.truth_times(cfg["timing_ns"])
    t_on = TM.online_query_time(Ctr, Vgo[:, :min(20, Vgo.shape[1])],
                                Wgo[:, :min(20, Wgo.shape[1])])
    pickle.dump(dict(truth=truth, t_online=t_on), open(f"{RESDIR}/timing.pkl", "wb"))
    _save_csv(f"{RESDIR}/timing.csv",
              [(nt, nu, tt, t_on, tt / t_on) for nt, nu, tt in truth],
              ["thermal_dofs", "velocity_dofs", "truth_ms", "online_ms", "speedup"])

    # ---- 8. verification (mesh, heat balance, backflow, tau, divergence, inf-sup) ----
    print("[verification] mesh convergence, heat balance, backflow, tau, inf-sup ...")
    ver = dict(
        straight=VER.straight_dissipation(n=min(40, max(16, cfg["n"]))),
        mesh_corner=VER.mesh_convergence_corners(cfg["corners"],
                    ns=(cfg["n"], cfg["n"] + 8, cfg["n"] + 16)),
        balance=[VER.heat_balance_and_backflow(m, n=min(48, cfg["n"] + 16)) for m in cfg["corners"]],
        tau=VER.tau_sensitivity((0.15, 0.9, 1.5, 800.0), n=cfg["n"]),
        divergence=VER.divergence_measure([(0.0, 1.0, 1.0), (0.1, 1.0, 1.0),
                                           (0.2, 1.5, 2.0), (0.2, 1.5, 0.5)], n=cfg["n"]),
        infsup=VER.stokes_infsup([(0.0, 1.0, 1.0), (0.1, 1.0, 1.0),
                                  (0.2, 1.5, 2.0), (0.2, 1.5, 0.5)], n=min(24, cfg["n"])),
    )
    pickle.dump(ver, open(f"{RESDIR}/verification.pkl", "wb"))
    _save_csv(f"{RESDIR}/verification_corners.csv",
              [(b["mu"], b["d"], b["Nu"], b["Dgeo"], b["Ppump"], b["deficit"], b["min_bn"])
               for b in ver["balance"]],
              ["mu", "d", "Nu_ref", "Dgeo", "Ppump", "heat_recovery_deficit", "min_outlet_vel"])

    # ---- 9. figures ----
    if make_figs:
        print("[figures] generating PDFs/PNGs ...")
        import figures
        figures.make_all(ref, RESDIR, FIGDIR)

    # ---- summary ----
    print("\n=== SUMMARY ===")
    print(f"straight-channel dissipation (->12): {ver['straight'][1.0]:.4f}")
    print(f"greedy:   GO max_err={hist_go[-1]['max_err']:.2e}  field={hist_fd[-1]['max_err']:.2e}")
    print(f"DWR effectivity (signed) at largest N: {eff[cfg['eff_Ns'][-1]]['theta_signed']:.3f}")
    print(f"bound rigorous at all N/test pts: "
          f"{all(eff[N]['rigorous'] for N in cfg['eff_Ns'])}; min alpha={eff[cfg['eff_Ns'][0]]['min_alpha']:.3f}")
    print(f"pumping rel.err (largest Nf): {prows[-1]['rel_err_med']:.2e}")
    print(f"online query: {t_on:.4f} ms; speedups: "
          f"{[round(tt/t_on) for _,_,tt in truth]}")
    print(f"affine/truth bound ratio (median): {float(np.median(dr['ratio'])):.2f}")
    print(f"gamma_h (Taylor-Hood inf-sup): {[round(v,3) for _,v in ver['infsup']]}")
    print(f"\nresults/ and figures/ written.  total wall time {time.time()-t_start:.1f}s")


def selftest():
    """Lightweight sanity check of the implementation. Writes NOTHING.
    Confirms the solver/certificate are wired correctly without producing any
    of the manuscript results or figures."""
    import numpy as np
    import fem, params as P, reduction as R
    print("=== self-test (no results/figures written) ===")
    ref = fem.RefMesh(n=12)
    Dg = fem.solve_stokes(ref, 0.0, 1.0, 1.0)["D_geo"]
    assert abs(Dg - 12.0) < 1e-3, Dg
    print(f"  straight-channel dissipation = {Dg:.4f} (expected 12)  OK")
    r = fem.solve_truth(ref, (0.1, 1.0, 1.0, 100.0))
    assert r["d"] > 0 and np.isfinite(r["Nu"])
    print(f"  truth solve: d={r['d']:.4f}, Nu_ref={r['Nu']:.2f}  OK")
    red = R.Reductor(ref)
    C = R.build_cache(ref, red, P.sample_params(4, seed=0))
    V = R.xorth_append(None, [C[0]["Tf"]], red)
    W = R.xorth_append(None, [C[0]["zf"]], red)
    e = R.eval_point(C[1], V, W, red)
    assert e["D_go"] >= e["err_corr"] - 1e-12
    print(f"  certified bound >= true error: {e['D_go']:.2e} >= {e['err_corr']:.2e}  OK")
    print("self-test PASSED.")
    print("\nThe full numerical study and figures are withheld at this stage.")
    print("To regenerate them (for the public deposit at publication), run:")
    print("    python run_all.py --reproduce            # full manuscript configuration")
    print("    python run_all.py --reproduce --quick    # fast coarse-mesh check")


if __name__ == "__main__":
    import sys
    if "--reproduce" in sys.argv:
        reproduce()
    else:
        selftest()
