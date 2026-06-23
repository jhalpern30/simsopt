"""
Single-iota true epsilon-constraint optimization with adaptive resolution and
dipole sparsity (remove outboard-midplane dipoles).

This follows the same objective, constraints, plotting, output schema, and gtol
logic as single_stage_true_epsilon_adaptive_res.py, but it runs a single iota /
fcp case and applies a sparse dipole set before optimization.
"""

import os
import sys
import json
import argparse
import warnings
import time
import numpy as np
from datetime import datetime
from scipy.optimize import minimize as scipy_minimize, OptimizeResult

from simsopt.geo import SurfaceRZFourier
from simsopt.geo.surfaceobjectives import BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.field import BiotSavart, coils_to_vtk, CurrentPenalty
from simsopt.objectives import QuadraticPenalty
from simsopt._core.optimizable import load, save
import matplotlib.pyplot as plt
from helper_functions import *
from boozer_functions import *


def coil_center_theta(coil, major_radius):
    gamma = coil.curve.gamma()
    center = np.mean(gamma, axis=0)
    x0, y0, z0 = center
    r0 = np.sqrt(x0 * x0 + y0 * y0)
    return np.mod(np.arctan2(z0, r0 - major_radius), 2 * np.pi)


def angular_distance_to_zero(theta):
    return np.minimum(theta, 2 * np.pi - theta)


# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(
    description=(
        "Single-iota true epsilon-constraint run with adaptive resolution and "
        "sparse dipoles (remove outboard coils)."
    )
)
parser.add_argument("--init-dir", type=str, required=True,
    help="Directory with bs_opt.json, results.json, and surf_opt.json used to start the run.")
parser.add_argument("--iota-target", type=float, required=True,
    help="Single iota target for this run.")
parser.add_argument("--f-cp-threshold", type=float, required=True,
    help="Current p-norm upper bound [A].")
parser.add_argument("--resolutions", type=str, required=True,
    help="Comma-separated list of mpol=ntor values (e.g. '6,9,12').")
parser.add_argument("--fb-thresholds", type=str, required=True,
    help="Comma-separated list of Boozer residual thresholds, one per resolution.")
parser.add_argument("--iota-threshold", type=float, default=0.0025,
    help="Absolute iota tolerance (default: 0.0025).")
parser.add_argument("--maxiter", type=int, default=200,
    help="Max BFGS iterations per resolution step (default: 200).")
parser.add_argument("--volume-target", type=float, default=0.3,
    help="Target volume of the Boozer surface (default: 0.3).")
parser.add_argument("--theta-tol", type=float, default=0.01,
    help="Outboard removal tolerance in theta around 0 (default: 0.01 rad).")
parser.add_argument("--output-root", type=str,
    default="../single_stage_true_epsilon_adaptive_res_sparse",
    help="Top-level output directory; run outputs are placed under "
         "<output-root>/<eq_name>/iota<X>_fcp<Y>kA_vt<Z>/mpol<M>_ntor<N>/.")
parser.add_argument(
    "--new",
    action="store_true",
    default=False,
    help=(
        "Start from scratch: do not skip resolution steps that already have a "
        "converged results.json."
    ),
)
args, _ = parser.parse_known_args()

try:
    resolutions = [int(r.strip()) for r in args.resolutions.split(",")]
except ValueError as exc:
    raise ValueError(f"--resolutions must be a comma-separated list of integers: {exc}") from exc

try:
    fb_thresholds = [float(t.strip()) for t in args.fb_thresholds.split(",")]
except ValueError as exc:
    raise ValueError(f"--fb-thresholds must be a comma-separated list of floats: {exc}") from exc

if len(resolutions) != len(fb_thresholds):
    raise ValueError(
        f"--resolutions and --fb-thresholds must have the same number of entries; "
        f"got {len(resolutions)} and {len(fb_thresholds)}."
    )
if len(resolutions) == 0:
    raise ValueError("--resolutions must contain at least one entry.")

INIT_DIR = args.init_dir
IOTA_TARGET = args.iota_target
FCP_THRESHOLD = args.f_cp_threshold
IOTA_THRESHOLD = args.iota_threshold
MAXITER = args.maxiter
VOL_TARGET = args.volume_target
THETA_TOL = args.theta_tol
OUTPUT_ROOT = args.output_root
START_FRESH = args.new

BOOZER_CW = 1.0
PENALTY_WEIGHT = 100.0
CURRENT_P_NORM = 20.0
GTOL = 1e-3
FB_WARN_MARGIN = 0.05

plot_config = PlotConfig(
    dpi=100, titlefontsize=16, axisfontsize=16,
    legendfontsize=14, ticklabelfontsize=14, cbarfontsize=16,
)


def plot_objective_vs_iterations(out_dir, config):
    history_path = os.path.join(out_dir, "iterations.json")
    if not os.path.isfile(history_path):
        return
    with open(history_path, "r") as f:
        history = json.load(f)
    rows = history.get("iterations") or []
    if len(rows) < 1:
        return

    it = np.array([r["iteration"] for r in rows], dtype=float)
    J_tot = np.array([r["J"] for r in rows], dtype=float)
    grad_norm = np.array([r["grad_norm"] for r in rows], dtype=float)

    contrib_keys = ("J_boozer_contrib", "J_iota_contrib", "J_current_contrib")

    def _row_has_contribs(r):
        return all(k in r for k in contrib_keys)

    any_components = any(_row_has_contribs(r) for r in rows)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), constrained_layout=True,
    )
    if any_components:
        J_qs = np.array([r["nonQS_ratio"] for r in rows], dtype=float)
        J_b = np.array(
            [float(r["J_boozer_contrib"]) if "J_boozer_contrib" in r else np.nan for r in rows],
            dtype=float,
        )
        J_i = np.array(
            [float(r["J_iota_contrib"]) if "J_iota_contrib" in r else np.nan for r in rows],
            dtype=float,
        )
        J_c = np.array(
            [float(r["J_current_contrib"]) if "J_current_contrib" in r else np.nan for r in rows],
            dtype=float,
        )
        ax1.plot(it, J_tot, color="k", lw=2.0, label=r"$J$ (total)")
        ax1.plot(it, J_qs, label=r"$J_{\mathrm{nonQS}}$")
        ax1.plot(it, J_b, label=r"$w_{\mathrm{pen}}\,J_{\mathrm{Boozer\,guard}}$")
        ax1.plot(it, J_i, label=r"$w_{\mathrm{pen}}\,J_{\iota}$")
        ax1.plot(it, J_c, label=r"$w_{\mathrm{pen}}\,J_{\mathrm{current\,guard}}$")
    else:
        ax1.plot(it, J_tot, color="k", lw=2.0, label=r"$J$ (total)")

    ax1.set_ylabel("Contribution to $J$", fontsize=config.axisfontsize)
    ax1.set_title("Objective and components vs iteration", fontsize=config.titlefontsize)
    ax1.legend(fontsize=config.legendfontsize, loc="best")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.tick_params(labelsize=config.ticklabelfontsize)
    ax1.set_yscale("symlog", linthresh=1e-4)

    ax2.plot(it, grad_norm, color="C5", lw=1.5)
    ax2.set_ylabel(r"$\|\nabla J\|_2$", fontsize=config.axisfontsize)
    ax2.set_xlabel("Iteration (accepted)", fontsize=config.axisfontsize)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.tick_params(labelsize=config.ticklabelfontsize)
    ax2.set_yscale("symlog", linthresh=1e-3)

    out_path = os.path.join(out_dir, "objective_vs_iteration.png")
    fig.savefig(out_path, dpi=config.dpi)
    plt.close(fig)
    print(f"Saved objective history plot to {out_path}")


init_results = load(os.path.join(INIT_DIR, "results.json"))
eq_name = init_results["eq_name"]

EQ_OUT_ROOT = os.path.join(OUTPUT_ROOT, eq_name)
os.makedirs(EQ_OUT_ROOT, exist_ok=True)


def _per_res_out_dir(mpol, ntor):
    parent = os.path.join(
        EQ_OUT_ROOT,
        f"iota{IOTA_TARGET:g}_fcp{FCP_THRESHOLD / 1e3:g}kA_vt{VOL_TARGET:g}",
    )
    return os.path.join(parent, f"mpol{mpol}_ntor{ntor}")


def _is_completed(out_dir):
    rpath = os.path.join(out_dir, "results.json")
    if not os.path.isfile(rpath):
        return False
    try:
        rj = load(rpath)
    except Exception:
        return False
    return bool(rj.get("optimization_success", False))


def _has_checkpoint(out_dir):
    needed = ["bs_opt.json", "surf_opt.json", "results.json"]
    return all(os.path.isfile(os.path.join(out_dir, f)) for f in needed)


def _next_iteration_index_from_history(out_dir):
    path = os.path.join(out_dir, "iterations.json")
    if not os.path.isfile(path):
        return 1
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 1
    iters = data.get("iterations")
    if not isinstance(iters, list) or not iters:
        return 1
    last = iters[-1]
    if not isinstance(last, dict) or "iteration" not in last:
        return 1
    return int(last["iteration"]) + 1


print(f"Adaptive-resolution sparse true epsilon run: {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  init-dir:    {INIT_DIR}")
print(f"  output root: {EQ_OUT_ROOT}")
print(f"  iota target: {IOTA_TARGET:g}")
print(f"  f_CP:        {FCP_THRESHOLD:.0f} A")
print(f"  vt:          {VOL_TARGET:g}")
print(f"  theta_tol:   {THETA_TOL:g}")
print(f"  resolution ladder:")
for res, fbt in zip(resolutions, fb_thresholds):
    print(f"    mpol=ntor={res:2d}  fb_threshold={fbt:.2e}")
print(f"  penalty:     {PENALTY_WEIGHT}, gtol={GTOL} (x0.1 at highest res), maxiter/step={MAXITER}")

bs = load(os.path.join(INIT_DIR, "bs_opt.json"))
surf_opt_path = os.path.join(INIT_DIR, "surf_opt.json")
if os.path.exists(surf_opt_path):
    surf = load(surf_opt_path)
    nPhi = surf.quadpoints_phi.size
    nTheta = surf.quadpoints_theta.size
else:
    nPhi, nTheta = 128, 64
    eq_path = os.path.join(init_results["eq_dir"], init_results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_path, s=init_results["surf_s"], range="half period", nphi=nPhi, ntheta=nTheta,
    )
    surf.set_dofs(init_results["surf_dof_scale"] * surf.get_dofs())

shared_meta = {
    "eq_name": init_results["eq_name"],
    "eq_dir": init_results["eq_dir"],
    "surf_nfp": init_results["surf_nfp"],
    "surf_s": init_results["surf_s"],
    "VV_R0": init_results["VV_R0"],
    "VV_a": init_results["VV_a"],
    "VV_b": init_results["VV_b"],
    "ntf": init_results["ntf"],
}
if "surf_dof_scale" in init_results:
    shared_meta["surf_dof_scale"] = init_results["surf_dof_scale"]

VV = SurfaceRZFourier(nfp=init_results["surf_nfp"])
VV.set_rc(0, 0, init_results["VV_R0"])
VV.set_rc(1, 0, init_results["VV_a"])
VV.set_zs(1, 0, init_results["VV_b"])

num_tf = init_results["ntf"] * 2 * init_results["surf_nfp"]
coils_dense = bs.coils
tf_coils = coils_dense[:num_tf]
dipole_coils_dense = coils_dense[num_tf:]

removed_dipoles = []
dipole_coils = []
for idx, coil in enumerate(dipole_coils_dense):
    theta = coil_center_theta(coil, init_results["VV_R0"])
    if angular_distance_to_zero(theta) <= THETA_TOL:
        removed_dipoles.append((idx, theta))
    else:
        dipole_coils.append(coil)
if len(dipole_coils) == 0:
    raise RuntimeError("All dipole coils were removed; reduce --theta-tol.")

coils = tf_coils + dipole_coils
bs = BiotSavart(coils)
for c in dipole_coils:
    c.curve.fix_all()
for c in tf_coils:
    c.current.fix_all()

print(f"  dipoles: kept {len(dipole_coils)} / {len(dipole_coils_dense)}, removed {len(removed_dipoles)}")


def optimize_one_resolution(prev_load_dir, mpol, ntor, fb_threshold, gtol=None, resume_this_step=False):
    if gtol is None:
        gtol = GTOL
    out_dir = _per_res_out_dir(mpol, ntor)
    os.makedirs(out_dir, exist_ok=True)

    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
    boozer_surface = initialize_boozer_surface(
        surf, mpol, ntor, bs, VOL_TARGET, BOOZER_CW, IOTA_TARGET, G0,
    )

    bs_obj = BiotSavart(coils)
    JnonQSRatio = sum([NonQuasiSymmetricRatio(boozer_surface, bs_obj)])
    JBoozerResidual = sum([BoozerResidual(boozer_surface, bs_obj)])
    JBoozerGuard = (1.0 / fb_threshold**2) * QuadraticPenalty(
        JBoozerResidual, fb_threshold, f="max",
    )
    iota = Iotas(boozer_surface)
    Jiota = (1.0 / IOTA_THRESHOLD**2) * QuadraticPenalty(iota, IOTA_TARGET)
    Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=CURRENT_P_NORM)
    JCurrentGuard = (1.0 / FCP_THRESHOLD**2) * QuadraticPenalty(
        Jcurrent, FCP_THRESHOLD, f="max",
    )
    JF = JnonQSRatio + PENALTY_WEIGHT * (JBoozerGuard + Jiota + JCurrentGuard)

    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    log_path = os.path.join(out_dir, "log.txt")
    log_mode = "a" if resume_this_step else "w"
    log_file = open(log_path, log_mode, buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    start_time = time.time()
    print(f"\n{'=' * 70}")
    print(f"Adaptive-resolution sparse step -- {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}")
    print(f"Output:     {out_dir}")
    print(f"warm-start: {prev_load_dir}")
    print(f"iota_target={IOTA_TARGET:g},  f_b={fb_threshold:.1e},  f_CP={FCP_THRESHOLD:.0f} A")

    if resume_this_step:
        next_it = _next_iteration_index_from_history(out_dir)
        n_done = max(0, next_it - 1)
    else:
        next_it = 1
        n_done = 0
    maxiter_scipy = max(0, MAXITER - n_done)

    print(f"mpol={mpol}, ntor={ntor}, maxiter_cap={MAXITER}, gtol={gtol:.2e}")
    if resume_this_step:
        print(
            f"  resume: {n_done} accepted iteration(s) on disk -> "
            f"up to {maxiter_scipy} further BFGS iteration(s) (cap {MAXITER})."
        )
    else:
        print(f"  fresh step: up to {maxiter_scipy} BFGS iteration(s) (cap {MAXITER}).")
    print(f"Resolution ladder: {resolutions}  fb_thresholds: {fb_thresholds}")
    print(f"Dipole sparsity: kept={len(dipole_coils)} removed={len(removed_dipoles)} theta_tol={THETA_TOL:g}")
    if resume_this_step:
        print(f"Resuming in-place from existing checkpoint at callback it={next_it}.")

    x0 = JF.x.copy()
    run_dict = {
        "sdofs": boozer_surface.surface.x.copy(),
        "iota": boozer_surface.res["iota"],
        "G": boozer_surface.res["G"],
        "J": JF.J(),
        "dJ": JF.dJ().copy(),
        "it": next_it,
        "lscount": 0,
        "x_prev": x0.copy(),
        "failed_boozer_solves": 0,
    }

    def fun(x):
        dx = np.linalg.norm(x - run_dict["x_prev"])
        run_dict["x_prev"] = x.copy()
        run_dict["lscount"] += 1

        boozer_surface.surface.x = run_dict["sdofs"]
        boozer_surface.res["iota"] = run_dict["iota"]
        boozer_surface.res["G"] = run_dict["G"]

        JF.x = x
        boozer_surface.run_code(run_dict["iota"], run_dict["G"])

        try:
            ok = boozer_surface.res["success"] and not boozer_surface.surface.is_self_intersecting()
        except Exception:
            ok = False

        if ok:
            run_dict["failed_boozer_solves"] = 0
            J, dJ = JF.J(), JF.dJ()
        else:
            run_dict["failed_boozer_solves"] += 1
            J, dJ = run_dict["J"], -run_dict["dJ"]
            boozer_surface.surface.x = run_dict["sdofs"]
            boozer_surface.res["iota"] = run_dict["iota"]
            boozer_surface.res["G"] = run_dict["G"]

        max_I = float(np.max(np.abs([c.current.get_value() for c in dipole_coils])))
        print(
            f"  step={dx:.2e}  J={J:.6e}  ||grad J||={np.linalg.norm(dJ):.6e}  "
            f"QS={JnonQSRatio.J():.6e}  fb={JBoozerResidual.J():.6e}  "
            f"iota={iota.J():.4f}  Imax={max_I:.0f}A"
        )
        return J, dJ

    def callback(x):
        run_dict["lscount"] = 0
        run_dict["sdofs"] = boozer_surface.surface.x.copy()
        run_dict["iota"] = boozer_surface.res["iota"]
        run_dict["G"] = boozer_surface.res["G"]
        run_dict["J"] = JF.J()
        run_dict["dJ"] = JF.dJ().copy()

        J = run_dict["J"]
        grad = run_dict["dJ"]
        qs_val = float(JnonQSRatio.J())
        fb_val = float(JBoozerResidual.J())
        iota_val = float(iota.J())
        cp_val = float(Jcurrent.J())
        max_I = float(np.max([abs(c.current.get_value()) for c in dipole_coils]))

        nphi_b = boozer_surface.surface.quadpoints_phi.size
        ntheta_b = boozer_surface.surface.quadpoints_theta.size
        BdotN = float(np.mean(np.abs(np.sum(
            bs.B().reshape((nphi_b, ntheta_b, 3)) * boozer_surface.surface.unitnormal(), axis=2,
        ))))
        vol = float(boozer_surface.surface.volume())

        history_path = os.path.join(out_dir, "iterations.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = {
                "config": {
                    "iota_target": IOTA_TARGET,
                    "f_b_threshold": fb_threshold,
                    "f_cp_threshold": FCP_THRESHOLD,
                    "iota_threshold": IOTA_THRESHOLD,
                    "penalty_weight": PENALTY_WEIGHT,
                    "mpol": mpol, "ntor": ntor,
                    "maxiter": MAXITER,
                    "adaptive_resolution": True,
                    "resolution_ladder": resolutions,
                    "fb_threshold_ladder": fb_thresholds,
                    "warm_start_from": prev_load_dir,
                    "sparse_run": True,
                    "theta_tol": THETA_TOL,
                    "removed_outboard_dipoles": len(removed_dipoles),
                },
                "iterations": [],
            }
        jb_g = float(JBoozerGuard.J())
        ji_g = float(Jiota.J())
        jc_g = float(JCurrentGuard.J())
        history["iterations"].append({
            "iteration": int(run_dict["it"]),
            "J": float(J),
            "grad_norm": float(np.linalg.norm(grad)),
            "nonQS_ratio": qs_val,
            "boozer_residual": fb_val,
            "iota": iota_val,
            "current_pnorm": cp_val,
            "max_current": max_I,
            "BdotN": BdotN,
            "volume": vol,
            "J_boozer_contrib": float(PENALTY_WEIGHT * jb_g),
            "J_iota_contrib": float(PENALTY_WEIGHT * ji_g),
            "J_current_contrib": float(PENALTY_WEIGHT * jc_g),
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        bs.save(os.path.join(out_dir, "bs_opt.json"))
        boozer_surface.surface.save(os.path.join(out_dir, "surf_opt.json"))

        partial_results = {
            "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
            "method": "true_epsilon_constraint_adaptive_res_sparse",
            "mpol": mpol,
            "ntor": ntor,
            "iota_target": IOTA_TARGET,
            "f_b_threshold": fb_threshold,
            "f_cp_threshold": FCP_THRESHOLD,
            "optimization_success": None,
            "optimization_message": "partial snapshot from callback",
            "final_objective": float(J),
            "nonQS_ratio": qs_val,
            "boozer_residual": fb_val,
            "final_iota": iota_val,
            "max_current": max_I,
            "# TF coils": len(tf_coils),
            "# dipole coils": len(dipole_coils),
            "removed_outboard_dipoles": len(removed_dipoles),
            "theta_tol": THETA_TOL,
            **shared_meta,
        }
        save(partial_results, os.path.join(out_dir, "results.json"))

        run_dict["it"] += 1

    if maxiter_scipy <= 0:
        J0, dJ0 = float(JF.J()), JF.dJ()
        res = OptimizeResult(
            x=x0,
            success=False,
            status=0,
            message=(
                f"Iteration budget exhausted: {n_done} accepted iteration(s) "
                f"already recorded (cap {MAXITER}); no further BFGS iterations run."
            ),
            fun=J0,
            jac=dJ0,
            nit=0,
            nfev=0,
            njev=0,
        )
    else:
        res = scipy_minimize(
            fun, x0, jac=True, method="BFGS", callback=callback,
            options={"maxiter": maxiter_scipy, "gtol": gtol},
        )
    print(f"\nOptimizer: {res.message}")

    coils_to_vtk(coils, filename=os.path.join(out_dir, "coils_opt"), close=True)
    bs.save(os.path.join(out_dir, "bs_opt.json"))
    VV.to_vtk(os.path.join(out_dir, "vacuum_vessel"))

    bs.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
    B = bs.B().reshape((nPhi, nTheta, 3))
    modB = np.sqrt(np.sum(B**2, axis=2))[:, :, None]
    BdotN_surf = np.sum(B * boozer_surface.surface.unitnormal(), axis=2)[:, :, None]
    pointData = {"B_N/B": BdotN_surf / modB}
    boozer_surface.surface.to_vtk(os.path.join(out_dir, "surf_opt"), extra_data=pointData)
    boozer_surface.surface.save(os.path.join(out_dir, "surf_opt.json"))

    max_I = float(np.max([abs(c.current.get_value()) for c in dipole_coils]))
    final_qs = float(JnonQSRatio.J())
    final_fb = float(JBoozerResidual.J())
    final_iota = float(iota.J())
    final_vol = float(boozer_surface.surface.volume())
    final_cp = float(Jcurrent.J())

    plot_relBfinal_norm_modB(bs, boozer_surface.surface, out_dir, "optimized", plot_config)
    plot_cross_section(
        boozer_surface.surface, VV, out_dir, "optimized", plot_config,
        base_dipole_coils=dipole_coils,
    )
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, out_dir, "optimized", plot_config)
    plot_objective_vs_iterations(out_dir, plot_config)

    results_output = {
        "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
        "method": "true_epsilon_constraint_adaptive_res_sparse",
        "mpol": mpol,
        "ntor": ntor,
        "maxiter": MAXITER,
        "iota_target": IOTA_TARGET,
        "f_b_threshold": fb_threshold,
        "f_cp_threshold": FCP_THRESHOLD,
        "iota_threshold": IOTA_THRESHOLD,
        "penalty_weight": PENALTY_WEIGHT,
        "current_pnorm_p": CURRENT_P_NORM,
        "gtol": gtol,
        "adaptive_resolution": True,
        "resolution_ladder": resolutions,
        "fb_threshold_ladder": fb_thresholds,
        "sparse_run": True,
        "theta_tol": THETA_TOL,
        "optimization_success": bool(res.success),
        "optimization_message": str(res.message),
        "final_objective": float(JF.J()),
        "nonQS_ratio": final_qs,
        "boozer_residual": final_fb,
        "final_iota": final_iota,
        "final_volume": final_vol,
        "iota_penalty": float(Jiota.J()),
        "current_pnorm": final_cp,
        "max_current": max_I,
        "# TF coils": len(tf_coils),
        "# dipole coils": len(dipole_coils),
        "removed_outboard_dipoles": len(removed_dipoles),
        **shared_meta,
    }
    save(results_output, os.path.join(out_dir, "results.json"))

    elapsed = time.time() - start_time
    print(f"Wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")

    log_file.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr

    return out_dir


walk_start = time.time()
prev_load_dir = INIT_DIR

for j, (res_val, fb_thresh) in enumerate(zip(resolutions, fb_thresholds)):
    mpol = ntor = res_val
    out_dir = _per_res_out_dir(mpol, ntor)
    step_label = f"[res {j + 1}/{len(resolutions)}] iota={IOTA_TARGET:g} mpol={mpol}"

    if (not START_FRESH) and _is_completed(out_dir):
        print(f"{step_label}: already completed, loading {out_dir} as warm start.")
        bs = load(os.path.join(out_dir, "bs_opt.json"))
        surf = load(os.path.join(out_dir, "surf_opt.json"))
        # Refresh global coil references to point at the freshly-loaded bs.
        # Without this, optimize_one_resolution() would still use the original
        # in-memory coils (with INIT_DIR currents), causing a corrupted warm
        # start that manifests as a huge J/Boozer-residual at iteration 0.
        coils = bs.coils
        tf_coils = coils[:num_tf]
        dipole_coils = coils[num_tf:]
        for c in dipole_coils:
            c.curve.fix_all()
        for c in tf_coils:
            c.current.fix_all()
        prev_load_dir = out_dir
        continue

    resume_this_step = False
    if (not START_FRESH) and _has_checkpoint(out_dir):
        print(f"{step_label}: found checkpoint, resuming from {out_dir}.")
        bs = load(os.path.join(out_dir, "bs_opt.json"))
        surf = load(os.path.join(out_dir, "surf_opt.json"))
        # Same fix as above for the in-progress checkpoint case: the loaded bs
        # is sparse already (saved post-filter), so we just re-extract coil
        # references and re-apply the fix_all settings.
        coils = bs.coils
        tf_coils = coils[:num_tf]
        dipole_coils = coils[num_tf:]
        for c in dipole_coils:
            c.curve.fix_all()
        for c in tf_coils:
            c.current.fix_all()
        resume_this_step = True

    print(f"{step_label}: starting (warm start from {prev_load_dir})")
    step_t0 = time.time()
    is_highest_res = (j == len(resolutions) - 1)
    step_gtol = 0.1 * GTOL if is_highest_res else GTOL
    out_dir = optimize_one_resolution(
        prev_load_dir, mpol, ntor, fb_thresh,
        gtol=step_gtol,
        resume_this_step=resume_this_step,
    )

    try:
        step_results = load(os.path.join(out_dir, "results.json"))
        final_fb = float(step_results.get("boozer_residual", float("inf")))
    except Exception:
        final_fb = float("inf")

    if final_fb > fb_thresh * (1.0 + FB_WARN_MARGIN):
        warnings.warn(
            f"{step_label}: Boozer residual {final_fb:.3e} exceeds threshold "
            f"{fb_thresh:.3e} by more than {FB_WARN_MARGIN * 100:.0f}%.",
            RuntimeWarning,
            stacklevel=2,
        )

    print(f"{step_label}: done in {(time.time() - step_t0) / 60:.1f} min -> {out_dir}")
    surf = load(os.path.join(out_dir, "surf_opt.json"))
    prev_load_dir = out_dir

print(
    f"Adaptive-resolution sparse run complete: "
    f"{len(resolutions)} resolution steps in {(time.time() - walk_start) / 60:.1f} min total."
)
