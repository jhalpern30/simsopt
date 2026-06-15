"""
Sequential / warm-started true epsilon-constraint single-stage scan over iota
with **adaptive resolution**.

Same physics, objective, constraints, output schema, and per-run directory
layout as ``single_stage_true_epsilon_sequential.py``, but instead of a fixed
(mpol, ntor) throughout the walk, a *ladder* of resolutions is climbed for
**each** iota target before the walk advances to the next iota.

Walk sequence (iota × resolution):
  iota_min, res[0]  ->  iota_min, res[1]  ->  ...  ->  iota_min, res[-1]
  iota_1,   res[0]  ->  iota_1,   res[1]  ->  ...  ->  iota_1,   res[-1]
  ...
  iota_max, res[0]  ->  iota_max, res[1]  ->  ...  ->  iota_max, res[-1]

Warm-start chain:
  - (iota_0,   res[0])   initialised from INIT_DIR
  - (iota_i,   res[j+1]) warm-started from (iota_i,   res[j])   output
  - (iota_{i+1}, res[0]) warm-started from (iota_i,   res[-1])  output  ← highest res

Each (iota, resolution) pair writes its output under
  <output-root>/<eq_name>/iota{X}_fcp{Y}kA_vt{Z}/mpol{M}_ntor{N}/
which is the same layout as the single-resolution sequential script so all
downstream postprocessing works unchanged.

A RuntimeWarning (not an error) is raised if any resolution step finishes
with a Boozer residual that is more than 5 % above its target fb_threshold.

Usage:
  python single_stage_true_epsilon_adaptive_res.py \\
      --init-dir <stage2_dir> \\
      --iota-min 0.10 --iota-max 0.25 --iota-count 16 \\
      --f-cp-threshold 150000 \\
      --resolutions 6,9,12 \\
      --fb-thresholds 1e-4,5e-5,2.5e-5

Re-running with the same (iota grid, fcp, vt) skips (iota, res) pairs that
already have a converged results.json; pass --new to force a full re-run.
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

# ==============================================================================
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(
    description=(
        "Sequential warm-started true epsilon-constraint single-stage scan "
        "with adaptive resolution: walks iota from min to max, climbing a "
        "resolution ladder for each iota target before advancing."
    )
)
parser.add_argument("--init-dir", type=str, required=True,
    help="Directory with bs_opt.json, results.json, and surf_opt.json used to start "
         "the very first (iota, resolution) point in the walk.  Can be either a "
         "stage-2 output directory OR a previous single-stage run directory.")
parser.add_argument("--iota-min", type=float, required=True,
    help="Smallest iota target in the sequential walk (start of the walk).")
parser.add_argument("--iota-max", type=float, required=True,
    help="Largest iota target in the sequential walk (end of the walk).")
parser.add_argument("--iota-count", type=int, required=True,
    help="Number of iota targets (linspace(iota-min, iota-max, iota-count)).")
parser.add_argument("--f-cp-threshold", type=float, required=True,
    help="Current p-norm upper bound [A] (constant across the walk).")
parser.add_argument("--resolutions", type=str, required=True,
    help="Comma-separated list of mpol=ntor resolution values to climb for each "
         "iota target (e.g. '6,9,12').  Must have the same length as --fb-thresholds.")
parser.add_argument("--fb-thresholds", type=str, required=True,
    help="Comma-separated list of Boozer residual upper bounds, one per resolution "
         "(e.g. '1e-4,5e-5,2.5e-5').  A RuntimeWarning is raised (not an error) if a "
         "step finishes with residual > threshold * 1.05.")
parser.add_argument("--iota-threshold", type=float, default=0.0025,
    help="Absolute iota tolerance (default: 0.0025).")
parser.add_argument("--maxiter", type=int, default=200,
    help="Max accepted BFGS iterations per (iota, resolution) step, counting "
         "any prior progress when resuming from checkpoint (default: 200). "
         "Example: --maxiter 200 with 100 rows in iterations.json runs at most "
         "100 more iterations.")
parser.add_argument("--volume-target", type=float, default=0.3,
    help="Target volume of the Boozer surface (default: 0.3).")
parser.add_argument("--output-root", type=str,
    default="../single_stage_true_epsilon_adaptive_res",
    help="Top-level output directory; per-iota subdirs are placed under "
         "<output-root>/<eq_name>/iota<X>_fcp<Y>kA_vt<Z>/mpol<M>_ntor<N>/.")
parser.add_argument(
    "--new",
    action="store_true",
    default=False,
    help=(
        "Start the walk from scratch: do not skip (iota, resolution) pairs that "
        "already have a converged results.json.  Default behaviour resumes a "
        "partially finished walk."
    ),
)
args, _ = parser.parse_known_args()

# ---------- parse resolution ladder ----------
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

# ---------- unpack remaining args ----------
INIT_DIR       = args.init_dir
IOTA_MIN       = args.iota_min
IOTA_MAX       = args.iota_max
IOTA_COUNT     = args.iota_count
FCP_THRESHOLD  = args.f_cp_threshold
IOTA_THRESHOLD = args.iota_threshold
MAXITER        = args.maxiter
VOL_TARGET     = args.volume_target
OUTPUT_ROOT    = args.output_root
START_FRESH    = args.new

if IOTA_COUNT < 1:
    raise ValueError("--iota-count must be >= 1.")
if IOTA_MAX < IOTA_MIN:
    raise ValueError("--iota-max must be >= --iota-min.")

# Fixed internal parameters (same as the single-resolution sequential script).
BOOZER_CW        = 1.0
PENALTY_WEIGHT   = 100.0
CURRENT_P_NORM   = 20.0
GTOL             = 1e-3
CURRENT_SCALE    = 100.0 # penalize the current more strongly than the residual

# Tolerance for the per-step convergence warning (5 % above threshold).
FB_WARN_MARGIN   = 0.05

plot_config = PlotConfig(
    dpi=100, titlefontsize=16, axisfontsize=16,
    legendfontsize=14, ticklabelfontsize=14, cbarfontsize=16,
)


def plot_objective_vs_iterations(out_dir, config):
    """
    Plot total objective J and each contribution vs accepted iteration using
    iterations.json, plus ||grad J|| on the lower panel.  Identical to the
    single-resolution sequential script so all downstream postprocessing works.
    """
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


# ==============================================================================
# LOAD INIT STATE (only once for the whole walk)
# ==============================================================================
init_results = load(os.path.join(INIT_DIR, "results.json"))
eq_name = init_results["eq_name"]

EQ_OUT_ROOT = os.path.join(OUTPUT_ROOT, eq_name)
os.makedirs(EQ_OUT_ROOT, exist_ok=True)

iota_targets = np.linspace(IOTA_MIN, IOTA_MAX, IOTA_COUNT)


def _per_iota_out_dir(iota_target, mpol, ntor):
    """Directory for a single (iota, mpol, ntor) point -- same naming as the
    single-resolution scripts so downstream postprocessing is unchanged."""
    parent = os.path.join(
        EQ_OUT_ROOT,
        f"iota{iota_target:g}_fcp{FCP_THRESHOLD / 1e3:g}kA_vt{VOL_TARGET:g}",
    )
    return os.path.join(parent, f"mpol{mpol}_ntor{ntor}")


def _is_completed(out_dir):
    """True only if the optimizer terminated with success."""
    rpath = os.path.join(out_dir, "results.json")
    if not os.path.isfile(rpath):
        return False
    try:
        rj = load(rpath)
    except Exception:
        return False
    return bool(rj.get("optimization_success", False))


def _has_checkpoint(out_dir):
    """Checkpoint required to resume an in-progress (iota, res) step."""
    needed = ["bs_opt.json", "surf_opt.json", "results.json"]
    return all(os.path.isfile(os.path.join(out_dir, f)) for f in needed)


def _next_iteration_index_from_history(out_dir):
    """1-based callback index after existing iterations.json rows."""
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


# ==============================================================================
# Initial coils + surface
# ==============================================================================
print(f"Adaptive-resolution sequential true epsilon-constraint scan: "
      f"{datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  init-dir:    {INIT_DIR}")
print(f"  output root: {EQ_OUT_ROOT}")
print(f"  iota grid:   {[f'{x:g}' for x in iota_targets]}")
print(f"  f_CP:        {FCP_THRESHOLD:.0f} A")
print(f"  vt:          {VOL_TARGET:g}")
print(f"  resolution ladder:")
for res, fbt in zip(resolutions, fb_thresholds):
    print(f"    mpol=ntor={res:2d}  fb_threshold={fbt:.2e}")
print(f"  penalty:     {PENALTY_WEIGHT}, gtol={GTOL} (x0.1 at highest res), "
      f"maxiter/step={MAXITER} (cumulative per step incl. resume)")

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
    "eq_name":  init_results["eq_name"],
    "eq_dir":   init_results["eq_dir"],
    "surf_nfp": init_results["surf_nfp"],
    "surf_s":   init_results["surf_s"],
    "VV_R0":    init_results["VV_R0"],
    "VV_a":     init_results["VV_a"],
    "VV_b":     init_results["VV_b"],
    "ntf":      init_results["ntf"],
}
if "surf_dof_scale" in init_results:
    shared_meta["surf_dof_scale"] = init_results["surf_dof_scale"]

VV = SurfaceRZFourier(nfp=init_results["surf_nfp"])
VV.set_rc(0, 0, init_results["VV_R0"])
VV.set_rc(1, 0, init_results["VV_a"])
VV.set_zs(1, 0, init_results["VV_b"])

num_tf = init_results["ntf"] * 2 * init_results["surf_nfp"]
coils = bs.coils
tf_coils = coils[:num_tf]
dipole_coils = coils[num_tf:]
for c in dipole_coils:
    c.curve.fix_all()
for c in tf_coils:
    c.current.fix_all()


# ==============================================================================
# OPTIMIZE A SINGLE (IOTA, RESOLUTION) STEP
# ==============================================================================
def optimize_one_iota(iota_target, prev_load_dir, mpol, ntor, fb_threshold,
                      gtol=None, resume_this_step=False):
    """Run BFGS for one (iota, mpol/ntor) point, warm-started from current bs/surf.

    Parameters
    ----------
    iota_target : float
        Rotational transform target for this step.
    prev_load_dir : str
        Directory that provided the warm start (recorded in results.json for
        provenance).
    mpol : int
        Boozer surface poloidal resolution.
    ntor : int
        Boozer surface toroidal resolution.
    fb_threshold : float
        Boozer residual upper bound for this resolution level.
    gtol : float, optional
        Gradient norm convergence tolerance for BFGS.  Defaults to the global
        GTOL.  Pass ``GTOL * 0.1`` for the highest-resolution stage.
    resume_this_step : bool
        If True, append to the existing log and continue from the checkpoint.
        The global ``MAXITER`` is a cumulative cap for this (iota, resolution)
        directory: remaining BFGS iterations are ``MAXITER`` minus accepted
        iterations already stored in ``iterations.json``.
    """
    if gtol is None:
        gtol = GTOL
    out_dir = _per_iota_out_dir(iota_target, mpol, ntor)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Boozer surface for this (iota, resolution) ----
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
    boozer_surface = initialize_boozer_surface(
        surf, mpol, ntor, bs, VOL_TARGET, BOOZER_CW, iota_target, G0,
    )
    print(f"[iota={iota_target:g}, mpol={mpol}] Initial Boozer volume: "
          f"{boozer_surface.surface.volume():.4f}")

    # ---- Objective + constraints ----
    bs_obj = BiotSavart(coils)
    JnonQSRatio    = sum([NonQuasiSymmetricRatio(boozer_surface, bs_obj)])
    JBoozerResidual = sum([BoozerResidual(boozer_surface, bs_obj)])

    JBoozerGuard = (1.0 / fb_threshold**2) * QuadraticPenalty(
        JBoozerResidual, fb_threshold, f="max",
    )
    iota = Iotas(boozer_surface)
    Jiota = (1.0 / IOTA_THRESHOLD**2) * QuadraticPenalty(iota, iota_target)
    Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=CURRENT_P_NORM)
    JCurrentGuard = (CURRENT_SCALE / FCP_THRESHOLD**2) * QuadraticPenalty(
        Jcurrent, FCP_THRESHOLD, f="max",
    )
    JF = JnonQSRatio + PENALTY_WEIGHT * (JBoozerGuard + Jiota + JCurrentGuard)

    # ---- Per-step log file ----
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    log_path = os.path.join(out_dir, "log.txt")
    log_mode = "a" if resume_this_step else "w"
    log_file = open(log_path, log_mode, buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    start_time = time.time()
    print(f"\n{'=' * 70}")
    print(f"Adaptive-resolution sequential step -- {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}")
    print(f"Output:     {out_dir}")
    print(f"warm-start: {prev_load_dir}")
    print(f"iota_target={iota_target:g},  f_b={fb_threshold:.1e},  f_CP={FCP_THRESHOLD:.0f} A")
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
    if resume_this_step:
        print(f"Resuming in-place from existing checkpoint at callback it={next_it}.")
    print(f"# TF: {len(tf_coils)},  # dipole: {len(dipole_coils)}")
    print(f"Objective: nonQS + {PENALTY_WEIGHT} * (Boozer_guard + iota_penalty + current_guard)\n")

    print(f"Initial nonQS ratio:      {JnonQSRatio.J():.6e}")
    print(f"Initial Boozer residual:  {JBoozerResidual.J():.6e}  (threshold {fb_threshold:.1e})")
    print(f"Initial iota:             {iota.J():.6f}  (target {iota_target:g} +/- {IOTA_THRESHOLD:g})")
    print(f"Initial current p-norm:   {Jcurrent.J():.1f} A  (threshold {FCP_THRESHOLD:.0f} A)")

    # Plot initial cross-section only for the very first (iota, res) step.
    is_first_step = (prev_load_dir == INIT_DIR) and (not resume_this_step)
    if is_first_step:
        plot_cross_section(
            boozer_surface.surface, VV, out_dir, "initial", plot_config,
            base_dipole_coils=dipole_coils,
        )

    # ---- Optimizer state ----
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
        """Evaluate objective and gradient; reject bad Boozer solves."""
        dx = np.linalg.norm(x - run_dict["x_prev"])
        run_dict["x_prev"] = x.copy()
        run_dict["lscount"] += 1

        # Reset to last accepted Boozer state
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
            print(f"/!\\ Boozer rejected (consecutive: {run_dict['failed_boozer_solves']})")
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
        """Accept step: cache state, log diagnostics, save partial outputs."""
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

        fb_flag = "  [!>fb]" if fb_val > fb_threshold else ""
        cp_flag = "  [!>fcp]" if cp_val > FCP_THRESHOLD else ""
        print(f"\n{'=' * 60}")
        print(f"ITER {run_dict['it']:3d}  J={J:.6e}  ||grad J||={np.linalg.norm(grad):.6e}")
        print(f"  nonQS={qs_val:.6e}")
        print(f"  Boozer={fb_val:.6e}{fb_flag}")
        print(f"  iota={iota_val:.4f} (target {iota_target:g})")
        print(f"  <|B.n|>={BdotN:.6e}")
        print(f"  I_pnorm={cp_val:.0f}A (limit {FCP_THRESHOLD:.0f}A){cp_flag}")
        print(f"  I_max={max_I:.0f}A")
        print(f"  volume={vol:.4f}\n")

        # ---- Iteration history JSON ----
        history_path = os.path.join(out_dir, "iterations.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = {
                "config": {
                    "iota_target": iota_target,
                    "f_b_threshold": fb_threshold,
                    "f_cp_threshold": FCP_THRESHOLD,
                    "iota_threshold": IOTA_THRESHOLD,
                    "penalty_weight": PENALTY_WEIGHT,
                    "mpol": mpol, "ntor": ntor,
                    "maxiter": MAXITER,
                    "sequential_walk": True,
                    "adaptive_resolution": True,
                    "resolution_ladder": resolutions,
                    "fb_threshold_ladder": fb_thresholds,
                    "warm_start_from": prev_load_dir,
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

        # ---- Partial saves ----
        bs.save(os.path.join(out_dir, "bs_opt.json"))
        boozer_surface.surface.save(os.path.join(out_dir, "surf_opt.json"))

        partial_results = {
            "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
            "method": "true_epsilon_constraint_sequential_adaptive_res",
            "mpol": mpol, "ntor": ntor,
            "iota_target": iota_target,
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
            **shared_meta,
        }
        save(partial_results, os.path.join(out_dir, "results.json"))

        run_dict["it"] += 1

    # ---- Run BFGS for this (iota, resolution) step ----
    # MAXITER is a cumulative cap vs accepted iterations already in iterations.json
    # when resuming (n_done); not an additional 200 on top of prior work.
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

    # ==========================================================================
    # SAVE FINAL OUTPUTS
    # ==========================================================================
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

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS (iota_target={iota_target:g}, mpol={mpol}, ntor={ntor})")
    print(f"{'=' * 70}")
    print(f"  nonQS ratio:      {final_qs:.6e}")
    print(f"  Boozer residual:  {final_fb:.6e}  (threshold {fb_threshold:.1e})")
    print(f"  Iota:             {final_iota:.6f}  (target {iota_target:g})")
    print(f"  Max current:      {max_I:.0f} A  (threshold {FCP_THRESHOLD:.0f} A)")
    print(f"  Current p-norm:   {final_cp:.0f} A")
    print(f"  Volume:           {final_vol:.4f}")

    plot_relBfinal_norm_modB(bs, boozer_surface.surface, out_dir, "optimized", plot_config)
    plot_cross_section(
        boozer_surface.surface, VV, out_dir, "optimized", plot_config,
        base_dipole_coils=dipole_coils,
    )
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, out_dir, "optimized", plot_config)
    plot_objective_vs_iterations(out_dir, plot_config)

    results_output = {
        "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
        "method": "true_epsilon_constraint_sequential_adaptive_res",

        # Configuration
        "mpol": mpol,
        "ntor": ntor,
        "maxiter": MAXITER,
        "iota_target": iota_target,
        "f_b_threshold": fb_threshold,
        "f_cp_threshold": FCP_THRESHOLD,
        "iota_threshold": IOTA_THRESHOLD,
        "penalty_weight": PENALTY_WEIGHT,
        "current_pnorm_p": CURRENT_P_NORM,
        "gtol": gtol,
        "sequential_walk": True,
        "adaptive_resolution": True,
        "resolution_ladder": resolutions,
        "fb_threshold_ladder": fb_thresholds,
        "iota_walk_min": IOTA_MIN,
        "iota_walk_max": IOTA_MAX,
        "iota_walk_count": IOTA_COUNT,

        # Optimization results
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

        # Coil counts
        "# TF coils": len(tf_coils),
        "# dipole coils": len(dipole_coils),

        # Inherited from upstream
        **shared_meta,
    }
    save(results_output, os.path.join(out_dir, "results.json"))
    print(f"\nResults saved to {os.path.join(out_dir, 'results.json')}")

    elapsed = time.time() - start_time
    print(f"Wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")

    log_file.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr

    return out_dir, boozer_surface


# ==============================================================================
# WALK  iota_min -> iota_max  ×  res[0] -> res[-1]
#
# Outer loop: iota targets (low -> high, warm-started from highest-res prev iota)
# Inner loop: resolution ladder (low -> high, warm-started within each iota)
# ==============================================================================
walk_start = time.time()
# prev_load_dir tracks the warm-start for the *next* (iota, res) step:
#   - after (iota_i, res[j])  it points to that step's output dir
#   - so (iota_i, res[j+1])  warm-starts correctly
#   - and (iota_{i+1}, res[0]) warm-starts from the res[-1] of iota_i
prev_load_dir = INIT_DIR

for k, iota_target in enumerate(iota_targets):
    for j, (res_val, fb_thresh) in enumerate(zip(resolutions, fb_thresholds)):
        mpol = ntor = res_val
        out_dir = _per_iota_out_dir(iota_target, mpol, ntor)

        step_label = (f"[iota {k + 1}/{IOTA_COUNT}, res {j + 1}/{len(resolutions)}]"
                      f"  iota={iota_target:g}  mpol={mpol}")

        # ---- Resume: skip already-converged (iota, res) pairs ----
        if (not START_FRESH) and _is_completed(out_dir):
            print(
                f"\n{step_label}: already completed, "
                f"loading {out_dir} as warm start for next step."
            )
            bs = load(os.path.join(out_dir, "bs_opt.json"))
            surf = load(os.path.join(out_dir, "surf_opt.json"))
            coils = bs.coils
            tf_coils = coils[:num_tf]
            dipole_coils = coils[num_tf:]
            for c in dipole_coils:
                c.curve.fix_all()
            for c in tf_coils:
                c.current.fix_all()
            prev_load_dir = out_dir
            continue

        # ---- Resume: in-progress checkpoint ----
        resume_this_step = False
        if (not START_FRESH) and _has_checkpoint(out_dir):
            print(
                f"\n{step_label}: found in-progress checkpoint, resuming from {out_dir}."
            )
            bs = load(os.path.join(out_dir, "bs_opt.json"))
            surf = load(os.path.join(out_dir, "surf_opt.json"))
            coils = bs.coils
            tf_coils = coils[:num_tf]
            dipole_coils = coils[num_tf:]
            for c in dipole_coils:
                c.curve.fix_all()
            for c in tf_coils:
                c.current.fix_all()
            # Do NOT update prev_load_dir here.  It already holds the actual
            # upstream warm-start directory (set by a prior completed-skip or
            # INIT_DIR for the first step) and is used for accurate
            # graph.load_dir provenance in results.json.  The coil/surface
            # state is loaded from the checkpoint above; prev_load_dir is only
            # metadata and will be updated to out_dir after the step completes.
            resume_this_step = True

        print(f"\n{step_label}: starting  (warm start from {prev_load_dir})")
        step_t0 = time.time()
        is_highest_res = (j == len(resolutions) - 1)
        step_gtol = 0.1 * GTOL if is_highest_res else GTOL
        out_dir, boozer_surface = optimize_one_iota(
            iota_target, prev_load_dir, mpol, ntor, fb_thresh,
            gtol=step_gtol,
            resume_this_step=resume_this_step,
        )

        # ---- Convergence warning ----
        try:
            step_results = load(os.path.join(out_dir, "results.json"))
            final_fb = float(step_results.get("boozer_residual", float("inf")))
        except Exception:
            final_fb = float("inf")

        if final_fb > fb_thresh * (1.0 + FB_WARN_MARGIN):
            warnings.warn(
                f"{step_label}: Boozer residual {final_fb:.3e} exceeds threshold "
                f"{fb_thresh:.3e} by more than {FB_WARN_MARGIN * 100:.0f}%. "
                f"Consider increasing MAXITER or using a looser fb_threshold for this "
                f"resolution level.",
                RuntimeWarning,
                stacklevel=2,
            )

        print(
            f"{step_label}: done in {(time.time() - step_t0) / 60:.1f} min -> {out_dir}"
        )

        # Re-load surf from the just-saved JSON so initialize_boozer_surface()
        # in the next step gets a clean SurfaceRZFourier initial guess.
        surf = load(os.path.join(out_dir, "surf_opt.json"))
        prev_load_dir = out_dir

print(
    f"\nAdaptive-resolution sequential walk complete: "
    f"{IOTA_COUNT} iota targets × {len(resolutions)} resolutions = "
    f"{IOTA_COUNT * len(resolutions)} total steps in "
    f"{(time.time() - walk_start) / 60:.1f} min total."
)
