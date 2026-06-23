"""
Sequential / warm-started true epsilon-constraint single-stage scan over iota.

Same physics, objective, constraints, output schema, and per-run directory layout
as ``single_stage_true_epsilon.py`` (one directory per (iota, fcp, vt, mpol, ntor)
case, with bs_opt.json / surf_opt.json / results.json / iterations.json / log.txt
/ plots), but instead of cold-starting every (iota, fcp) point from the stage-2
solution we walk iota from --iota-min up to --iota-max in --iota-count steps,
and each step is initialized from the previous step's optimized coils + Boozer
surface.

This dramatically reduces sensitivity to local minima along the iota axis: each
new iota target is only ~Delta_iota away from a feasible point, so BFGS just has
to nudge the coils a bit, and the Boozer surface stays well-resolved.

Parallelism in a sweep is achieved across f_CP values (each f_CP launches its
own python process, each of which walks iota sequentially internally).

Usage:
  python single_stage_true_epsilon_sequential.py --init-dir <stage2_dir> \
      --iota-min 0.10 --iota-max 0.25 --iota-count 16 \
      --f-cp-threshold 150000

Re-running with the same (iota grid, fcp, vt, mpol, ntor) skips iota points
that already have a converged results.json under the sequential output tree;
pass --new to force a fresh walk.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from datetime import datetime
from scipy.optimize import minimize as scipy_minimize

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
        "Sequential warm-started true epsilon-constraint single-stage scan: "
        "walks iota from min to max, each step initialized from the previous."
    )
)
parser.add_argument("--init-dir", type=str, required=True,
    help="Directory with bs_opt.json, results.json, and surf_opt.json used to start the "
         "very first iota point in the walk. Can be either a stage-2 output directory OR "
         "a previous single-stage run directory (e.g. the lowest-iota subdirectory from a "
         "previously completed sequential walk at a different f_CP). Both layouts save the "
         "same files and the same inherited metadata (eq_name/eq_dir/VV_*/surf_nfp/surf_s/"
         "ntf, optionally surf_dof_scale), so this script handles either transparently.")
parser.add_argument("--iota-min", type=float, required=True,
    help="Smallest iota target in the sequential walk (start of the walk).")
parser.add_argument("--iota-max", type=float, required=True,
    help="Largest iota target in the sequential walk (end of the walk).")
parser.add_argument("--iota-count", type=int, required=True,
    help="Number of iota targets (linspace(iota-min, iota-max, iota-count)).")
parser.add_argument("--f-cp-threshold", type=float, required=True,
    help="Current p-norm upper bound [A] (constant across the walk).")
parser.add_argument("--f-b-threshold", type=float, default=1e-4,
    help="Boozer residual upper bound (default: 1e-4).")
parser.add_argument("--iota-threshold", type=float, default=0.0025,
    help="Absolute iota tolerance (default: 0.0025).")
parser.add_argument("--mpol", type=int, default=6,
    help="Boozer surface poloidal resolution (default: 6).")
parser.add_argument("--ntor", type=int, default=6,
    help="Boozer surface toroidal resolution (default: 6).")
parser.add_argument("--maxiter", type=int, default=200,
    help="Max BFGS iterations per iota step (default: 200). Warm starts converge fast, "
         "so this can be much smaller than the cold-start single_stage_true_epsilon.py default.")
parser.add_argument("--volume-target", type=float, default=0.3,
    help="Target volume of the Boozer surface (default: 0.3).")
parser.add_argument("--output-root", type=str, default="../single_stage_true_epsilon_sequential",
    help="Top-level output directory; per-iota subdirs are placed under "
         "<output-root>/<eq_name>/iota<X>_fcp<Y>kA_vt<Z>/mpol<M>_ntor<N>/.")
parser.add_argument(
    "--new",
    action="store_true",
    default=False,
    help=(
        "Start the walk from scratch: do not skip iota points that already have a "
        "converged results.json under the sequential output tree. Default behavior "
        "skips already-completed iota steps so a partially finished walk can be resumed."
    ),
)
args, _ = parser.parse_known_args()

# Unpack
INIT_DIR       = args.init_dir
IOTA_MIN       = args.iota_min
IOTA_MAX       = args.iota_max
IOTA_COUNT     = args.iota_count
FCP_THRESHOLD  = args.f_cp_threshold
FB_THRESHOLD   = args.f_b_threshold
IOTA_THRESHOLD = args.iota_threshold
mpol           = args.mpol
ntor           = args.ntor
MAXITER        = args.maxiter
VOL_TARGET     = args.volume_target
OUTPUT_ROOT    = args.output_root
START_FRESH    = args.new

if IOTA_COUNT < 1:
    raise ValueError("--iota-count must be >= 1.")
if IOTA_MAX < IOTA_MIN:
    raise ValueError("--iota-max must be >= --iota-min.")

# Fixed internal parameters
# These are tuned from the completed cold-start runs in
# examples/single_stage_true_epsilon/ (penalty_weight=1000, gtol=1e-3): with
# warm starts each iota step lands very close to feasibility, so we keep the
# same penalty (still drives Boozer residual exactly to f_b, iota exactly to
# target across the whole completed sweep) and keep the matching gtol, but
# can use a much smaller maxiter per step.
BOOZER_CW        = 1.0     # BoozerSurface least-squares constraint weight
PENALTY_WEIGHT   = 100.0     # single knob: how aggressively constraints are enforced vs QS objective
CURRENT_P_NORM   = 20.0    # p-norm exponent (smooth max-current proxy)
GTOL             = 1e-3

plot_config = PlotConfig(
    dpi=100, titlefontsize=16, axisfontsize=16,
    legendfontsize=14, ticklabelfontsize=14, cbarfontsize=16,
)


def plot_objective_vs_iterations(out_dir, config):
    """
    Plot total objective J and each contribution vs accepted iteration using
    iterations.json, plus ||grad J|| on the lower panel. Identical to the
    cold-start single_stage_true_epsilon.py version so all downstream
    postprocessing keeps working unchanged.
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
# INIT_DIR can be either a stage-2 directory or a previous single-stage run
# directory; both layouts save bs_opt.json + surf_opt.json + a results.json
# containing the same inherited metadata (eq_name/eq_dir/VV_*/surf_nfp/surf_s/
# ntf, optionally surf_dof_scale). This is what enables chaining sequential
# walks across f_CP values: the next walk simply points its --init-dir at the
# lowest-iota subdirectory of the previously-completed walk.
init_results = load(os.path.join(INIT_DIR, "results.json"))
eq_name = init_results["eq_name"]

# Where every per-iota run will be saved (one subdir per iota, same naming
# convention as the cold-start script for downstream postprocessing).
EQ_OUT_ROOT = os.path.join(OUTPUT_ROOT, eq_name)
os.makedirs(EQ_OUT_ROOT, exist_ok=True)

# Build the iota schedule (low -> high). The walk is sequential: iota_targets[k]
# starts from the optimized state at iota_targets[k-1].
iota_targets = np.linspace(IOTA_MIN, IOTA_MAX, IOTA_COUNT)


def _per_iota_out_dir(iota_target):
    """Same naming convention as single_stage_true_epsilon.py."""
    parent = os.path.join(
        EQ_OUT_ROOT,
        f"iota{iota_target:g}_fcp{FCP_THRESHOLD / 1e3:g}kA_vt{VOL_TARGET:g}",
    )
    return os.path.join(parent, f"mpol{mpol}_ntor{ntor}")


def _is_completed(out_dir):
    """A step is resumably complete only if optimizer terminated successfully.

    We intentionally do NOT skip runs that ended due to maxiter (or other
    non-success exits), even if they wrote final results.json.
    """
    rpath = os.path.join(out_dir, "results.json")
    if not os.path.isfile(rpath):
        return False
    try:
        rj = load(rpath)
    except Exception:
        return False
    return bool(rj.get("optimization_success", False))


def _has_checkpoint(out_dir):
    """Checkpoint required to resume an in-progress iota step."""
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
# Initial coils + surface (used as warm start for the first iota in the walk)
# ==============================================================================
print(f"Sequential true epsilon-constraint scan: {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"  init-dir:    {INIT_DIR}")
print(f"  output root: {EQ_OUT_ROOT}")
print(f"  iota grid:   {[f'{x:g}' for x in iota_targets]}")
print(f"  f_CP:        {FCP_THRESHOLD:.0f} A")
print(f"  vt:          {VOL_TARGET:g}")
print(f"  mpol/ntor:   {mpol}/{ntor}")
print(f"  penalty:     {PENALTY_WEIGHT}, gtol={GTOL}, maxiter/step={MAXITER}")

# bs and surf live across the loop. They get re-pointed to the freshly-saved
# optimized state at the end of each iota step.
bs = load(os.path.join(INIT_DIR, "bs_opt.json"))
surf_opt_path = os.path.join(INIT_DIR, "surf_opt.json")
if os.path.exists(surf_opt_path):
    surf = load(surf_opt_path)
    nPhi = surf.quadpoints_phi.size
    nTheta = surf.quadpoints_theta.size
else:
    # Only stage-2 (and not previous single-stage runs) hits this branch
    # because every single-stage run we ever save also writes surf_opt.json.
    nPhi, nTheta = 128, 64
    eq_path = os.path.join(init_results["eq_dir"], init_results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_path, s=init_results["surf_s"], range="half period", nphi=nPhi, ntheta=nTheta,
    )
    surf.set_dofs(init_results["surf_dof_scale"] * surf.get_dofs())

# Inherited metadata propagated into every per-iota results.json (same schema
# as the cold-start script). Comes from either stage-2 results.json or a
# previous single-stage results.json -- both store these fields.
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

# Vacuum vessel surface (only used for plots, geometry never changes).
VV = SurfaceRZFourier(nfp=init_results["surf_nfp"])
VV.set_rc(0, 0, init_results["VV_R0"])
VV.set_rc(1, 0, init_results["VV_a"])
VV.set_zs(1, 0, init_results["VV_b"])

# Split coils once -- coil identity is shared across every iota step (we only
# update the dipole currents). 'ntf' is preserved in both stage-2 and
# single-stage results.json so this works no matter which kind of init dir
# we're chaining from.
num_tf = init_results["ntf"] * 2 * init_results["surf_nfp"]
coils = bs.coils
tf_coils = coils[:num_tf]
dipole_coils = coils[num_tf:]
for c in dipole_coils:
    c.curve.fix_all()
for c in tf_coils:
    c.current.fix_all()


# ==============================================================================
# OPTIMIZE A SINGLE IOTA STEP (called in a loop below)
# ==============================================================================
def optimize_one_iota(iota_target, prev_load_dir, resume_this_step=False):
    """Run BFGS for a single iota target, warm-started from current bs/surf.

    prev_load_dir is recorded into results.json as the upstream of this step
    (so the run graph reflects the sequential walk).
    """
    out_dir = _per_iota_out_dir(iota_target)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Boozer surface for this iota ----
    # Re-initialize from the (warm) surf so the BoozerSurface object carries
    # the current iota target as its initial guess.
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
    boozer_surface = initialize_boozer_surface(
        surf, mpol, ntor, bs, VOL_TARGET, BOOZER_CW, iota_target, G0,
    )
    print(f"[iota={iota_target:g}] Initial Boozer volume: {boozer_surface.surface.volume():.4f}")

    # ---- Objective + constraints (rebuilt every step because Jiota target
    # changes; every other piece is structurally identical to the cold-start
    # script). ----
    bs_obj = BiotSavart(coils)
    JnonQSRatio = sum([NonQuasiSymmetricRatio(boozer_surface, bs_obj)])
    JBoozerResidual = sum([BoozerResidual(boozer_surface, bs_obj)])

    JBoozerGuard = (1.0 / FB_THRESHOLD**2) * QuadraticPenalty(
        JBoozerResidual, FB_THRESHOLD, f="max",
    )
    iota = Iotas(boozer_surface)
    Jiota = (1.0 / IOTA_THRESHOLD**2) * QuadraticPenalty(iota, iota_target)
    Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=CURRENT_P_NORM)
    JCurrentGuard = (1.0 / FCP_THRESHOLD**2) * QuadraticPenalty(
        Jcurrent, FCP_THRESHOLD, f="max",
    )
    JF = JnonQSRatio + PENALTY_WEIGHT * (JBoozerGuard + Jiota + JCurrentGuard)

    # ---- Per-step log file ----
    # Fresh run rewrites log.txt; in-place resume appends to preserve history.
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    log_path = os.path.join(out_dir, "log.txt")
    log_mode = "a" if resume_this_step else "w"
    log_file = open(log_path, log_mode, buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    start_time = time.time()
    print(f"\n{'=' * 70}")
    print(f"Sequential true epsilon-constraint step -- {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}")
    print(f"Output:     {out_dir}")
    print(f"warm-start: {prev_load_dir}")
    print(f"iota_target={iota_target:g},  f_b={FB_THRESHOLD:.1e},  f_CP={FCP_THRESHOLD:.0f} A")
    print(f"mpol={mpol}, ntor={ntor}, maxiter={MAXITER}")
    if resume_this_step:
        print(f"Resuming in-place from existing checkpoint at callback it={_next_iteration_index_from_history(out_dir)}.")
    print(f"# TF: {len(tf_coils)},  # dipole: {len(dipole_coils)}")
    print(f"Objective: nonQS + {PENALTY_WEIGHT} * (Boozer_guard + iota_penalty + current_guard)\n")

    print(f"Initial nonQS ratio:      {JnonQSRatio.J():.6e}")
    print(f"Initial Boozer residual:  {JBoozerResidual.J():.6e}  (threshold {FB_THRESHOLD:.1e})")
    print(f"Initial iota:             {iota.J():.6f}  (target {iota_target:g} +/- {IOTA_THRESHOLD:g})")
    print(f"Initial current p-norm:   {Jcurrent.J():.1f} A  (threshold {FCP_THRESHOLD:.0f} A)")

    # First iota in the walk gets initial cross-section plotted. Subsequent
    # iota steps skip it (their "initial" state is essentially the previous
    # step's optimized state and would just clutter).
    is_first_step = (prev_load_dir == INIT_DIR) and (not resume_this_step)
    if is_first_step:
        plot_cross_section(
            boozer_surface.surface, VV, out_dir, "initial", plot_config,
            base_dipole_coils=dipole_coils,
        )

    # ---- Optimizer state (mirrors cold-start script's run_dict) ----
    x0 = JF.x.copy()
    run_dict = {
        "sdofs": boozer_surface.surface.x.copy(),
        "iota": boozer_surface.res["iota"],
        "G": boozer_surface.res["G"],
        "J": JF.J(),
        "dJ": JF.dJ().copy(),
        "it": _next_iteration_index_from_history(out_dir) if resume_this_step else 1,
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

        # Console diagnostics (per accepted iteration)
        fb_flag = "  [!>fb]" if fb_val > FB_THRESHOLD else ""
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
                    "f_b_threshold": FB_THRESHOLD,
                    "f_cp_threshold": FCP_THRESHOLD,
                    "iota_threshold": IOTA_THRESHOLD,
                    "penalty_weight": PENALTY_WEIGHT,
                    "mpol": mpol, "ntor": ntor,
                    "maxiter": MAXITER,
                    "sequential_walk": True,
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
            # Pieces of J = J_nonQS + w_pen * (JBoozerGuard + Jiota + JCurrentGuard)
            "J_boozer_contrib": float(PENALTY_WEIGHT * jb_g),
            "J_iota_contrib": float(PENALTY_WEIGHT * ji_g),
            "J_current_contrib": float(PENALTY_WEIGHT * jc_g),
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # ---- Partial saves (survive early termination of any single step) ----
        bs.save(os.path.join(out_dir, "bs_opt.json"))
        boozer_surface.surface.save(os.path.join(out_dir, "surf_opt.json"))

        partial_results = {
            "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
            "method": "true_epsilon_constraint_sequential",
            "mpol": mpol, "ntor": ntor,
            "iota_target": iota_target,
            "f_b_threshold": FB_THRESHOLD,
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

    # ---- Run L-BFGS for this iota step ----
    res = scipy_minimize(
        fun, x0, jac=True, method="BFGS", callback=callback,
        options={"maxiter": MAXITER, "gtol": GTOL},
    )
    print(f"\nOptimizer: {res.message}")

    # ==========================================================================
    # SAVE FINAL OUTPUTS FOR THIS IOTA STEP (matches cold-start script)
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
    print(f"FINAL RESULTS (iota_target={iota_target:g})")
    print(f"{'=' * 70}")
    print(f"  nonQS ratio:      {final_qs:.6e}")
    print(f"  Boozer residual:  {final_fb:.6e}  (threshold {FB_THRESHOLD:.1e})")
    print(f"  Iota:             {final_iota:.6f}  (target {iota_target:g})")
    print(f"  Max current:      {max_I:.0f} A  (threshold {FCP_THRESHOLD:.0f} A)")
    print(f"  Current p-norm:   {final_cp:.0f} A")
    print(f"  Volume:           {final_vol:.4f}")

    # Diagnostic plots (same set as cold-start script).
    plot_relBfinal_norm_modB(bs, boozer_surface.surface, out_dir, "optimized", plot_config)
    plot_cross_section(
        boozer_surface.surface, VV, out_dir, "optimized", plot_config,
        base_dipole_coils=dipole_coils,
    )
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, out_dir, "optimized", plot_config)
    plot_objective_vs_iterations(out_dir, plot_config)

    results_output = {
        "graph": {"init_dir": INIT_DIR, "load_dir": prev_load_dir},
        "method": "true_epsilon_constraint_sequential",

        # Configuration
        "mpol": mpol,
        "ntor": ntor,
        "maxiter": MAXITER,
        "iota_target": iota_target,
        "f_b_threshold": FB_THRESHOLD,
        "f_cp_threshold": FCP_THRESHOLD,
        "iota_threshold": IOTA_THRESHOLD,
        "penalty_weight": PENALTY_WEIGHT,
        "current_pnorm_p": CURRENT_P_NORM,
        "gtol": GTOL,
        "sequential_walk": True,
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

    # Restore streams
    log_file.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr

    return out_dir, boozer_surface


# ==============================================================================
# WALK iota_min -> iota_max, warm-starting each step from the previous one
# ==============================================================================
walk_start = time.time()
prev_load_dir = INIT_DIR  # very first step warm-starts from stage-2

for k, iota_target in enumerate(iota_targets):
    out_dir = _per_iota_out_dir(iota_target)

    # Resume support: skip iota points that already have a converged
    # results.json under the sequential output tree, but only when the user
    # didn't pass --new. We still need to re-load bs/surf from the skipped
    # step's output so the next un-finished step warm-starts correctly.
    if (not START_FRESH) and _is_completed(out_dir):
        print(
            f"\n[step {k + 1}/{IOTA_COUNT}] iota={iota_target:g}: already completed, "
            f"loading {out_dir} as warm start for next step."
        )
        bs = load(os.path.join(out_dir, "bs_opt.json"))
        surf = load(os.path.join(out_dir, "surf_opt.json"))
        # Re-split coils from the freshly-loaded bs (object identity changed).
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
        print(
            f"\n[step {k + 1}/{IOTA_COUNT}] iota={iota_target:g}: found in-progress checkpoint, "
            f"resuming from {out_dir}."
        )
        bs = load(os.path.join(out_dir, "bs_opt.json"))
        surf = load(os.path.join(out_dir, "surf_opt.json"))
        # Re-split coils from the freshly-loaded bs (object identity changed).
        coils = bs.coils
        tf_coils = coils[:num_tf]
        dipole_coils = coils[num_tf:]
        for c in dipole_coils:
            c.curve.fix_all()
        for c in tf_coils:
            c.current.fix_all()
        prev_load_dir = out_dir
        resume_this_step = True

    print(
        f"\n[step {k + 1}/{IOTA_COUNT}] Starting iota={iota_target:g}  "
        f"(warm start from {prev_load_dir})"
    )
    step_t0 = time.time()
    out_dir, boozer_surface = optimize_one_iota(
        iota_target, prev_load_dir, resume_this_step=resume_this_step
    )
    print(
        f"[step {k + 1}/{IOTA_COUNT}] iota={iota_target:g} done in "
        f"{(time.time() - step_t0) / 60:.1f} min -> {out_dir}"
    )

    # The optimized surface from this step becomes the warm start for the
    # next iota target. We re-load surf from the just-saved JSON to get a
    # clean SurfaceRZFourier-compatible initial guess for
    # initialize_boozer_surface() (which calls surf_prev.gamma()).
    surf = load(os.path.join(out_dir, "surf_opt.json"))
    prev_load_dir = out_dir

print(
    f"\nSequential walk complete: {IOTA_COUNT} iota targets in "
    f"{(time.time() - walk_start) / 60:.1f} min total."
)
