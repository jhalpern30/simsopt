"""
True epsilon-constraint single-stage optimization.

Objective: minimize nonQS ratio (quasi-symmetry error)
Constraints (quadratic penalties):
  - Boozer residual  <= f_b   (default 1e-4, good flux surfaces)
  - |iota - iota*|   <= eps_i (absolute iota tolerance)
  - ||I||_p          <= f_CP  (current p-norm threshold)

Usage:
  python single_stage_true_epsilon.py --init-dir /path/to/results \\
      --iota-target 0.15 --f-cp-threshold 50000

  Re-run the same command with the same iota / f_CP / volume / mpol / ntor to continue
  for another --maxiter BFGS steps in the existing output directory. Use --new to discard
  that output history and start over (still from stage-2 or a lower-resolution sibling).

  Per-run outputs live under --output-root/<eq_name>/iota*_fcp*kA_vt*/mpolM_ntorN/ (see
  single_stage_true_epsilon_sequential.py for the same layout).
"""

import os
import re
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
# CLI — only the essentials
# ==============================================================================
parser = argparse.ArgumentParser(
    description="True epsilon-constraint: minimize QS error with f_b, iota, and current constraints."
)
parser.add_argument("--init-dir", type=str, required=True,
    help="Directory with bs_opt.json, results.json, and optionally surf_opt.json.")
parser.add_argument("--iota-target", type=float, required=True,
    help="Target iota value.")
parser.add_argument("--f-cp-threshold", type=float, required=True,
    help="Current p-norm upper bound [A].")
parser.add_argument("--f-b-threshold", type=float, default=1e-4,
    help="Boozer residual upper bound (default: 1e-4).")
parser.add_argument("--iota-threshold", type=float, default=0.0025,
    help="Absolute iota tolerance (default: 0.0025).")
parser.add_argument("--mpol", type=int, default=6,
    help="Boozer surface poloidal resolution (default: 6).")
parser.add_argument("--ntor", type=int, default=6,
    help="Boozer surface toroidal resolution (default: 6).")
parser.add_argument("--maxiter", type=int, default=200,
    help="Max BFGS iterations (default: 200).")
parser.add_argument("--gtol", type=float, default=1e-3,
    help="BFGS gradient tolerance (default: 1e-3).")
parser.add_argument("--volume-target", type=float, default=0.3,
    help="Target volume of the Boozer surface (default: 0.3).")
parser.add_argument(
    "--output-root",
    type=str,
    default="../single_stage_true_epsilon_sequential_bigger_step",
    help=(
        "Directory under which <eq_name>/iota<X>_fcp<Y>kA_vt<Z>/mpol<M>_ntor<N>/ outputs "
        "are written; also where lower-resolution siblings are found for auto-resume "
        "(default: ../single_stage_true_epsilon_sequential_bigger_step)."
    ),
)
parser.add_argument(
    "--new",
    action="store_true",
    default=False,
    help=(
        "Start from scratch: do not resume from an existing output directory; "
        "reset iteration history (and log) for this run. Default is to continue "
        "for another --maxiter iterations when the same output dir already exists."
    ),
)
args, _ = parser.parse_known_args()

# Unpack
INIT_DIR       = args.init_dir
IOTA_TARGET    = args.iota_target
FCP_THRESHOLD  = args.f_cp_threshold
FB_THRESHOLD   = args.f_b_threshold
IOTA_THRESHOLD = args.iota_threshold
mpol           = args.mpol
ntor           = args.ntor
MAXITER        = args.maxiter
GTOL           = args.gtol
VOL_TARGET     = args.volume_target
OUTPUT_ROOT    = args.output_root
START_FRESH    = args.new

# Fixed internal parameters
BOOZER_CW        = 1.0    # BoozerSurface least-squares constraint weight
PENALTY_WEIGHT   = 100.0 # single knob: how aggressively constraints are enforced vs QS objective
CURRENT_P_NORM   = 20.0   # p-norm exponent (smooth max-current proxy)

plot_config = PlotConfig(
    dpi=100, titlefontsize=16, axisfontsize=16,
    legendfontsize=14, ticklabelfontsize=14, cbarfontsize=16,
)


def plot_objective_vs_iterations(out_dir, config):
    """
    Plot total objective J and each contribution vs accepted iteration using
    iterations.json, plus ||∇J|| on the lower panel (same layout as
    single_stage_epsilon_constraint.py).

    J = J_nonQS + J_boozer_contrib + J_iota_contrib + J_current_contrib
    where the *_contrib rows include PENALTY_WEIGHT (see callback).
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
# LOAD DATA AND INITIALIZE
# ==============================================================================
# We always need eq_name/eq_dir/VV_*/etc., which live in the stage-2 results.json.
# When an earlier (lower-resolution) single-stage run already exists for the same
# (iota, f_cp, volume) target, we prefer continuing from it (auto-resume).
stage2_results = load(os.path.join(INIT_DIR, "results.json"))
eq_name = stage2_results["eq_name"]

# Output directory:  <output-root>/<eq>/iota<X>_fcp<Y>kA_vt<Z>/mpol_ntor/
TARGET_PARENT = os.path.join(
    OUTPUT_ROOT,
    eq_name,
    f"iota{IOTA_TARGET:g}_fcp{FCP_THRESHOLD / 1e3:g}kA_vt{VOL_TARGET:g}",
)
OUT_DIR = os.path.join(TARGET_PARENT, f"mpol{mpol}_ntor{ntor}")
os.makedirs(OUT_DIR, exist_ok=True)


def _out_dir_has_checkpoint(out_dir):
    """Enough state to resume the same-resolution run from out_dir."""
    needed = ["bs_opt.json", "surf_opt.json", "results.json"]
    return all(os.path.isfile(os.path.join(out_dir, f)) for f in needed)


def _next_iteration_index_from_history(out_dir):
    """1-based index for the next callback row (after existing iterations.json)."""
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


def _find_best_prior_run(parent, max_mpol, max_ntor):
    """Find the highest-resolution completed mpolM_ntorN sibling with M<=max_mpol,
    N<=max_ntor, (M,N) != (max_mpol,max_ntor), containing bs_opt.json + surf_opt.json +
    results.json. Returns its path, or None."""
    if not os.path.isdir(parent):
        return None
    pat = re.compile(r"^mpol(\d+)_ntor(\d+)$")
    candidates = []
    for name in os.listdir(parent):
        m = pat.match(name)
        if not m:
            continue
        mp, nt = int(m.group(1)), int(m.group(2))
        if (mp, nt) == (max_mpol, max_ntor):
            continue
        if mp > max_mpol or nt > max_ntor:
            continue
        d = os.path.join(parent, name)
        needed = ["bs_opt.json", "surf_opt.json", "results.json"]
        if not all(os.path.exists(os.path.join(d, f)) for f in needed):
            continue
        candidates.append((mp, nt, d))
    if not candidates:
        return None
    # Pick highest (mpol, ntor).
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


AUTO_RESUME_DIR = _find_best_prior_run(TARGET_PARENT, mpol, ntor)

EXACT_RESUME = False
LOG_MODE = "a"

if START_FRESH:
    if AUTO_RESUME_DIR is not None:
        LOAD_DIR = AUTO_RESUME_DIR
        print(f"--new: initializing from lower-resolution run (not existing OUT_DIR): {LOAD_DIR}")
    else:
        LOAD_DIR = INIT_DIR
        print(f"--new: initializing from stage-2: {LOAD_DIR}")
    it_path = os.path.join(OUT_DIR, "iterations.json")
    if os.path.isfile(it_path):
        os.remove(it_path)
    LOG_MODE = "w"
elif _out_dir_has_checkpoint(OUT_DIR):
    LOAD_DIR = OUT_DIR
    EXACT_RESUME = True
    print(
        f"Continuing existing run in {OUT_DIR} for up to {MAXITER} more BFGS iterations "
        f"(same iota / f_CP / volume / mpol / ntor)."
    )
elif AUTO_RESUME_DIR is not None:
    LOAD_DIR = AUTO_RESUME_DIR
    print(f"Auto-resuming from lower-resolution run: {LOAD_DIR}")
else:
    LOAD_DIR = INIT_DIR
    print(f"No prior single-stage run found; initializing from stage-2: {LOAD_DIR}")

# The stage-2 results.json carries metadata (eq_name/eq_dir/VV_*/ntf/surf_nfp/...).
# Previous single-stage results.json also carry these fields (see save block below),
# so either source works. Use whichever matches LOAD_DIR so we stay consistent.
results = load(os.path.join(LOAD_DIR, "results.json"))

print(f"Starting true epsilon-constraint optimization: {datetime.now():%Y-%m-%d %H:%M:%S}")

# ---- Coils ----
bs = load(os.path.join(LOAD_DIR, "bs_opt.json"))

# ---- Surface (stage-2 VMEC or previous single-stage) ----
surf_opt_path = os.path.join(LOAD_DIR, "surf_opt.json")
if os.path.exists(surf_opt_path):
    surf = load(surf_opt_path)
    nPhi = surf.quadpoints_phi.size
    nTheta = surf.quadpoints_theta.size
else:
    nPhi, nTheta = 128, 64
    eq_path = os.path.join(results["eq_dir"], results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_path, s=results["surf_s"], range="half period", nphi=nPhi, ntheta=nTheta,
    )
    surf.set_dofs(results["surf_dof_scale"] * surf.get_dofs())

# ---- Split coils ----
num_tf = results["ntf"] * 2 * results["surf_nfp"]
coils = bs.coils
tf_coils = coils[:num_tf]
dipole_coils = coils[num_tf:]

for c in dipole_coils:
    c.curve.fix_all()
for c in tf_coils:
    c.current.fix_all()

# ---- Vacuum vessel (plots only) ----
VV = SurfaceRZFourier(nfp=results["surf_nfp"])
VV.set_rc(0, 0, results["VV_R0"])
VV.set_rc(1, 0, results["VV_a"])
VV.set_zs(1, 0, results["VV_b"])

# ---- Boozer surface ----
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
boozer_surface = initialize_boozer_surface(
    surf, mpol, ntor, bs, VOL_TARGET, BOOZER_CW, IOTA_TARGET, G0,
)
print(f"Initial Boozer surface volume: {boozer_surface.surface.volume():.4f}")

# ==============================================================================
# OBJECTIVE AND CONSTRAINTS
# ==============================================================================
bs_obj = BiotSavart(coils)

# PRIMARY OBJECTIVE: nonQS ratio — to be minimized
JnonQSRatio = sum([NonQuasiSymmetricRatio(boozer_surface, bs_obj)])

# Boozer residual (field quality)
JBoozerResidual = sum([BoozerResidual(boozer_surface, bs_obj)])

# CONSTRAINT 1: Boozer residual <= f_b
# Each guard is normalized by threshold² so penalty = 1.0 when violated by 100%
# of the threshold.  PENALTY_WEIGHT is the single knob controlling enforcement.
JBoozerGuard = (1.0 / FB_THRESHOLD**2) * QuadraticPenalty(
    JBoozerResidual, FB_THRESHOLD, f="max",
)

# CONSTRAINT 2: iota near target (two-sided)
iota = Iotas(boozer_surface)
Jiota = (1.0 / IOTA_THRESHOLD**2) * QuadraticPenalty(iota, IOTA_TARGET)

# CONSTRAINT 3: current p-norm <= f_CP
Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=CURRENT_P_NORM)
JCurrentGuard = (1.0 / FCP_THRESHOLD**2) * QuadraticPenalty(
    Jcurrent, FCP_THRESHOLD, f="max",
)

# TOTAL: min QS  +  penalty * (constraints)
JF = JnonQSRatio + PENALTY_WEIGHT * (JBoozerGuard + Jiota + JCurrentGuard)

# Print initial state
print(f"Initial nonQS ratio:      {JnonQSRatio.J():.6e}")
print(f"Initial Boozer residual:  {JBoozerResidual.J():.6e}  (threshold {FB_THRESHOLD:.1e})")
print(f"Initial iota:             {iota.J():.6f}  (target {IOTA_TARGET:g} ± {IOTA_THRESHOLD:g})")
print(f"Initial current p-norm:   {Jcurrent.J():.1f} A  (threshold {FCP_THRESHOLD:.0f} A)")
print(f"Penalty weight:           {PENALTY_WEIGHT}")
print(f"# TF coils: {len(tf_coils)},  # dipole coils: {len(dipole_coils)}")

# ==============================================================================
# OPTIMIZER
# ==============================================================================
x0 = JF.x.copy()
_next_it = _next_iteration_index_from_history(OUT_DIR) if EXACT_RESUME else 1
run_dict = {
    "sdofs": boozer_surface.surface.x.copy(),
    "iota": boozer_surface.res["iota"],
    "G": boozer_surface.res["G"],
    "J": JF.J(),
    "dJ": JF.dJ().copy(),
    "it": _next_it,
    "lscount": 0,
    "x_prev": x0.copy(),
    "failed_boozer_solves": 0,
}

# Redirect stdout/stderr to log file
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
log_path = os.path.join(OUT_DIR, "log.txt")
log_file = open(log_path, LOG_MODE, buffering=1)
sys.stdout = log_file
sys.stderr = log_file

start_time = time.time()
print(f"\n{'=' * 70}")
print(f"True Epsilon-Constraint Optimization — {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"{'=' * 70}")
print(f"Output root: {OUTPUT_ROOT}")
print(f"Output:     {OUT_DIR}")
print(f"init-dir:   {INIT_DIR}")
print(f"iota_target={IOTA_TARGET:g},  f_b={FB_THRESHOLD:.1e},  f_CP={FCP_THRESHOLD:.0f} A")
print(f"mpol={mpol}, ntor={ntor}, maxiter={MAXITER}")
print(f"# TF: {len(tf_coils)},  # dipole: {len(dipole_coils)}")
print(f"Objective: nonQS + {PENALTY_WEIGHT} * (Boozer_guard + iota_penalty + current_guard)\n")
if EXACT_RESUME:
    print(f"Resuming from iteration counter it={run_dict['it']} (appending to iterations.json).\n")
else:
    # Initial diagnostic plots (skip on exact resume to avoid overwriting first-run plots)
    plot_cross_section(
        boozer_surface.surface, VV, OUT_DIR, "initial", plot_config,
        base_dipole_coils=dipole_coils,
    )


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
    res = boozer_surface.run_code(run_dict["iota"], run_dict["G"])

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
        f"  step={dx:.2e}  J={J:.6e}  ||∇J||={np.linalg.norm(dJ):.6e}  "
        f"QS={JnonQSRatio.J():.6e}  fb={JBoozerResidual.J():.6e}  "
        f"ι={iota.J():.4f}  Imax={max_I:.0f}A"
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

    # Console diagnostics
    fb_flag = "  [!>fb]" if fb_val > FB_THRESHOLD else ""
    cp_flag = "  [!>fcp]" if cp_val > FCP_THRESHOLD else ""
    print(f"\n")
    print(f"{'=' * 60}")
    print(f"ITER {run_dict['it']:3d}  J={J:.6e}  ||∇J||={np.linalg.norm(grad):.6e}")
    print(f"  nonQS={qs_val:.6e}")
    print(f"  Boozer={fb_val:.6e}{fb_flag}")
    print(f"  iota={iota_val:.4f} (target {IOTA_TARGET:g})")
    print(f"  ⟨|B·n|⟩={BdotN:.6e}")
    print(f"  I_pnorm={cp_val:.0f}A (limit {FCP_THRESHOLD:.0f}A){cp_flag}")
    print(f"  I_max={max_I:.0f}A")
    print(f"  volume={vol:.4f}")
    print(f"\n")

    # ---- Iteration history JSON ----
    history_path = os.path.join(OUT_DIR, "iterations.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {
            "config": {
                "iota_target": IOTA_TARGET,
                "f_b_threshold": FB_THRESHOLD,
                "f_cp_threshold": FCP_THRESHOLD,
                "iota_threshold": IOTA_THRESHOLD,
                "penalty_weight": PENALTY_WEIGHT,
                "mpol": mpol, "ntor": ntor,
                "maxiter": MAXITER,
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

    # ---- Partial saves (survive early termination) ----
    bs.save(os.path.join(OUT_DIR, "bs_opt.json"))
    boozer_surface.surface.save(os.path.join(OUT_DIR, "surf_opt.json"))

    partial_results = {
        "graph": {"init_dir": INIT_DIR, "load_dir": LOAD_DIR},
        "method": "true_epsilon_constraint",
        "mpol": mpol, "ntor": ntor,
        "iota_target": IOTA_TARGET,
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
        "eq_name": results["eq_name"],
        "eq_dir": results["eq_dir"],
        "surf_nfp": results["surf_nfp"],
        "surf_s": results["surf_s"],
        "VV_R0": results["VV_R0"],
        "VV_a": results["VV_a"],
        "VV_b": results["VV_b"],
        "ntf": results["ntf"],
    }
    save(partial_results, os.path.join(OUT_DIR, "results.json"))

    run_dict["it"] += 1


# ---- Run L-BFGS ----
res = scipy_minimize(
    fun, x0, jac=True, method="BFGS", callback=callback,
    options={"maxiter": MAXITER, "gtol": GTOL},
)
print(f"\nOptimizer: {res.message}")

# ==============================================================================
# SAVE FINAL OUTPUTS
# ==============================================================================
# Coils
coils_to_vtk(coils, filename=os.path.join(OUT_DIR, "coils_opt"), close=True)
bs.save(os.path.join(OUT_DIR, "bs_opt.json"))
VV.to_vtk(os.path.join(OUT_DIR, "vacuum_vessel"))

# Surface with B_N/B overlay
bs.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
B = bs.B().reshape((nPhi, nTheta, 3))
modB = np.sqrt(np.sum(B**2, axis=2))[:, :, None]
BdotN_surf = np.sum(B * boozer_surface.surface.unitnormal(), axis=2)[:, :, None]
pointData = {"B_N/B": BdotN_surf / modB}
boozer_surface.surface.to_vtk(os.path.join(OUT_DIR, "surf_opt"), extra_data=pointData)
boozer_surface.surface.save(os.path.join(OUT_DIR, "surf_opt.json"))

# Final diagnostics
max_I = float(np.max([abs(c.current.get_value()) for c in dipole_coils]))
final_qs = float(JnonQSRatio.J())
final_fb = float(JBoozerResidual.J())
final_iota = float(iota.J())
final_vol = float(boozer_surface.surface.volume())
final_cp = float(Jcurrent.J())

print(f"\n{'=' * 70}")
print(f"FINAL RESULTS")
print(f"{'=' * 70}")
print(f"  nonQS ratio:      {final_qs:.6e}")
print(f"  Boozer residual:  {final_fb:.6e}  (threshold {FB_THRESHOLD:.1e})")
print(f"  Iota:             {final_iota:.6f}  (target {IOTA_TARGET:g})")
print(f"  Max current:      {max_I:.0f} A  (threshold {FCP_THRESHOLD:.0f} A)")
print(f"  Current p-norm:   {final_cp:.0f} A")
print(f"  Volume:           {final_vol:.4f}")

# Diagnostic plots
plot_relBfinal_norm_modB(bs, boozer_surface.surface, OUT_DIR, "optimized", plot_config)
plot_cross_section(
    boozer_surface.surface, VV, OUT_DIR, "optimized", plot_config,
    base_dipole_coils=dipole_coils,
)
plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, OUT_DIR, "optimized", plot_config)
plot_objective_vs_iterations(OUT_DIR, plot_config)

# Results JSON
results_output = {
    "graph": {"init_dir": INIT_DIR, "load_dir": LOAD_DIR},
    "method": "true_epsilon_constraint",

    # Configuration
    "mpol": mpol,
    "ntor": ntor,
    "maxiter": MAXITER,
    "iota_target": IOTA_TARGET,
    "f_b_threshold": FB_THRESHOLD,
    "f_cp_threshold": FCP_THRESHOLD,
    "iota_threshold": IOTA_THRESHOLD,
    "penalty_weight": PENALTY_WEIGHT,
    "current_pnorm_p": CURRENT_P_NORM,
    "gtol": GTOL,

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

    # Inherited from upstream (needed for downstream runs)
    "eq_name": results["eq_name"],
    "eq_dir": results["eq_dir"],
    "surf_nfp": results["surf_nfp"],
    "surf_s": results["surf_s"],
    "VV_R0": results["VV_R0"],
    "VV_a": results["VV_a"],
    "VV_b": results["VV_b"],
    "ntf": results["ntf"],
}
if "surf_dof_scale" in results:
    results_output["surf_dof_scale"] = results["surf_dof_scale"]

save(results_output, os.path.join(OUT_DIR, "results.json"))
print(f"\nResults saved to {os.path.join(OUT_DIR, 'results.json')}")

elapsed = time.time() - start_time
print(f"Wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")

# Restore streams
log_file.close()
sys.stdout = _orig_stdout
sys.stderr = _orig_stderr
print(f"Optimization complete. Results in {OUT_DIR}")
