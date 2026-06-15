"""
Single-stage optimization: minimize nonQS ratio + w_CP * (Jcurrent / I0),
with quadratic guards on the Boozer residual and |iota - iota_target|.

Objective (per stage):
    J = J_nonQS  +  (w_CP / I0) * J_current
          +  PENALTY_WEIGHT * ( (1/f_b^2) * QuadPen(J_boozer, f_b, f="max")
                              + (1/eps_i^2) * QuadPen(iota, iota_target) )

where J_current = CurrentPenalty([dipole currents], p=CURRENT_P_NORM=20) and
I0 is its value at the very start of the first stage (i.e. normalization from
the initial configuration), so --current-weight-schedule values are
dimensionless and consistent across runs.

Key differences from single_stage_true_epsilon.py:
  - No threshold on the dipole current; w_CP multiplies a dimensionless
    current term in the objective.
  - --current-weight-schedule steps the optimizer through an ordered
    list of w_CP values, warm-starting each stage from the previous one.

Directory layout (stage/resolution aware, same spirit as
single_stage_epsilon_constraint.py combined with the auto-resume of
single_stage_true_epsilon.py):

  <SCAN_ROOT>/<eq_name>/iota{I}_vt{V}/stage{NN}_cw{W}/mpol{M}_ntor{N}/

Resume behavior:
  - Stages whose results.json reports optimization_success=True are SKIPPED.
    The stage's saved bs/surf is loaded to warm-start later stages.
  - For the first non-completed stage, we warm-start from (in priority order):
        1. this stage's own highest-resolution completed sibling with
           (mpol' <= mpol, ntor' <= ntor), (mpol', ntor') != (mpol, ntor);
        2. the last completed earlier stage (any resolution);
        3. --init-dir (stage 2 output).
  - If the current stage/resolution folder already has a checkpoint but was
    not finished, we EXACT-resume it for another --maxiter BFGS iterations
    (appending iterations.json).
  - --new discards iterations.json and the log for the current stage's
    mpol/ntor folder and re-initializes via the warm-start logic above.

Usage:
  python single_stage_qs_cp.py --init-dir /path/to/stage2_results \\
      --iota-target 0.15 --current-weight-schedule 0.01,0.1,1 \\
      --mpol 6 --ntor 6 --maxiter 200
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
# CLI
# ==============================================================================
parser = argparse.ArgumentParser(
    description=(
        "Single-stage QS + w_CP*current objective with Boozer/iota guards and "
        "current-weight continuation."
    )
)
parser.add_argument("--init-dir", type=str, required=True,
    help="Directory with bs_opt.json, results.json, and optionally surf_opt.json.")
parser.add_argument("--iota-target", type=float, required=True,
    help="Target iota value on the Boozer surface.")
parser.add_argument("--current-weight-schedule", type=str, default="0.01,0.1,1",
    help="Comma-separated list of current weights w_CP for continuation "
         "(default: '0.01,0.1,1'). Each value becomes one stage.")
parser.add_argument("--f-b-threshold", type=float, default=1e-4,
    help="Boozer residual guard threshold f_b (default: 1e-4).")
parser.add_argument("--iota-threshold", type=float, default=0.0025,
    help="Absolute iota scale used to normalize the iota penalty "
         "(default: 0.0025). Note: the guard is two-sided (QuadraticPenalty "
         "to target, not a deadband), matching single_stage_true_epsilon.py.")
parser.add_argument("--mpol", type=int, default=6,
    help="Boozer surface poloidal resolution (default: 6).")
parser.add_argument("--ntor", type=int, default=6,
    help="Boozer surface toroidal resolution (default: 6).")
parser.add_argument("--maxiter", type=int, default=200,
    help="Max BFGS iterations per stage (default: 200).")
parser.add_argument("--volume-target", type=float, default=0.3,
    help="Target volume of the Boozer surface (default: 0.3).")
parser.add_argument("--scan-root", type=str, default="../single_stage_qs_cp",
    help="Root directory for this scan (default: '../single_stage_qs_cp').")
parser.add_argument(
    "--new",
    action="store_true",
    default=False,
    help=(
        "Start the CURRENT stage's mpol/ntor folder from scratch: clear its "
        "iterations.json and log. Earlier completed stages are still skipped. "
        "Default is to continue incomplete stages for another --maxiter "
        "BFGS iterations."
    ),
)
args, _ = parser.parse_known_args()

# Unpack
INIT_DIR       = args.init_dir
IOTA_TARGET    = args.iota_target
FB_THRESHOLD   = args.f_b_threshold
IOTA_THRESHOLD = args.iota_threshold
mpol           = args.mpol
ntor           = args.ntor
MAXITER        = args.maxiter
VOL_TARGET     = args.volume_target
SCAN_ROOT      = args.scan_root
START_FRESH    = args.new
CURRENT_WEIGHT_SCHEDULE = [
    float(x) for x in args.current_weight_schedule.split(",") if x.strip() != ""
]
if not CURRENT_WEIGHT_SCHEDULE:
    raise ValueError("--current-weight-schedule must contain at least one value.")

# Fixed internal parameters
BOOZER_CW      = 1.0     # BoozerSurface least-squares constraint weight
PENALTY_WEIGHT = 100.0   # single knob: how aggressively guards are enforced vs objective
CURRENT_P_NORM = 20.0    # p-norm exponent (smooth max-current proxy)
GTOL           = 1e-3

plot_config = PlotConfig(
    dpi=100, titlefontsize=16, axisfontsize=16,
    legendfontsize=14, ticklabelfontsize=14, cbarfontsize=16,
)


# ==============================================================================
# HELPERS
# ==============================================================================
def _has_checkpoint(d):
    """Enough saved state in d to warm-start or exact-resume from it."""
    needed = ["bs_opt.json", "surf_opt.json", "results.json"]
    return all(os.path.isfile(os.path.join(d, f)) for f in needed)


def _is_completed(d):
    """A stage/resolution folder is 'completed' iff it has a checkpoint AND
    results.json reports optimization_success == True (i.e. not a partial
    snapshot from the callback)."""
    if not _has_checkpoint(d):
        return False
    try:
        r = load(os.path.join(d, "results.json"))
    except Exception:
        return False
    # callback-written partials explicitly set optimization_success to None.
    return bool(r.get("optimization_success", False))


def _next_iteration_index_from_history(d):
    """1-based index for the next callback row, based on iterations.json in d."""
    path = os.path.join(d, "iterations.json")
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


def _find_best_prior_resolution(stage_dir, max_mpol, max_ntor):
    """Highest-resolution completed sibling inside stage_dir with
    (mpol'<=max_mpol, ntor'<=max_ntor) and (mpol',ntor') != (max_mpol,max_ntor).
    Returns its path, or None."""
    if not os.path.isdir(stage_dir):
        return None
    pat = re.compile(r"^mpol(\d+)_ntor(\d+)$")
    candidates = []
    for name in os.listdir(stage_dir):
        m = pat.match(name)
        if not m:
            continue
        mp, nt = int(m.group(1)), int(m.group(2))
        if (mp, nt) == (max_mpol, max_ntor):
            continue
        if mp > max_mpol or nt > max_ntor:
            continue
        d = os.path.join(stage_dir, name)
        if not _is_completed(d):
            continue
        candidates.append((mp, nt, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _find_best_resolution_any(stage_dir, max_mpol, max_ntor):
    """Like _find_best_prior_resolution but also includes the (max_mpol,max_ntor)
    folder itself (used when locating a completed earlier-stage warm start)."""
    if not os.path.isdir(stage_dir):
        return None
    pat = re.compile(r"^mpol(\d+)_ntor(\d+)$")
    candidates = []
    for name in os.listdir(stage_dir):
        m = pat.match(name)
        if not m:
            continue
        mp, nt = int(m.group(1)), int(m.group(2))
        if mp > max_mpol or nt > max_ntor:
            continue
        d = os.path.join(stage_dir, name)
        if not _is_completed(d):
            continue
        candidates.append((mp, nt, d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def plot_objective_vs_iterations(out_dir, I0, current_weight, config):
    """Plot J and its pieces vs iteration (reads iterations.json written by callback).

    J = J_nonQS + (w_CP/I0) * J_current + w_pen*(J_boozer_guard + J_iota_guard)
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
    J_qs = np.array([r["nonQS_ratio"] for r in rows], dtype=float)
    J_curr_scaled = np.array(
        [float(r.get("J_current_contrib", np.nan)) for r in rows], dtype=float)
    J_b = np.array(
        [float(r.get("J_boozer_contrib", np.nan)) for r in rows], dtype=float)
    J_i = np.array(
        [float(r.get("J_iota_contrib", np.nan)) for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8), constrained_layout=True,
    )
    ax1.plot(it, J_tot, color="k", lw=2.0, label=r"$J$ (total)")
    ax1.plot(it, J_qs, label=r"$J_{\mathrm{nonQS}}$")
    ax1.plot(it, J_curr_scaled, label=r"$(w_{CP}/I_0)\,J_{\mathrm{current}}$")
    ax1.plot(it, J_b, label=r"$w_{\mathrm{pen}}\,J_{\mathrm{Boozer\,guard}}$")
    ax1.plot(it, J_i, label=r"$w_{\mathrm{pen}}\,J_{\iota}$")
    ax1.set_ylabel("Contribution to $J$", fontsize=config.axisfontsize)
    ax1.set_title(
        f"Objective and components vs iteration (w_CP={current_weight:g})",
        fontsize=config.titlefontsize,
    )
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
# LAYOUT AND WARM-START RESOLUTION
# ==============================================================================
# Stage-2 results (eq_name/eq_dir/VV_*/ntf/surf_nfp/...). These fields are also
# written through to every results.json we save, so later continuations can
# read them from a prior stage folder too.
stage2_results = load(os.path.join(INIT_DIR, "results.json"))
eq_name = stage2_results["eq_name"]

IOTA_LEAF = f"iota{IOTA_TARGET:g}_vt{VOL_TARGET:g}"
RUN_ROOT = os.path.join(SCAN_ROOT, eq_name, IOTA_LEAF)
os.makedirs(RUN_ROOT, exist_ok=True)


def _stage_dir(stage_idx, cw):
    return os.path.join(RUN_ROOT, f"stage{stage_idx:02d}_cw{cw:g}")


def _resolution_dir(stage_idx, cw):
    return os.path.join(_stage_dir(stage_idx, cw), f"mpol{mpol}_ntor{ntor}")


# Pre-compute, for each stage: whether it's already completed at this resolution,
# and where we would warm-start it from.
stage_status = []  # list of dicts per stage
last_completed_source = None  # path to use as warm start if no per-stage fallback

for stage_idx, cw in enumerate(CURRENT_WEIGHT_SCHEDULE):
    s_dir = _stage_dir(stage_idx, cw)
    r_dir = _resolution_dir(stage_idx, cw)

    completed_here = _is_completed(r_dir)
    same_stage_prior_res = _find_best_prior_resolution(s_dir, mpol, ntor)

    stage_status.append({
        "idx": stage_idx,
        "cw": float(cw),
        "stage_dir": s_dir,
        "res_dir": r_dir,
        "completed": completed_here,
        "same_stage_prior_res": same_stage_prior_res,
    })

    if completed_here:
        last_completed_source = r_dir

# The first stage that isn't already completed at this resolution is where we
# actually start running.
try:
    first_run_idx = next(i for i, s in enumerate(stage_status) if not s["completed"])
except StopIteration:
    first_run_idx = None

print(f"Scan root:        {SCAN_ROOT}")
print(f"Run root:         {RUN_ROOT}")
print(f"Schedule:         {CURRENT_WEIGHT_SCHEDULE}")
print(f"Resolution:       mpol={mpol}, ntor={ntor}")
for s in stage_status:
    if s["completed"]:
        status = "COMPLETED"
    elif first_run_idx is not None and s["idx"] == first_run_idx and \
            _has_checkpoint(s["res_dir"]) and not START_FRESH:
        status = "EXACT-RESUME"
    else:
        status = "WILL RUN"
    extra = ""
    if s["same_stage_prior_res"] and not s["completed"]:
        extra = f"  (same-stage prior-res: {s['same_stage_prior_res']})"
    print(f"  stage{s['idx']:02d}_cw{s['cw']:g}: {status}{extra}")

if first_run_idx is None:
    print("All scheduled stages already completed at this resolution. Nothing to do.")
    print("Use --new to re-run the last stage, or raise --mpol/--ntor.")
    sys.exit(0)

# ==============================================================================
# LOAD INITIAL STATE FOR THE FIRST STAGE WE WILL ACTUALLY RUN
# ==============================================================================
first = stage_status[first_run_idx]

LOAD_DIR = None
exact_resume_first = False
log_mode_first = "a"

# Priority 1: exact resume at this stage/resolution if a checkpoint exists and
# --new wasn't passed.
if not START_FRESH and _has_checkpoint(first["res_dir"]):
    LOAD_DIR = first["res_dir"]
    exact_resume_first = True
    print(f"Exact-resume stage{first['idx']:02d} at this resolution: {LOAD_DIR}")
else:
    # Priority 2: same-stage, lower-resolution completed sibling.
    if first["same_stage_prior_res"] is not None:
        LOAD_DIR = first["same_stage_prior_res"]
        print(f"Warm-starting stage{first['idx']:02d} from lower-resolution sibling: {LOAD_DIR}")
    else:
        # Priority 3: last completed earlier stage (at any resolution <= this one).
        for j in range(first_run_idx - 1, -1, -1):
            prev_src = _find_best_resolution_any(stage_status[j]["stage_dir"], mpol, ntor)
            if prev_src is not None:
                LOAD_DIR = prev_src
                print(f"Warm-starting stage{first['idx']:02d} from earlier stage{j:02d}: {LOAD_DIR}")
                break
        # Priority 4: stage-2 (--init-dir).
        if LOAD_DIR is None:
            LOAD_DIR = INIT_DIR
            print(f"Warm-starting stage{first['idx']:02d} from --init-dir: {LOAD_DIR}")

    # --new clears iterations.json/log for this exact folder only; warm-start
    # sources are untouched.
    if START_FRESH:
        it_path = os.path.join(first["res_dir"], "iterations.json")
        if os.path.isfile(it_path):
            os.remove(it_path)
        log_mode_first = "w"

os.makedirs(first["res_dir"], exist_ok=True)

# Read whichever results.json we're initializing from (carries metadata).
init_results = load(os.path.join(LOAD_DIR, "results.json"))

print(f"Starting single-stage QS+CP optimization: {datetime.now():%Y-%m-%d %H:%M:%S}")

# ---- Coils ----
bs = load(os.path.join(LOAD_DIR, "bs_opt.json"))

# ---- Surface ----
surf_opt_path = os.path.join(LOAD_DIR, "surf_opt.json")
if os.path.exists(surf_opt_path):
    surf = load(surf_opt_path)
    nPhi = surf.quadpoints_phi.size
    nTheta = surf.quadpoints_theta.size
else:
    nPhi, nTheta = 128, 64
    eq_path = os.path.join(init_results["eq_dir"], init_results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_path, s=init_results["surf_s"], range="half period",
        nphi=nPhi, ntheta=nTheta,
    )
    surf.set_dofs(init_results["surf_dof_scale"] * surf.get_dofs())

# ---- Split coils ----
num_tf = init_results["ntf"] * 2 * init_results["surf_nfp"]
coils = bs.coils
tf_coils = coils[:num_tf]
dipole_coils = coils[num_tf:]

for c in dipole_coils:
    c.curve.fix_all()
for c in tf_coils:
    c.current.fix_all()

# ---- Vacuum vessel (plotting only) ----
VV = SurfaceRZFourier(nfp=init_results["surf_nfp"])
VV.set_rc(0, 0, init_results["VV_R0"])
VV.set_rc(1, 0, init_results["VV_a"])
VV.set_zs(1, 0, init_results["VV_b"])

# ---- Boozer surface ----
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))
boozer_surface = initialize_boozer_surface(
    surf, mpol, ntor, bs, VOL_TARGET, BOOZER_CW, IOTA_TARGET, G0,
)
print(f"Initial Boozer surface volume: {boozer_surface.surface.volume():.4f}")

# ==============================================================================
# OBJECTIVE COMPONENTS (shared across all stages)
# ==============================================================================
bs_obj = BiotSavart(coils)

JnonQSRatio = sum([NonQuasiSymmetricRatio(boozer_surface, bs_obj)])
JBoozerResidual = sum([BoozerResidual(boozer_surface, bs_obj)])

iota = Iotas(boozer_surface)

# Guards (same form as single_stage_true_epsilon.py):
# each guard is normalized by threshold^2 so it equals 1.0 when the deviation
# is exactly `threshold`. PENALTY_WEIGHT scales how aggressively they bite.
JBoozerGuard = (1.0 / FB_THRESHOLD**2) * QuadraticPenalty(
    JBoozerResidual, FB_THRESHOLD, f="max",
)
Jiota = (1.0 / IOTA_THRESHOLD**2) * QuadraticPenalty(iota, IOTA_TARGET)

# Current p-norm (normalized): we divide by I0 (initial value at the first
# stage we actually run) so that w_CP is dimensionless and consistent between
# runs even when the starting configuration changes.
Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=CURRENT_P_NORM)
I0 = float(Jcurrent.J())
if not np.isfinite(I0) or I0 <= 0:
    raise ValueError(f"Invalid initial current p-norm for normalization: I0={I0}")

# Build a template JF; per stage we rebuild JF with the correct CURRENT_WEIGHT.
CURRENT_WEIGHT = CURRENT_WEIGHT_SCHEDULE[first_run_idx]
JF = JnonQSRatio + (CURRENT_WEIGHT / I0) * Jcurrent + PENALTY_WEIGHT * (
    JBoozerGuard + Jiota
)

print(f"Initial nonQS ratio:      {JnonQSRatio.J():.6e}")
print(f"Initial Boozer residual:  {JBoozerResidual.J():.6e}  (threshold {FB_THRESHOLD:.1e})")
print(f"Initial iota:             {iota.J():.6f}  (target {IOTA_TARGET:g} ± {IOTA_THRESHOLD:g})")
print(f"Initial current p-norm:   {Jcurrent.J():.1f} A   (I0={I0:.4e})")
print(f"Penalty weight:           {PENALTY_WEIGHT}")
print(f"# TF coils: {len(tf_coils)},  # dipole coils: {len(dipole_coils)}")


# ==============================================================================
# Per-stage optimization driver
# ==============================================================================
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr


def fun_factory(run_dict):
    """Build a fun(x) closure bound to the current stage's run_dict / JF."""
    def fun(x):
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
            ok = (
                boozer_surface.res["success"]
                and not boozer_surface.surface.is_self_intersecting()
            )
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
    return fun


def callback_factory(run_dict, OUT_DIR_ITER, cw):
    """Build a callback closure that writes iterations.json and partial
    results.json under OUT_DIR_ITER for the given stage."""
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
            bs.B().reshape((nphi_b, ntheta_b, 3)) * boozer_surface.surface.unitnormal(),
            axis=2,
        ))))
        vol = float(boozer_surface.surface.volume())

        fb_flag = "  [!>fb]" if fb_val > FB_THRESHOLD else ""
        print("\n")
        print(f"{'=' * 60}")
        print(f"ITER {run_dict['it']:3d}  J={J:.6e}  ||∇J||={np.linalg.norm(grad):.6e}")
        print(f"  nonQS={qs_val:.6e}")
        print(f"  Boozer={fb_val:.6e}{fb_flag}")
        print(f"  iota={iota_val:.4f} (target {IOTA_TARGET:g})")
        print(f"  ⟨|B·n|⟩={BdotN:.6e}")
        print(f"  I_pnorm={cp_val:.0f}A (I0={I0:.0f}A,  w_CP={cw:g})")
        print(f"  I_max={max_I:.0f}A")
        print(f"  volume={vol:.4f}")
        print("\n")

        history_path = os.path.join(OUT_DIR_ITER, "iterations.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = {
                "config": {
                    "iota_target": IOTA_TARGET,
                    "f_b_threshold": FB_THRESHOLD,
                    "iota_threshold": IOTA_THRESHOLD,
                    "penalty_weight": PENALTY_WEIGHT,
                    "current_weight": cw,
                    "current_weight_schedule": CURRENT_WEIGHT_SCHEDULE,
                    "stage_idx": run_dict["stage_idx"],
                    "I0": I0,
                    "current_pnorm_p": CURRENT_P_NORM,
                    "mpol": mpol, "ntor": ntor,
                    "maxiter": MAXITER,
                    "volume_target": VOL_TARGET,
                },
                "iterations": [],
            }
        jb_g = float(JBoozerGuard.J())
        ji_g = float(Jiota.J())
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
            # Pieces of J = J_nonQS + (cw/I0)*Jcurrent + w_pen*(JBoozerGuard + Jiota)
            "J_current_contrib": float((cw / I0) * cp_val),
            "J_boozer_contrib": float(PENALTY_WEIGHT * jb_g),
            "J_iota_contrib": float(PENALTY_WEIGHT * ji_g),
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Partial saves
        bs.save(os.path.join(OUT_DIR_ITER, "bs_opt.json"))
        boozer_surface.surface.save(os.path.join(OUT_DIR_ITER, "surf_opt.json"))

        partial_results = {
            "graph": {"init_dir": INIT_DIR, "load_dir": LOAD_DIR},
            "method": "qs_plus_cp_weighted",
            "stage_idx": run_dict["stage_idx"],
            "current_weight": cw,
            "current_weight_schedule": CURRENT_WEIGHT_SCHEDULE,
            "I0_current_pnorm": I0,
            "current_pnorm_p": CURRENT_P_NORM,
            "mpol": mpol, "ntor": ntor,
            "iota_target": IOTA_TARGET,
            "f_b_threshold": FB_THRESHOLD,
            "iota_threshold": IOTA_THRESHOLD,
            "penalty_weight": PENALTY_WEIGHT,
            "volume_target": VOL_TARGET,
            "optimization_success": None,
            "optimization_message": "partial snapshot from callback",
            "final_objective": float(J),
            "nonQS_ratio": qs_val,
            "boozer_residual": fb_val,
            "final_iota": iota_val,
            "current_pnorm": cp_val,
            "max_current": max_I,
            "# TF coils": len(tf_coils),
            "# dipole coils": len(dipole_coils),
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
            partial_results["surf_dof_scale"] = init_results["surf_dof_scale"]
        save(partial_results, os.path.join(OUT_DIR_ITER, "results.json"))

        run_dict["it"] += 1
    return callback


def run_stage(stage_idx, cw, exact_resume, log_mode):
    """Run one optimization stage with weight cw. Returns final x (warm-start seed)."""
    global JF

    OUT_DIR_ITER = _resolution_dir(stage_idx, cw)
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # Rebuild JF with this stage's weight so every Optimizable is reattached
    # to the same underlying graph of children.
    JF = JnonQSRatio + (cw / I0) * Jcurrent + PENALTY_WEIGHT * (JBoozerGuard + Jiota)

    x0 = JF.x.copy()
    next_it = _next_iteration_index_from_history(OUT_DIR_ITER) if exact_resume else 1

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
        "stage_idx": int(stage_idx),
    }

    log_path = os.path.join(OUT_DIR_ITER, "log.txt")
    log_file = open(log_path, log_mode, buffering=1)
    # Redirect this stage's stdout/stderr to its log
    sys.stdout = log_file
    sys.stderr = log_file

    start_time = time.time()
    print(f"\n{'=' * 70}")
    print(f"stage{stage_idx:02d}_cw{cw:g} — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{'=' * 70}")
    print(f"Output:     {OUT_DIR_ITER}")
    print(f"init-dir:   {INIT_DIR}")
    print(f"schedule:   {CURRENT_WEIGHT_SCHEDULE}")
    print(f"iota_target={IOTA_TARGET:g},  f_b={FB_THRESHOLD:.1e},  "
          f"eps_i={IOTA_THRESHOLD:g}")
    print(f"w_CP={cw:g} (normalized by I0={I0:.4e})")
    print(f"mpol={mpol}, ntor={ntor}, maxiter={MAXITER}")
    print(f"# TF: {len(tf_coils)},  # dipole: {len(dipole_coils)}")
    print(f"Objective: nonQS + (w_CP/I0)*Jcurrent + "
          f"{PENALTY_WEIGHT}*(Boozer_guard + iota_guard)\n")
    if exact_resume:
        print(f"Resuming from iteration counter it={run_dict['it']} "
              f"(appending to iterations.json).\n")
    else:
        plot_cross_section(
            boozer_surface.surface, VV, OUT_DIR_ITER, "initial", plot_config,
            base_dipole_coils=dipole_coils,
        )

    fun = fun_factory(run_dict)
    callback = callback_factory(run_dict, OUT_DIR_ITER, cw)

    res = scipy_minimize(
        fun, x0, jac=True, method="BFGS", callback=callback,
        options={"maxiter": MAXITER, "gtol": GTOL},
    )
    print(f"\nOptimizer: {res.message}")

    # ------------------ Save final outputs for this stage ------------------
    coils_to_vtk(coils, filename=os.path.join(OUT_DIR_ITER, "coils_opt"), close=True)
    bs.save(os.path.join(OUT_DIR_ITER, "bs_opt.json"))
    VV.to_vtk(os.path.join(OUT_DIR_ITER, "vacuum_vessel"))

    bs.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
    B = bs.B().reshape((nPhi, nTheta, 3))
    modB = np.sqrt(np.sum(B**2, axis=2))[:, :, None]
    BdotN_surf = np.sum(B * boozer_surface.surface.unitnormal(), axis=2)[:, :, None]
    pointData = {"B_N/B": BdotN_surf / modB}
    boozer_surface.surface.to_vtk(
        os.path.join(OUT_DIR_ITER, "surf_opt"), extra_data=pointData,
    )
    boozer_surface.surface.save(os.path.join(OUT_DIR_ITER, "surf_opt.json"))

    max_I = float(np.max([abs(c.current.get_value()) for c in dipole_coils]))
    final_qs = float(JnonQSRatio.J())
    final_fb = float(JBoozerResidual.J())
    final_iota = float(iota.J())
    final_vol = float(boozer_surface.surface.volume())
    final_cp = float(Jcurrent.J())

    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS (stage{stage_idx:02d}, w_CP={cw:g})")
    print(f"{'=' * 70}")
    print(f"  nonQS ratio:      {final_qs:.6e}")
    print(f"  Boozer residual:  {final_fb:.6e}  (threshold {FB_THRESHOLD:.1e})")
    print(f"  Iota:             {final_iota:.6f}  (target {IOTA_TARGET:g})")
    print(f"  Max current:      {max_I:.0f} A")
    print(f"  Current p-norm:   {final_cp:.0f} A  (I0={I0:.0f} A)")
    print(f"  Volume:           {final_vol:.4f}")

    plot_relBfinal_norm_modB(bs, boozer_surface.surface, OUT_DIR_ITER, "optimized", plot_config)
    plot_cross_section(
        boozer_surface.surface, VV, OUT_DIR_ITER, "optimized", plot_config,
        base_dipole_coils=dipole_coils,
    )
    plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, OUT_DIR_ITER, "optimized", plot_config)
    plot_objective_vs_iterations(OUT_DIR_ITER, I0, cw, plot_config)

    results_output = {
        "graph": {"init_dir": INIT_DIR, "load_dir": LOAD_DIR},
        "method": "qs_plus_cp_weighted",

        "stage_idx": int(stage_idx),
        "current_weight": float(cw),
        "current_weight_schedule": [float(v) for v in CURRENT_WEIGHT_SCHEDULE],
        "I0_current_pnorm": float(I0),
        "current_pnorm_p": float(CURRENT_P_NORM),

        "mpol": mpol,
        "ntor": ntor,
        "maxiter": MAXITER,
        "iota_target": IOTA_TARGET,
        "f_b_threshold": FB_THRESHOLD,
        "iota_threshold": IOTA_THRESHOLD,
        "penalty_weight": PENALTY_WEIGHT,
        "volume_target": VOL_TARGET,
        "gtol": GTOL,

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
        results_output["surf_dof_scale"] = init_results["surf_dof_scale"]
    save(results_output, os.path.join(OUT_DIR_ITER, "results.json"))
    print(f"\nResults saved to {os.path.join(OUT_DIR_ITER, 'results.json')}")

    elapsed = time.time() - start_time
    print(f"Stage wall time: {elapsed / 60:.1f} min ({elapsed:.0f} s)")

    log_file.close()
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    print(f"Finished stage{stage_idx:02d}_cw{cw:g} in {OUT_DIR_ITER}")

    return res.x.copy()


# ==============================================================================
# DRIVE THE SCHEDULE
# ==============================================================================
# First stage that actually runs: starts from the state we already loaded above
# (bs + surf + boozer_surface). For every subsequent stage we reuse the same
# in-memory state (warm-start) and just rebuild JF with the new weight.
for s in stage_status[:first_run_idx]:
    print(f"Skipping already-completed stage{s['idx']:02d}_cw{s['cw']:g} "
          f"(results.json reports success).")

for k in range(first_run_idx, len(stage_status)):
    s = stage_status[k]
    cw = s["cw"]
    if k == first_run_idx:
        er = exact_resume_first
        lm = log_mode_first
    else:
        # Subsequent stages are warm-started from the PREVIOUS stage's
        # in-memory state (the result of run_stage above). Any stale partial
        # checkpoint in this folder is discarded so we don't mix x histories.
        er = False
        lm = "w"
        it_path = os.path.join(s["res_dir"], "iterations.json")
        if os.path.isfile(it_path):
            os.remove(it_path)
    run_stage(s["idx"], cw, er, lm)

print(f"All scheduled stages completed. Run root: {RUN_ROOT}")
