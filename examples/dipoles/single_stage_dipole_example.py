import os
import io
import sys
import json
import argparse
import time
import numpy as np
from scipy.optimize import minimize

# SIMSOPT imports
from simsopt.geo import SurfaceRZFourier, curves_to_vtk, RotatedCurve
from simsopt.geo.surfaceobjectives import BoozerResidual, Iotas, NonQuasiSymmetricRatio
from simsopt.field import BiotSavart, Coil, Current, ScaledCurrent, coils_to_vtk, CurrentPenalty, coils_via_symmetries
from simsopt.objectives import QuadraticPenalty
from simsopt._core.optimizable import load, save
import matplotlib.pyplot as plt
from helper_functions import *
from boozer_functions import *

# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================
parser = argparse.ArgumentParser(
    description="Single-stage dipole optimization with configurable targets and weights."
)

parser.add_argument(
    "--init-dir",
    type=str,
    default="../outputs/20260225_scans/wout_nfp22ginsburg_000_000281/01_ntf4_diprad_0.05_VVa_0.2455263272670667_VV_R0_1.037468882271737_ellipticalVV",
    help="Directory containing results from previous optimization (bs_opt.json, surf_opt.json, results.json).",
)

parser.add_argument(
    "--iota-target",
    type=float,
    default=0.1,
    help="Target iota value on the Boozer surface.",
)
parser.add_argument(
    "--iota-weight",
    type=float,
    default=1e2,
    help="Weight for the iota penalty term.",
)
parser.add_argument(
    "--qs-weight",
    type=float,
    default=1e2,
    help="Weight for the quasi-symmetry (nonQS ratio) term.",
)
parser.add_argument(
    "--current-threshold",
    type=float,
    default=200000.0,
    help="Current threshold for the current penalty term [A].",
)
parser.add_argument(
    "--current-weight",
    type=float,
    default=1.0,
    help="Weight for the current penalty term.",
)

# Allow unknown args so this script can coexist with external launchers that add flags
_args, _unknown = parser.parse_known_args()

# Configuration parameters (can be overridden from the command line)
INIT_DIR = _args.init_dir
IOTA_TARGET = _args.iota_target
IOTA_WEIGHT = _args.iota_weight
QS_WEIGHT = _args.qs_weight
CURRENT_THRESHOLD = _args.current_threshold
CURRENT_WEIGHT = _args.current_weight

# Other parameters that are not set from the command line
CONSTRAINT_WEIGHT = 1.0
MAXITER = 75

# Create plot configuration
plot_config = PlotConfig(
    dpi=100,
    titlefontsize=16,
    axisfontsize=16,
    legendfontsize=14,
    ticklabelfontsize=14,
    cbarfontsize=16
)

def fun(x):
    """
    Objective function for L-BFGS-B optimization.

    Evaluates the total objective function and its gradient for a given set of
    degrees of freedom (coil parameters). Attempts to solve for a valid Boozer
    surface; if unsuccessful (solver failure or self-intersection), returns the
    last accepted objective value with negated gradient to reject the step.

    Args:
        x: Current degrees of freedom (coil parameters)

    Returns:
        J: Objective function value
        dJ: Gradient of objective function
    """
    dx = np.linalg.norm(x - run_dict['x_prev'])
    run_dict['x_prev'] = x.copy()
    print(f"Step size: {dx:.2e}")

    run_dict['lscount']+=1

    # initialize to last accepted surface values
    boozer_surface.surface.x = run_dict['sdofs']
    boozer_surface.res['iota'] = run_dict['iota']
    boozer_surface.res['G'] = run_dict['G']

    # Set new coil dofs
    JF.x = x

    # Run boozer surface
    res = boozer_surface.run_code(run_dict['iota'], run_dict['G'])

    # Check success
    try:
        success1 = boozer_surface.res['success']
        success2 = not boozer_surface.surface.is_self_intersecting()
    except Exception as e:
        print("Surface check failed:", e)
        success2 = False
    success = success1 and success2

    if success:
        # Reset failure counter on a successful Boozer solve
        run_dict['failed_boozer_solves'] = 0
        J = JF.J()
        dJ = JF.dJ()
    else:
        print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not success1:
            print("Boozer solver failed")
        if not success2:
            print("Surface is self-intersecting")

        # Increment counter of consecutive failed Boozer solves and abort if too many.
        run_dict['failed_boozer_solves'] += 1
        print(f"Consecutive failed Boozer solves: {run_dict['failed_boozer_solves']}")

        J = run_dict['J']
        dJ = -run_dict['dJ']
        boozer_surface.surface.x = run_dict['sdofs']
        boozer_surface.res['iota'] = run_dict['iota']
        boozer_surface.res['G'] = run_dict['G']

    print(f"Objective J: {J:.6e}, ||∇J||: {np.linalg.norm(dJ):.6e}")
    print(f"Individual scaled terms -- Boozer: {JBoozerResidual.J():.6e}, QS: {QS_WEIGHT * JnonQSRatio.J():.6e}, iota: {IOTA_WEIGHT * Jiota.J():.6e}, current: {CURRENT_WEIGHT * Jcurrent.J():.6e}")
    return J, dJ

def callback(x):
    """
    Callback function executed after each successful optimization iteration.

    Stores the accepted state (surface DOFs, iota, G), evaluates and prints
    detailed diagnostics for all objective function components, and logs the
    iteration summary to file. Used for monitoring optimization progress and
    recording convergence history.

    Args:
        x: Current degrees of freedom (coil parameters) from accepted step
    """
    # Update count for tracking
    run_dict['lscount'] = 0

    # Store last accepted state
    run_dict['sdofs'] = boozer_surface.surface.x.copy()
    run_dict['iota'] = boozer_surface.res['iota']
    run_dict['G'] = boozer_surface.res['G']
    run_dict['J'] = JF.J()
    run_dict['dJ'] = JF.dJ().copy()

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']

    J_QS = JnonQSRatio.J()
    dJ_QS = np.linalg.norm(JnonQSRatio.dJ())
    J_Boozer = JBoozerResidual.J()
    dJ_Boozer = np.linalg.norm(JBoozerResidual.dJ())
    J_iota = Jiota.J()
    dJ_iota = np.linalg.norm(Jiota.dJ())
    J_curr = Jcurrent.J()
    dJ_curr = np.linalg.norm(Jcurrent.dJ())

    iota_str = f"{iota.J():.4f}"
    volume_str = f"{boozer_surface.surface.volume():.4f}"

    nphi = boozer_surface.surface.quadpoints_phi.size
    ntheta = boozer_surface.surface.quadpoints_theta.size
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * boozer_surface.surface.unitnormal(), axis=2)))

    currents = np.array([abs(c.current.get_value()) for c in dipole_coils])
    num_over = np.sum(currents > CURRENT_THRESHOLD)
    max_current = np.max(currents)

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Current Penalty':{width}} = {J_curr:.6e} (dJ = {dJ_curr:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    print(f"{'Max current':{width}} = {max_current:.2f} A", file=buffer)
    print(f"{'# currents over threshold':{width}} = {num_over}", file=buffer)
    print("="*70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    # Save iteration diagnostics to JSON for postprocessing
    history_path = os.path.join(OUT_DIR_ITER, "iterations.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {
            "weights": {
                "QS_WEIGHT": QS_WEIGHT,
                "IOTA_WEIGHT": IOTA_WEIGHT,
                "CURRENT_WEIGHT": CURRENT_WEIGHT,
            },
            "targets": {
                "IOTA_TARGET": IOTA_TARGET,
                "CURRENT_THRESHOLD": CURRENT_THRESHOLD,
            },
            "tolerances": {
                "gtol": gtol_by_mpol.get(mpol),
            },
            "max_iterations": MAXITER,
            "iterations": [],
        }

    iter_record = {
        "iteration": int(run_dict["it"]),
        "J": float(J),
        "grad_norm": float(np.linalg.norm(grad)),
        "J_nonQS": float(J_QS),
        "J_Boozer": float(J_Boozer),
        "J_iota": float(J_iota),
        "J_current": float(J_curr),
        "J_nonQS_scaled": float(QS_WEIGHT * J_QS),
        "J_iota_scaled": float(IOTA_WEIGHT * J_iota),
        "J_current_scaled": float(CURRENT_WEIGHT * J_curr),
        "iota": float(iota.J()),
        "volume": float(boozer_surface.surface.volume()),
        "BdotN": float(BdotN),
        "max_current": float(max_current),
        "num_currents_over_threshold": int(num_over),
    }

    history["iterations"].append(iter_record)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save a partial results.json snapshot on every accepted iteration so that
    # runs that terminate early still leave usable metadata for postprocessing.
    partial_results = {
        "graph": {"init_dir": INIT_DIR},
        # Optimization configuration
        "mpol": mpol,
        "ntor": ntor,
        "maxiter": MAXITER,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "iota_target": IOTA_TARGET,
        "qs_weight": QS_WEIGHT,
        "iota_weight": IOTA_WEIGHT,
        "current_threshold": CURRENT_THRESHOLD,
        "current_weight": CURRENT_WEIGHT,
        # Convergence tolerances used
        "gtol": gtol_by_mpol.get(mpol),
        # Optimization results (partial)
        "optimization_success": None,
        "optimization_message": "partial snapshot from callback",
        "final_objective": float(J),
        "final_iota": float(iota.J()),
        "final_volume": float(boozer_surface.surface.volume()),
        # Diagnostic metrics
        "nonQS_ratio": float(JnonQSRatio.J()),
        "boozer_residual": float(JBoozerResidual.J()),
        "iota_penalty": float(Jiota.J()),
        "current_penalty": float(Jcurrent.J()),
        "max_current": float(max_current),
        "num_currents_over_threshold": int(num_over),
        # Coil information
        "# TF coils": len(tf_coils),
        "# dipole coils": len(dipole_coils),
        # Inherited from stage 2
        "eq_name": results["eq_name"],
        "eq_dir": results["eq_dir"],
        "surf_nfp": results["surf_nfp"],
        "surf_s": results["surf_s"],
        "VV_R0": results["VV_R0"],
        "VV_a": results["VV_a"],
        "VV_b": results["VV_b"],
        "ntf": results["ntf"],
    }
    save(partial_results, os.path.join(OUT_DIR_ITER, "results.json"))
    # Save the coils
    bs.save(OUT_DIR_ITER + "/bs_opt.json")

    # Advance iteration counter
    run_dict['it'] += 1

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================
# This is created by and contains the results from stage 2, which we use to initialize single stage
results = load(os.path.join(INIT_DIR, 'results.json'))

# Boozer surface resolution
mpol = 6
ntor = 6

# EMPIRICAL convergence tolerances for different mpol values
gtol_by_mpol = {6: 1e-8, 8: 1e-8, 10: 5e-9, 12: 1e-9}

# Output directory setup
eq_name = results["eq_name"]
OUT_ROOT = f"../single_stage_scans_no_sparsity_cp_fix/{eq_name}"
os.makedirs(OUT_ROOT, exist_ok=True)

# Determine root from unique time
current_time = time.strftime("%y%m%d_%H%M%S")
run_meta = (
    f"iota_tar{IOTA_TARGET:g}"
    f"_weight{IOTA_WEIGHT:g}"
    f"_qs_weight{QS_WEIGHT:g}"
    f"_cur_tar{CURRENT_THRESHOLD:g}"
    f"_weight{CURRENT_WEIGHT:g}"
)
OUT_DIR_RUN = os.path.join(OUT_ROOT, f"{current_time}_{run_meta}")
os.makedirs(OUT_DIR_RUN, exist_ok=True)

# Send this to terminal/slurm output
print(f"Output directory: {OUT_DIR_RUN}")

boozer_type = {'initial': 'least_squares', 'final': 'exact'}  # example
stage = 'initial'  # or 'final', depending on what you want

# ==============================================================================
# SURFACE GEOMETRY DEFINITIONS
# ==============================================================================
# Solely for visualization purposes
VV = SurfaceRZFourier(nfp=results["surf_nfp"])
VV.set_rc(0, 0, results["VV_R0"])
VV.set_rc(1, 0, results["VV_a"])
VV.set_zs(1, 0, results["VV_b"])

# ==============================================================================
# LOAD EQUILIBRIUM AND COILS
# ==============================================================================
# Load coil set from previous run
bs = load(os.path.join(INIT_DIR, 'bs_opt.json'))

# Initialize the boundary magnetic surface
surf_opt_path = os.path.join(INIT_DIR, 'surf_opt.json')
if os.path.exists(surf_opt_path): # load from single-stage
    surf = load(surf_opt_path)
    plas_nPhi = surf.quadpoints_phi.size
    plas_nTheta = surf.quadpoints_theta.size
else: # load and scale the surface from stage 2
    plas_nPhi = 128
    plas_nTheta = 64
    eq_name_full = os.path.join(results["eq_dir"], results["eq_name"] + ".nc")
    surf = SurfaceRZFourier.from_wout(
        eq_name_full, s=results["surf_s"], range="half period", nphi=plas_nPhi, ntheta=plas_nTheta
    )
    surf.set_dofs(results["surf_dof_scale"] * surf.get_dofs())
VOL_TARGET = 0.3 # surf.volume() # we'll assert that this is our target for now consistency

# Extract coil information
num_tf_coils = results["ntf"] * 2 * results["surf_nfp"]  # ntf is the number of TF coils per half-period, so this is the total number
coils = bs.coils
curves = [c.curve for c in coils]
tf_coils = coils[:num_tf_coils]
tf_curves = [c.curve for c in tf_coils]
dipole_coils = coils[num_tf_coils:]
dipole_curves = [c.curve for c in dipole_coils]

# Just triple make sure they're fixed
for c in dipole_curves:
    c.fix_all()

# ==============================================================================
# BEGIN SINGLE STAGE OPTIMIZATION
# ==============================================================================
# Within each run, create a subfolder for the specific (mpol, ntor) resolution
OUT_DIR_ITER = os.path.join(OUT_DIR_RUN, f"mpol{mpol}_ntor{ntor}")
os.makedirs(OUT_DIR_ITER, exist_ok=True)

# Redirect all standard output and errors to a log file in OUT_DIR_ITER
log_path = os.path.join(OUT_DIR_ITER, "log.txt")
log_file = open(log_path, "a", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

start_time = time.time()
print(f"Timer started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

print(f"\n===== Starting single stage optimization for mpol = {mpol} and ntor = {ntor} =====")
print(f"# of TF coils: {len(tf_coils)}")
print(f"# of Dipole coils: {len(dipole_coils)}")
print("Starting equilibrium = ", eq_name)
print(f"Target volume: {VOL_TARGET}")
print(f"Target iota: {IOTA_TARGET}")
print(f"Target current threshold: {CURRENT_THRESHOLD}")
print(f"Current weight: {CURRENT_WEIGHT}")
print(f"QS (nonQS ratio) weight: {QS_WEIGHT}")
print(f"Iota weight: {IOTA_WEIGHT}\n")

# Initialize Boozer surface using the loaded in coils + surface
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
boozer_surface = initialize_boozer_surface(surf, mpol, ntor, bs, VOL_TARGET, CONSTRAINT_WEIGHT, IOTA_TARGET, G0)
print(f"Initial boozer surface volume: {boozer_surface.surface.volume()}")

# ==============================================================================
# SAVE INITIAL STATE
# ==============================================================================
# Not saving anything for now - takes up a lot of space, and can be obtained from the stage 2 run
# # Save initial coil configurations
# coils_to_vtk(coils, filename=OUT_DIR_ITER + "/coils_init", close=True)
# bs.save(OUT_DIR_ITER + f"/bs_init.json")

# # Save initial surface with magnetic field normal component data
# pointData = {"B_N/B": np.sum(bs.B().reshape((plas_nPhi, plas_nTheta, 3)) *
#     boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((plas_nPhi, plas_nTheta, 3))**2, axis=2))[:, :, None]}
# boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_init", extra_data=pointData)
# boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_init.json")

# # Generate initial diagnostic plots
# plot_relBfinal_norm_modB(bs, boozer_surface.surface, OUT_DIR_ITER, "initial", plot_config)
# plot_cross_section(boozer_surface.surface, VV, OUT_DIR_ITER, "initial", plot_config)
# plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, OUT_DIR_ITER, "initial", plot_config)

# ==============================================================================
# DEFINE OBJECTIVE FUNCTION COMPONENTS
# ==============================================================================
# Biot-Savart field calculation
bs_obj = BiotSavart(coils)

# Quasi-symmetry and Boozer coordinate residuals
nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
if boozer_type[stage]=='exact':
    brs = [BoozerResidualExact(boozer_surface, bs_obj)]
else:
    brs = [BoozerResidual(boozer_surface, bs_obj)]

# Individual objective terms
iota = Iotas(boozer_surface)
Jiota = QuadraticPenalty(iota, IOTA_TARGET)
JnonQSRatio = sum(nonQSs)
JBoozerResidual = sum(brs)
Jcurrent = CurrentPenalty([c.current for c in dipole_coils], p=10.0)

# Combined objective function
JF = JBoozerResidual + QS_WEIGHT * JnonQSRatio + IOTA_WEIGHT * Jiota + CURRENT_WEIGHT * Jcurrent

# Extract degrees of freedom
dofs = JF.x
# ==============================================================================
# INITIALIZE OPTIMIZATION STATE
# ==============================================================================
# Initialize run_dict after JF and boozer_surface are ready
run_dict = {
    'sdofs': boozer_surface.surface.x.copy(),
    'iota': boozer_surface.res['iota'],
    'G': boozer_surface.res['G'],
    'J': JF.J(),
    'dJ': JF.dJ().copy(),
    'it': 1,
    'lscount': 0,
    'x_prev': dofs.copy(),
    'failed_boozer_solves': 0,
}

# ==============================================================================
# RUN OPTIMIZATION
# ==============================================================================
# Use Python BFGS instead of Fortran L-BFGS-B: the simsoptpp C++ extension
# dynamically loads Cray libsci which overwrites the ddot symbol used by
# L-BFGS-B's Fortran code, causing a broken initial step (~1e10 * ||g||).
# Python BFGS uses numpy's own BLAS (OpenBLAS) and is unaffected.
# With 84 DOFs the full Hessian approximation is negligible in memory.
# We only use gtol here because BFGS primarily uses gradient information.
res = minimize(fun, dofs, jac=True, method='BFGS',
            callback=callback,
            options={'maxiter': MAXITER, 'gtol': gtol_by_mpol.get(mpol)})
print(res.message)

# ==============================================================================
# SAVE OPTIMIZED STATE AND PERFORM POSTPROCESSING
# ==============================================================================
# Save optimized coil configurations
coils_to_vtk(coils, filename=OUT_DIR_ITER + "/coils_opt", close=True)
bs.save(OUT_DIR_ITER + "/bs_opt.json")

# Save vacuum vessel for visualization
VV.to_vtk(os.path.join(OUT_DIR_ITER, "vacuum_vessel"))

# Save optimized surface with magnetic field normal component data
pointData = {"B_N/B": np.sum(bs.B().reshape((plas_nPhi, plas_nTheta, 3)) *
    boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((plas_nPhi, plas_nTheta, 3))**2, axis=2))[:, :, None]}
boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_opt", extra_data=pointData)
boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_opt.json")

# Print final results
print("\n===== Final results =====\n")
print(f"Boozer surface volume: {boozer_surface.surface.volume()}")
print(f"Iota: {Iotas(boozer_surface).J()}")
print(f"Max current: {np.max([abs(c.current.get_value()) for c in dipole_coils])} A")
print(f"Number of currents over threshold: {np.sum([abs(c.current.get_value()) > CURRENT_THRESHOLD for c in dipole_coils])}")
print(f"Non-QS ratio: {JnonQSRatio.J()}")
print(f"Boozer residual: {JBoozerResidual.J()}")
print(f"Iota penalty: {Jiota.J()}")
print(f"Current penalty: {Jcurrent.J()}\n")

# Generate final diagnostic plots
plot_relBfinal_norm_modB(bs, boozer_surface.surface, OUT_DIR_ITER, "optimized", plot_config)
plot_cross_section(boozer_surface.surface, VV, OUT_DIR_ITER, "optimized", plot_config)
plot_coil_currents_on_theta_phi_grid(dipole_coils, VV, OUT_DIR_ITER, "optimized", plot_config)

# Save results dictionary for reproducibility and downstream use
results_output = {
    # Stage 2 directory (in a \"graph\" sub-dict so postprocessing can parse init_id)
    "graph": {
        "init_dir": INIT_DIR,
    },

    # Optimization configuration
    "mpol": mpol,
    "ntor": ntor,
    "maxiter": MAXITER,
    "constraint_weight": CONSTRAINT_WEIGHT,
    "iota_target": IOTA_TARGET,
    "qs_weight": QS_WEIGHT,
    "iota_weight": IOTA_WEIGHT,
    "current_threshold": CURRENT_THRESHOLD,
    "current_weight": CURRENT_WEIGHT,

    # Convergence tolerances used
    "gtol": gtol_by_mpol.get(mpol),

    # Optimization results
    "optimization_success": res.success,
    "optimization_message": res.message,
    "final_objective": JF.J(),
    "final_iota": float(Iotas(boozer_surface).J()),
    "final_volume": float(boozer_surface.surface.volume()),

    # Diagnostic metrics
    "nonQS_ratio": float(JnonQSRatio.J()),
    "boozer_residual": float(JBoozerResidual.J()),
    "iota_penalty": float(Jiota.J()),
    "current_penalty": float(Jcurrent.J()),
    "max_current": float(np.max([abs(c.current.get_value()) for c in dipole_coils])),
    "num_currents_over_threshold": int(np.sum([abs(c.current.get_value()) > CURRENT_THRESHOLD for c in dipole_coils])),

    # Coil information
    "# TF coils": len(tf_coils),
    "# dipole coils": len(dipole_coils),

    # Inherited from stage 2
    "eq_name": results["eq_name"],
    "eq_dir": results["eq_dir"],
    "surf_nfp": results["surf_nfp"],
    "surf_s": results["surf_s"],
    "VV_R0": results["VV_R0"],
    "VV_a": results["VV_a"],
    "VV_b": results["VV_b"],
    "ntf": results["ntf"],
}

# Save to file
save(results_output, os.path.join(OUT_DIR_ITER, "results.json"))
print(f"Results saved to {os.path.join(OUT_DIR_ITER, 'results.json')}")

# Print total wall time
end_time = time.time()
elapsed = end_time - start_time
print(f"Total wall time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")

# Close log file
log_file.close()