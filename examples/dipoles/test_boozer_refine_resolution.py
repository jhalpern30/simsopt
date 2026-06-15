"""
Quick test: load a random mpol6/ntor6 single_stage_true_epsilon run, measure the
Boozer residual there, then re-initialize the Boozer surface at higher resolution
(default mpol=ntor=9) via `initialize_boozer_surface` and measure the residual
again. Prints a before/after comparison.

Usage:
    python test_boozer_refine_resolution.py
    python test_boozer_refine_resolution.py --new-mpol 9 --new-ntor 9
    python test_boozer_refine_resolution.py --run-dir ../single_stage_true_epsilon/<eq>/iota.../mpol6_ntor6
"""

import os
import re
import glob
import random
import argparse

import numpy as np

from simsopt._core.optimizable import load
from simsopt.field import BiotSavart
from simsopt.geo.surfaceobjectives import BoozerResidual
from boozer_functions import initialize_boozer_surface

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-root",
    type=str,
    default="../single_stage_true_epsilon/wout_nfp22ginsburg_000_000281",
    help="Parent containing iota*_fcp*_vt*/mpol6_ntor6 dirs.",
)
parser.add_argument(
    "--run-dir",
    type=str,
    default=None,
    help="Explicit mpol6_ntor6 run dir. If unset, a random one is picked.",
)
parser.add_argument("--base-mpol", type=int, default=6)
parser.add_argument("--base-ntor", type=int, default=6)
parser.add_argument("--new-mpol", type=int, default=9)
parser.add_argument("--new-ntor", type=int, default=9)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

# ---- Pick a run dir ----
if args.run_dir:
    run_dir = args.run_dir
else:
    pattern = os.path.join(
        args.results_root,
        "iota*_fcp*_vt*",
        f"mpol{args.base_mpol}_ntor{args.base_ntor}",
    )
    candidates = [
        d for d in glob.glob(pattern)
        if os.path.isfile(os.path.join(d, "bs_opt.json"))
        and os.path.isfile(os.path.join(d, "surf_opt.json"))
        and os.path.isfile(os.path.join(d, "results.json"))
    ]
    if not candidates:
        raise SystemExit(f"No complete mpol{args.base_mpol}_ntor{args.base_ntor} runs under {args.results_root}")
    if args.seed is not None:
        random.seed(args.seed)
    run_dir = random.choice(candidates)

print(f"Using run dir: {run_dir}")

# ---- Extract target metadata from directory name ----
m = re.search(r"iota([0-9.]+)_fcp([0-9.]+)kA_vt([0-9.]+)", run_dir)
if not m:
    raise SystemExit(f"Could not parse target params from {run_dir}")
iota_target = float(m.group(1))
fcp_kA = float(m.group(2))
vol_target = float(m.group(3))
print(f"  iota_target={iota_target}, f_cp={fcp_kA}kA, vol_target={vol_target}")

# ---- Load stored objects ----
results = load(os.path.join(run_dir, "results.json"))
bs = load(os.path.join(run_dir, "bs_opt.json"))
surf = load(os.path.join(run_dir, "surf_opt.json"))

# Split coils into TF / dipole using ntf from results
num_tf = results["ntf"] * 2 * results["surf_nfp"]
coils = bs.coils
tf_coils = coils[:num_tf]

# G0 from the TF currents (same formula as in single_stage_true_epsilon.py)
current_sum = sum(abs(c.current.get_value()) for c in tf_coils)
G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 1e-7 / (2 * np.pi))

bs_obj = BiotSavart(coils)

BOOZER_CW = 1.0  # least-squares formulation, same as production run

# ======================================================================
# BEFORE: solve at base resolution (mpol6/ntor6)
# ======================================================================
print(f"\n--- Solving Boozer at mpol={args.base_mpol}, ntor={args.base_ntor} ---")
bs_surf_base = initialize_boozer_surface(
    surf, args.base_mpol, args.base_ntor, bs_obj,
    vol_target, BOOZER_CW, iota_target, G0,
)
J_before = float(BoozerResidual(bs_surf_base, bs_obj).J())
iota_before = float(bs_surf_base.res["iota"])
G_before = float(bs_surf_base.res["G"])
vol_before = float(bs_surf_base.surface.volume())
print(f"  iota = {iota_before:.6f}")
print(f"  G    = {G_before:.6e}")
print(f"  vol  = {vol_before:.6f}")
print(f"  BoozerResidual = {J_before:.6e}")

# ======================================================================
# AFTER: re-initialize at higher resolution from the solved surface
# ======================================================================
print(f"\n--- Re-initializing at mpol={args.new_mpol}, ntor={args.new_ntor} ---")
bs_surf_hi = initialize_boozer_surface(
    bs_surf_base.surface, args.new_mpol, args.new_ntor, bs_obj,
    vol_target, BOOZER_CW, iota_before, G_before,
)
J_after = float(BoozerResidual(bs_surf_hi, bs_obj).J())
iota_after = float(bs_surf_hi.res["iota"])
G_after = float(bs_surf_hi.res["G"])
vol_after = float(bs_surf_hi.surface.volume())
print(f"  iota = {iota_after:.6f}")
print(f"  G    = {G_after:.6e}")
print(f"  vol  = {vol_after:.6f}")
print(f"  BoozerResidual = {J_after:.6e}")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  run_dir:                 {run_dir}")
print(f"  mpol/ntor (before):      {args.base_mpol}/{args.base_ntor}")
print(f"  mpol/ntor (after):       {args.new_mpol}/{args.new_ntor}")
print(f"  BoozerResidual (before): {J_before:.6e}")
print(f"  BoozerResidual (after):  {J_after:.6e}")
if J_before > 0:
    ratio = J_after / J_before
    print(f"  ratio after/before:      {ratio:.3f}x")
    if ratio < 1:
        print(f"  -> residual DECREASED by {(1 - ratio) * 100:.1f}%")
    else:
        print(f"  -> residual INCREASED by {(ratio - 1) * 100:.1f}%")
print(f"  iota:   {iota_before:.6f}  ->  {iota_after:.6f}  (target {iota_target})")
print(f"  volume: {vol_before:.6f}  ->  {vol_after:.6f}  (target {vol_target})")
