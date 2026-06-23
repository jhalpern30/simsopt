#!/usr/bin/env python3
"""
Batch-compute the Boozer surface symmetry-breaking fraction and store it in results.json.

For each run directory under a scan root that contains ``surf_opt.json``,
``bs_opt.json``, and ``results.json``, this script:

  1) runs VMEC from a template input (default: ``input.fixed``),
  2) runs BOOZXFORM on the resulting ``wout`` file,
  3) evaluates the symmetry-breaking fraction

         f(s) = sqrt( sum_{n != 0} |B_{m,n}(s)|^2 / sum_{m,n} |B_{m,n}(s)|^2 )

     using ``booz_xform`` (same formula as :func:`plot_boozxform` in
     :file:`postprocess_single_stage_runs.py`), and
  4) writes ``mean_surface_symmetry_breaking_fraction`` (mean of ``f(s)`` over
     normalized toroidal flux) into ``results.json``.

VMEC and Boozer outputs are written under each run directory. Existing
``wout`` / ``boozmn`` files are reused unless ``--force-rerun`` is set.

Typical usage (from ``examples/dipoles``)::

    python add_surface_symmetry_breaking_to_results.py --dry-run
    python add_surface_symmetry_breaking_to_results.py --continue-on-error
    python add_surface_symmetry_breaking_to_results.py --mpol-ntor 12

The default ``--scan-root`` is ``../single_stage_true_epsilon_adaptive_res``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import booz_xform as bx
import numpy as np

from simsopt._core.optimizable import load
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.mhd.vmec import Vmec


_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FIXED = _SCRIPT_DIR / "input.fixed"
DEFAULT_SCAN_ROOT = _SCRIPT_DIR.parent / "single_stage_true_epsilon_adaptive_res"

METRIC_KEY = "mean_surface_symmetry_breaking_fraction"
MBOZ_KEY = "surface_symmetry_breaking_booz_xform_mboz"
NBOZ_KEY = "surface_symmetry_breaking_booz_xform_nboz"
NOTE_KEY = "surface_symmetry_breaking_fraction_note"


def vmec_output_basenames(
    input_fixed_path: os.PathLike,
    mpi_group: int = 0,
    iteration: int = 0,
) -> tuple[str, str]:
    """Basenames of ``wout`` and ``boozmn`` after a VMEC run (iteration 0)."""
    stem = Path(input_fixed_path).name
    tagged = f"{stem}_{mpi_group:03d}_{iteration:06d}"
    wout = tagged.replace("input.", "wout_") + ".nc"
    boozmn = tagged.replace("input.", "boozmn_") + ".nc"
    return wout, boozmn


@contextmanager
def working_directory(path: os.PathLike):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _payload(results: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(results.get("graph"), dict):
        return results["graph"]
    return results


def has_cached_metric(results: Dict[str, Any]) -> bool:
    payload = _payload(results)
    val = payload.get(METRIC_KEY)
    return isinstance(val, (int, float))


def iter_run_directories(scan_root: Path) -> Iterable[Path]:
    """
    Yield run directories under ``scan_root`` with surf_opt.json, bs_opt.json,
    and results.json.
    """
    scan_root = Path(scan_root).resolve()
    if not scan_root.is_dir():
        raise FileNotFoundError(f"scan root is not a directory: {scan_root}")

    seen: set[Path] = set()
    for results_path in sorted(scan_root.rglob("results.json")):
        run_dir = results_path.parent.resolve()
        if run_dir in seen:
            continue
        if not (run_dir / "surf_opt.json").is_file():
            continue
        if not (run_dir / "bs_opt.json").is_file():
            continue
        seen.add(run_dir)
        yield run_dir


def _extract_mpol_ntor_from_run_dir(run_dir: Path) -> Optional[Tuple[int, int]]:
    """Parse (mpol, ntor) from a leaf directory name like ``mpol6_ntor6``."""
    match = re.fullmatch(r"mpol(\d+)_ntor(\d+)", run_dir.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def filter_run_directories_by_resolution(
    runs: Iterable[Path],
    mpol_ntor: Optional[int],
) -> list[Path]:
    """Keep only runs in ``mpol<N>_ntor<N>`` directories when ``mpol_ntor`` is set."""
    if mpol_ntor is None:
        return list(runs)

    selected: list[Path] = []
    for run_dir in runs:
        parsed = _extract_mpol_ntor_from_run_dir(run_dir)
        if parsed is None:
            continue
        run_mpol, run_ntor = parsed
        if run_mpol == mpol_ntor and run_ntor == mpol_ntor:
            selected.append(run_dir)
    return selected


def run_vmec(input_fixed_path: Path) -> None:
    """Run VMEC in the current directory using surf_opt.json and bs_opt.json."""
    input_fixed_path = Path(input_fixed_path).resolve()
    if not input_fixed_path.is_file():
        raise FileNotFoundError(f"VMEC input template not found: {input_fixed_path}")

    surf = load("surf_opt.json")
    bs = load("bs_opt.json")
    tf = ToroidalFlux(surf, bs)

    surf_rz = surf.to_RZFourier()

    vmec = Vmec(str(input_fixed_path))
    vmec.boundary = surf_rz
    vmec.indata.phiedge = tf.J()
    vmec.indata.mpol = surf_rz.mpol
    vmec.indata.ntor = surf_rz.ntor
    vmec.indata.nfp = surf_rz.nfp
    vmec.indata.nzeta = 3 * surf_rz.ntor
    vmec.indata.ntheta = 3 * surf_rz.mpol
    vmec.indata.raxis_cc[0] = surf_rz.get_rc(0, 0)   # for SurfaceRZFourier

    vmec.run()


def run_boozxform(wout_name: str, boozmn_name: str, mboz: int, nboz: int) -> None:
    """Run BOOZXFORM on ``wout_name`` in the current directory."""
    if not os.path.isfile(wout_name):
        raise FileNotFoundError(
            f"Missing {wout_name}; run the VMEC step first (cwd={os.getcwd()})."
        )

    b = bx.Booz_xform()
    b.verbose = False
    b.read_wout(wout_name)
    b.mboz = mboz
    b.nboz = nboz
    b.run()
    b.write_boozmn(boozmn_name)


def mean_symmetry_breaking_fraction_from_boozmn(boozmn_name: str) -> float:
    """
    Return mean over s of the non-symmetric Boozer-mode fraction of |B|.

    Uses bmnc_b and xn_b from booz_xform, excluding modes with xn == 0.
    """
    b = bx.Booz_xform()
    b.read_boozmn(boozmn_name)

    bmnc = b.bmnc_b
    xn = b.xn_b
    denom = np.sum(bmnc**2, axis=0)
    if np.any(denom <= 0.0):
        raise ValueError(f"Non-positive Boozer |B|^2 denominator in {boozmn_name}")

    f = np.sqrt(np.sum(bmnc[xn != 0, :] ** 2, axis=0) / denom)
    return float(np.mean(f))


def compute_metric_for_run(
    run_dir: Path,
    input_fixed_path: Path,
    mboz: int,
    nboz: int,
    force_rerun: bool,
) -> float:
    """Run VMEC/BOOZXFORM as needed and return the mean symmetry-breaking fraction."""
    input_fixed_path = Path(input_fixed_path).resolve()
    wout_name, boozmn_name = vmec_output_basenames(input_fixed_path)

    with working_directory(run_dir):
        if force_rerun or not os.path.isfile(wout_name):
            run_vmec(input_fixed_path)
        if force_rerun or not os.path.isfile(boozmn_name):
            run_boozxform(wout_name, boozmn_name, mboz, nboz)
        return mean_symmetry_breaking_fraction_from_boozmn(boozmn_name)


def update_one_run(
    run_dir: Path,
    input_fixed_path: Path,
    mboz: int,
    nboz: int,
    force_rerun: bool,
    dry_run: bool,
    recompute_cached: bool,
) -> Tuple[bool, str]:
    results_path = run_dir / "results.json"

    try:
        with results_path.open("r") as f:
            results = json.load(f)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"Failed reading results.json: {exc}"

    if has_cached_metric(results) and not force_rerun and not recompute_cached:
        payload = _payload(results)
        cached = float(payload[METRIC_KEY])
        return True, f"skipped (cached {METRIC_KEY}={cached:.6e})"

    try:
        metric = compute_metric_for_run(
            run_dir,
            input_fixed_path,
            mboz,
            nboz,
            force_rerun=force_rerun,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f"Failed computing metric: {exc}"

    payload = _payload(results)
    payload[METRIC_KEY] = metric
    payload[MBOZ_KEY] = mboz
    payload[NBOZ_KEY] = nboz
    payload[NOTE_KEY] = (
        "mean over s of sqrt(sum_{n!=0} bmnc^2 / sum bmnc^2) from booz_xform on VMEC wout"
    )

    if not dry_run:
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")

    action = "would update" if dry_run else "updated"
    return True, f"{action} {METRIC_KEY}={metric:.6e}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-root",
        type=Path,
        default=DEFAULT_SCAN_ROOT,
        help=f"Root directory to scan (default: {DEFAULT_SCAN_ROOT})",
    )
    p.add_argument(
        "--input-fixed",
        type=Path,
        default=DEFAULT_INPUT_FIXED,
        help=f"VMEC input template (default: {DEFAULT_INPUT_FIXED})",
    )
    p.add_argument("--mboz", type=int, default=48, help="BOOZXFORM mboz (default: 48)")
    p.add_argument("--nboz", type=int, default=48, help="BOOZXFORM nboz (default: 48)")
    p.add_argument(
        "--mpol-ntor",
        type=int,
        default=None,
        help="Only process runs in mpolN_ntorN directories for this N",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List run directories and skip VMEC/Boozer/results.json writes",
    )
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run VMEC and BOOZXFORM even if wout/boozmn already exist",
    )
    p.add_argument(
        "--recompute-cached",
        action="store_true",
        help=f"Recompute even when {METRIC_KEY} is already in results.json",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep processing after a run fails (prints traceback)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=5,
        metavar="N",
        help="Print a progress line every N computed runs (default: 5)",
    )
    args = p.parse_args(argv)

    runs = filter_run_directories_by_resolution(
        iter_run_directories(args.scan_root),
        mpol_ntor=args.mpol_ntor,
    )
    if not runs:
        if args.mpol_ntor is not None:
            print(
                f"No run directories matching mpol{args.mpol_ntor}_ntor{args.mpol_ntor} "
                f"under {args.scan_root.resolve()!s}",
                file=sys.stderr,
            )
        else:
            print(f"No run directories found under {args.scan_root.resolve()!s}", file=sys.stderr)
        return 1

    resolution_note = ""
    if args.mpol_ntor is not None:
        resolution_note = f" (mpol{args.mpol_ntor}_ntor{args.mpol_ntor})"
    print(
        f"Found {len(runs)} run director(y|ies) under "
        f"{args.scan_root.resolve()!s}{resolution_note}."
    )
    for run_dir in runs:
        print(f"  {run_dir}")

    if args.dry_run:
        print("Dry run: no VMEC/Boozer execution or results.json updates.")
        return 0

    input_fixed = Path(args.input_fixed).resolve()
    updated = 0
    skipped = 0
    computed = 0
    failed = 0
    exit_code = 0

    for idx, run_dir in enumerate(runs, start=1):
        print(f"\n=== [{idx}/{len(runs)}] {run_dir} ===", flush=True)
        try:
            ok, msg = update_one_run(
                run_dir,
                input_fixed,
                args.mboz,
                args.nboz,
                force_rerun=args.force_rerun,
                dry_run=False,
                recompute_cached=args.recompute_cached,
            )
        except Exception:
            exit_code = 1
            failed += 1
            traceback.print_exc()
            if not args.continue_on_error:
                break
            continue

        if ok:
            updated += 1
            if msg.startswith("skipped"):
                skipped += 1
            else:
                computed += 1
                if computed % args.progress_every == 0:
                    print(msg, flush=True)
            print(msg, flush=True)
        else:
            failed += 1
            exit_code = 1
            print(f"[FAIL] {msg}", file=sys.stderr)
            if not args.continue_on_error:
                break

    print(
        f"\nDone: {updated} runs handled ({skipped} cached, {computed} computed); "
        f"{failed} failed."
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
