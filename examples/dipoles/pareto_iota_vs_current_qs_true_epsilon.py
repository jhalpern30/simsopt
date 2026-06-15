#!/usr/bin/env python3
"""
Scatter plot of max dipole current vs rotational transform for true-epsilon scans,
colored by QS error.

Designed for runs under:
  examples/single_stage_true_epsilon/<eq_dir>/iota*_fcp*kA*/mpol*_ntor*/

This script:
  - loads final records from each run (preferring iterations.json final iteration),
  - by default, uses only runs that satisfy all true-epsilon constraints within a
    fractional tolerance (default 2%):
      * Boozer residual threshold (f_b_threshold),
      * current threshold (f_cp_threshold),
      * iota tolerance (|iota - iota_target| <= iota_threshold),
  - writes a publication-style scatter plot with:
      x-axis = achieved iota, y-axis = max current (kA), color = QS error.
  - writes a line plot: x = iota, y = QS error, one line per distinct current bound
    (f_cp_threshold / kA), using the same default constraint filter.
  - optional --fcps: comma-separated f_CP bounds in kA to include (default: all).
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults (kept consistent with existing script)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "mathtext.fontset": "cm",
    }
)


def find_run_dirs(scan_root: Path):
    """Yield run directories matching mpol*_ntor* with iterations.json present."""
    scan_root = Path(scan_root).resolve()
    if not scan_root.is_dir():
        return
    for run_dir in scan_root.rglob("mpol*_ntor*"):
        if not run_dir.is_dir():
            continue
        if (run_dir / "iterations.json").is_file():
            yield run_dir


def _select_highest_resolution_run_dirs(
    run_dirs: list[Path],
    mpol_ntor: int | None,
) -> list[Path]:
    """
    Select which run directories to load.

    If mpol_ntor is provided, keep only mpol==ntor==mpol_ntor.
    If mpol_ntor is None, keep only the highest-resolution (mpol, ntor) run
    per parent iota/fcp/vt directory.
    """
    if mpol_ntor is not None:
        selected = []
        for d in run_dirs:
            mn = _extract_mpol_ntor_from_run_dir(d)
            if mn is None:
                continue
            if mn[0] == mpol_ntor and mn[1] == mpol_ntor:
                selected.append(d)
        return selected

    # Default: one run per parent directory, choosing highest (mpol, ntor).
    best_by_parent: dict[Path, tuple[tuple[int, int], Path]] = {}
    for d in run_dirs:
        mn = _extract_mpol_ntor_from_run_dir(d)
        if mn is None:
            continue
        parent = d.parent
        prev = best_by_parent.get(parent)
        if prev is None or mn > prev[0]:
            best_by_parent[parent] = (mn, d)
    return [v[1] for v in best_by_parent.values()]


def _extract_vol_target_from_run_dir(run_dir: Path) -> float | None:
    """
    Parse volume target from parent directory name like:
      iota0.1_fcp100kA_vt0.3
    """
    parent_name = run_dir.parent.name
    match = re.search(r"_vt([0-9eE+\-.]+)$", parent_name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_mpol_ntor_from_run_dir(run_dir: Path) -> tuple[int, int] | None:
    """
    Parse (mpol, ntor) from leaf directory name like:
      mpol6_ntor6
    """
    name = run_dir.name
    match = re.fullmatch(r"mpol(\d+)_ntor(\d+)", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _read_json(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _results_payload(results_data: dict) -> dict:
    """
    Return the dict that contains scalar results fields.

    Handles both:
      1) plain results dict with fields at top level, and
      2) SIMSON-wrapped dict where fields are nested under "graph".
    """
    graph = results_data.get("graph")
    if isinstance(graph, dict):
        return graph
    return results_data


def _frac_violation(actual: float, limit: float) -> float:
    """
    Relative violation amount for constraint actual <= limit.
    Returns 0 for satisfied constraints, otherwise (actual/limit - 1).
    """
    if limit <= 0:
        return np.inf
    if actual <= limit:
        return 0.0
    return float(actual / limit - 1.0)


def _iota_frac_violation(iota: float, iota_target: float, iota_threshold: float) -> float:
    """
    Relative violation amount for |iota - iota_target| <= iota_threshold.
    Returns 0 for satisfied constraints, otherwise (abs_delta/threshold - 1).
    """
    if iota_threshold <= 0:
        return np.inf
    abs_delta = abs(iota - iota_target)
    if abs_delta <= iota_threshold:
        return 0.0
    return float(abs_delta / iota_threshold - 1.0)


def _parse_fcps_kA(s: str | None) -> list[float] | None:
    """
    Parse comma-separated f_CP bounds in kA from CLI.
    Returns None if unset or empty (meaning: include all).
    """
    if s is None:
        return None
    parts = [p.strip() for p in str(s).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return [float(p) for p in parts]


def _filter_records_by_fcp_kA(
    records: list[dict],
    fcps_kA: list[float] | None,
    atol_a: float = 1.0,
) -> list[dict]:
    """Keep records whose f_cp_threshold (A) matches one of the listed bounds (kA)."""
    if fcps_kA is None:
        return records
    thresholds_a = [fka * 1e3 for fka in fcps_kA]
    out = []
    for r in records:
        fcp = float(r["f_cp_threshold"])
        if any(np.isclose(fcp, t, rtol=0.0, atol=atol_a) for t in thresholds_a):
            out.append(r)
    return out


def load_scan(
    scan_root: Path,
    allowed_violation_frac: float,
    vol_target: float | None,
    mpol_ntor: int | None,
):
    """
    Load one record per run with valid iterations + results.

    Each record includes ``feasible`` (True iff all three constraints are within
    ``allowed_violation_frac``). Per-constraint violation counts are incremented
    only for runs with complete data that fail at least one constraint.
    """
    records = []
    stats = {
        "total_dirs": 0,
        "rejected_bad_iterations_json": 0,
        "rejected_bad_results_json": 0,
        "rejected_missing_iterations_or_config_fields": 0,
        "rejected_missing_results_fields": 0,
        "rejected_constraint_violation": 0,
        "skipped_nonmatching_vol_target": 0,
        "skipped_nonmatching_mpol_ntor": 0,
        "violations_boozer": 0,
        "violations_current": 0,
        "violations_iota": 0,
    }

    run_dirs_all = list(find_run_dirs(scan_root))
    stats["total_dirs"] = len(run_dirs_all)
    run_dirs = _select_highest_resolution_run_dirs(run_dirs_all, mpol_ntor)
    stats["skipped_nonmatching_mpol_ntor"] = len(run_dirs_all) - len(run_dirs)

    for run_dir in run_dirs:
        run_vol_target = _extract_vol_target_from_run_dir(run_dir)
        if vol_target is not None:
            if run_vol_target is None or not np.isclose(
                run_vol_target,
                vol_target,
                rtol=0.0,
                atol=1e-12,
            ):
                stats["skipped_nonmatching_vol_target"] += 1
                continue
        run_mpol_ntor = _extract_mpol_ntor_from_run_dir(run_dir)

        iter_data = _read_json(run_dir / "iterations.json")
        if not isinstance(iter_data, dict):
            stats["rejected_bad_iterations_json"] += 1
            continue
        iterations = iter_data.get("iterations")
        config = iter_data.get("config")
        if not isinstance(iterations, list) or not iterations or not isinstance(config, dict):
            stats["rejected_missing_iterations_or_config_fields"] += 1
            continue
        last = iterations[-1]
        if not isinstance(last, dict):
            stats["rejected_missing_iterations_or_config_fields"] += 1
            continue

        # Last-iteration fields from single_stage_true_epsilon.py history.
        iota = last.get("iota")
        max_current = last.get("max_current")
        boozer = last.get("boozer_residual")
        qs_error = last.get("nonQS_ratio")
        iota_target = config.get("iota_target")
        iota_threshold = config.get("iota_threshold")
        f_b_threshold = config.get("f_b_threshold")
        f_cp_threshold = config.get("f_cp_threshold")
        required_iter_vals = (
            iota,
            max_current,
            boozer,
            qs_error,
            iota_target,
            iota_threshold,
            f_b_threshold,
            f_cp_threshold,
        )
        if any(v is None for v in required_iter_vals):
            stats["rejected_missing_iterations_or_config_fields"] += 1
            continue

        # Keep results.json read to ensure run is complete and capture metadata when present.
        results_data = _read_json(run_dir / "results.json")
        if not isinstance(results_data, dict):
            stats["rejected_bad_results_json"] += 1
            continue
        results_payload = _results_payload(results_data)
        if results_payload.get("optimization_message") is None:
            stats["rejected_missing_results_fields"] += 1
            continue

        iota = float(iota)
        max_current = float(max_current)
        boozer = float(boozer)
        qs_error = float(qs_error)
        iota_target = float(iota_target)
        iota_threshold = float(iota_threshold)
        f_b_threshold = float(f_b_threshold)
        f_cp_threshold = float(f_cp_threshold)

        boozer_violation = _frac_violation(boozer, f_b_threshold)
        current_violation = _frac_violation(max_current, f_cp_threshold)
        iota_violation = _iota_frac_violation(iota, iota_target, iota_threshold)
        feasible = (
            boozer_violation <= allowed_violation_frac
            and current_violation <= allowed_violation_frac
            and iota_violation <= allowed_violation_frac
        )
        if not feasible:
            stats["rejected_constraint_violation"] += 1
            if boozer_violation > allowed_violation_frac:
                stats["violations_boozer"] += 1
            if current_violation > allowed_violation_frac:
                stats["violations_current"] += 1
            if iota_violation > allowed_violation_frac:
                stats["violations_iota"] += 1

        records.append(
            {
                "run_dir": str(run_dir),
                "feasible": feasible,
                "iota": iota,
                "vol_target": run_vol_target,
                "mpol_ntor": run_mpol_ntor[0] if run_mpol_ntor is not None else None,
                "max_current_kA": max_current / 1e3,
                "qs_error": qs_error,
                "boozer_residual": boozer,
                "iota_target": iota_target,
                "iota_threshold": iota_threshold,
                "f_b_threshold": f_b_threshold,
                "f_cp_threshold": f_cp_threshold,
            }
        )

    return records, stats


def make_plot(
    iota: np.ndarray,
    current_kA: np.ndarray,
    qs_error: np.ndarray,
    y_max: float | None,
    out_path: Path,
):
    """Create max-current vs iota scatter, colored by QS error."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    sc = ax.scatter(
        iota,
        current_kA,
        c=qs_error,
        cmap="RdYlGn_r",
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.6,
        zorder=3,
    )

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$f_{\mathrm{QS}}$")
    cbar_fmt = ScalarFormatter(useMathText=True)
    cbar_fmt.set_scientific(True)
    cbar_fmt.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(cbar_fmt)

    ax.set_xlabel(r"Rotational transform $\iota$")
    ax.set_ylabel("Max current (kA)")
    if y_max is not None:
        ax.set_ylim(top=y_max)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_iota_qs_lines_by_fcp(
    records: list[dict],
    out_path: Path,
):
    """
    Line plot: x = iota, y = QS error, one polyline per distinct f_cp_threshold (kA in legend).
    """
    if not records:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Group by current bound (A); legend uses kA for readability.
    by_fcp: dict[float, list[tuple[float, float]]] = {}
    for r in records:
        fcp = float(r["f_cp_threshold"])
        by_fcp.setdefault(fcp, []).append((float(r["iota"]), float(r["qs_error"])))

    fcp_sorted = sorted(by_fcp.keys())
    cmap = plt.get_cmap("tab10")
    for i, fcp_a in enumerate(fcp_sorted):
        pts = sorted(by_fcp[fcp_a], key=lambda t: t[0])
        iota_line = np.array([p[0] for p in pts])
        qs_line = np.array([p[1] for p in pts])
        label = rf"$f_{{\mathrm{{CP}}}}$ bound = {fcp_a / 1e3:g} kA"
        color = cmap(i % cmap.N)
        ax.plot(
            iota_line,
            qs_line,
            color=color,
            marker="o",
            markersize=5,
            linewidth=1.4,
            label=label,
            zorder=2,
        )

    ax.set_xlabel(r"Rotational transform $\iota$")
    ax.set_ylabel(r"$f_{\mathrm{QS}}$")
    ax.legend(loc="best", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot max current vs achieved iota for true-epsilon single-stage scans, "
            "colored by QS error."
        )
    )
    parser.add_argument(
        "--scan-dir",
        type=str,
        default="../single_stage_true_epsilon",
        help="Scan root directory to search (default: ../single_stage_true_epsilon).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: <scan-dir>/postprocess_plots).",
    )
    parser.add_argument(
        "--constraint-violation-frac",
        type=float,
        default=0.02,
        help=(
            "Reject runs where any constraint exceeds this fractional violation "
            "(default: 0.02 means >2%% over threshold is rejected)."
        ),
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional y-axis upper limit in kA (scatter plot only).",
    )
    parser.add_argument(
        "--include-constraint-violations",
        action="store_true",
        help=(
            "Include runs that violate constraints in the plots (default: only feasible runs)."
        ),
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=None,
        help=(
            "Only include runs from folders with matching volume target in "
            "'..._vt<value>' (e.g. --vol-target 0.3). Default: include all."
        ),
    )
    parser.add_argument(
        "--mpol-ntor",
        type=int,
        default=None,
        help=(
            "Only include runs in folders named mpolN_ntorN for this N "
            "(assumes mpol and ntor are the same value)."
        ),
    )
    parser.add_argument(
        "--fcps",
        type=str,
        default=None,
        metavar="KALIST",
        help=(
            "Comma-separated f_CP current bounds in kA to plot (e.g. '100,150' for "
            "runs with f_cp_threshold matching 100 kA and 150 kA). "
            "Default: include all bounds present in the scan."
        ),
    )
    args = parser.parse_args()

    scan_root = Path(args.scan_dir).resolve()
    if not scan_root.is_dir():
        raise SystemExit(f"Scan directory not found: {scan_root}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else scan_root / "postprocess_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    records_all, stats = load_scan(
        scan_root=scan_root,
        allowed_violation_frac=float(args.constraint_violation_frac),
        vol_target=args.vol_target,
        mpol_ntor=args.mpol_ntor,
    )

    use_all = bool(args.include_constraint_violations)
    records = records_all if use_all else [r for r in records_all if r["feasible"]]

    fcps_kA = _parse_fcps_kA(args.fcps)
    records_before_fcp = records
    records = _filter_records_by_fcp_kA(records, fcps_kA)

    print(f"Scan root: {scan_root}")
    print(
        "Constraint filter: reject if any of [Boozer, current, iota] "
        f"violation fraction > {args.constraint_violation_frac:g}"
    )
    if args.vol_target is not None:
        print(f"Volume target filter: vt == {args.vol_target:g}")
    else:
        print("Volume target filter: none (all vt values included)")
    if args.mpol_ntor is not None:
        print(f"Resolution filter: mpol == ntor == {args.mpol_ntor}")
    else:
        print(
            "Resolution filter: default highest available (mpol,ntor) per iota/fcp/vt directory"
        )
    if fcps_kA is None:
        print("f_CP bound filter: none (all f_CP values included)")
    else:
        print(
            "f_CP bound filter: kA = "
            + ", ".join(f"{x:g}" for x in fcps_kA)
            + f"  (kept {len(records)} of {len(records_before_fcp)} runs)"
        )
    if use_all:
        print("Plotting: all runs with complete data (including infeasible).")
    else:
        print("Plotting: feasible runs only (default).")
    print(f"Total run directories discovered: {stats['total_dirs']}")
    print(f"Skipped (non-matching vol-target): {stats['skipped_nonmatching_vol_target']}")
    print(f"Skipped (non-matching mpol/ntor): {stats['skipped_nonmatching_mpol_ntor']}")
    print(f"Rejected (bad iterations.json): {stats['rejected_bad_iterations_json']}")
    print(f"Rejected (bad results.json): {stats['rejected_bad_results_json']}")
    print(
        "Rejected (missing iteration/config fields): "
        f"{stats['rejected_missing_iterations_or_config_fields']}"
    )
    print(f"Rejected (missing results fields): {stats['rejected_missing_results_fields']}")
    print(f"Rejected (any constraint violation): {stats['rejected_constraint_violation']}")
    print(
        "Among complete runs, count exceeding Boozer threshold: "
        f"{stats['violations_boozer']}"
    )
    print(
        "Among complete runs, count exceeding current threshold: "
        f"{stats['violations_current']}"
    )
    print(
        "Among complete runs, count exceeding iota tolerance: "
        f"{stats['violations_iota']}"
    )
    print(f"Loaded complete runs: {len(records_all)}")
    print(f"Used for plotting: {len(records)}")

    if not records:
        print("No runs selected for plotting; no plots written.")
        return

    iota = np.array([r["iota"] for r in records], dtype=float)
    current_kA = np.array([r["max_current_kA"] for r in records], dtype=float)
    qs_error = np.array([r["qs_error"] for r in records], dtype=float)

    out_path = out_dir / "pareto_iota_vs_max_current_color_qs_error_true_epsilon.png"
    make_plot(
        iota=iota,
        current_kA=current_kA,
        qs_error=qs_error,
        y_max=args.y_max,
        out_path=out_path,
    )
    print(f"Wrote: {out_path}")

    lines_path = out_dir / "iota_vs_qs_error_lines_by_fcp_true_epsilon.png"
    make_iota_qs_lines_by_fcp(records=records, out_path=lines_path)
    print(f"Wrote: {lines_path}")


if __name__ == "__main__":
    main()

