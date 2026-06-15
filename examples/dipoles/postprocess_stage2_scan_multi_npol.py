#!/usr/bin/env python3
"""
Publication-quality multi-panel Pareto plot for stage-2 dipole scans.

For a hardcoded equilibrium name and hardcoded npol values, this script scans
directories under ../outputs with naming pattern:
  stage_2_npol{npol}_ntor8_{eq_name}*

It ensures each run's results.json contains max dipole center-field metrics
(computed from bs_opt.json when missing, then cached for later runs), then
plots avg_Bnormal vs max_dipole_center_field_over_on_axis in one subplot per
npol, including the Pareto front in each panel.
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


# ---------------------------------------------------------------------------
# Hardcoded campaign settings (edit these values only)
# ---------------------------------------------------------------------------
EQ_NAME = "wout_nfp22ginsburg_000_000281"
NPOL_VALUES = [9, 10, 11]
NTOR_VALUE = 8
FIELD_ERROR_METRIC = "avg_Bnormal"
FIELD_ERROR_THRESHOLD = 5e-3
Y_METRIC_KEY = "max_dipole_center_field_over_on_axis"
CENTER_FIELD_PROGRESS_EVERY = 10


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from add_max_dipole_center_field_to_results import (  # noqa: E402
    has_cached_center_field_metrics,
    update_one_run,
)

OUTPUTS_ROOT = (SCRIPT_DIR / "../outputs").resolve()
PLOT_OUT_DIR = OUTPUTS_ROOT / "postprocess_plots"


# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults (matched to pareto script style)
# ---------------------------------------------------------------------------
matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
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


def compute_pareto_front(field_errors: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated points minimizing both x and y."""
    order = np.argsort(field_errors)
    pareto_idx = []
    best_y = np.inf
    for idx in order:
        if y_values[idx] <= best_y:
            pareto_idx.append(idx)
            best_y = y_values[idx]
    return np.array(pareto_idx, dtype=int)


def _float_from_results(res: dict[str, Any], key: str) -> float | None:
    val = res.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    graph = res.get("graph")
    if isinstance(graph, dict):
        val = graph.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def iter_candidate_run_dirs(npol: int):
    """Yield run directories under the npol campaign that have results.json."""
    pattern = f"stage_2_npol{npol}_ntor{NTOR_VALUE}_{EQ_NAME}*"
    for campaign_dir in sorted(OUTPUTS_ROOT.glob(pattern)):
        if not campaign_dir.is_dir():
            continue
        if campaign_dir.name.startswith("stage_2") and campaign_dir.name.endswith("_sparse"):
            continue
        for run_dir in sorted(campaign_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "results.json").is_file():
                yield run_dir


def ensure_dipole_center_field_metrics(run_dirs: list[Path]) -> None:
    """
    Compute and persist max dipole center-field metrics where missing.

    Skips runs that already have max_dipole_center_field_over_on_axis in
    results.json. Writes updated results.json via update_one_run().
    """
    need_compute: list[Path] = []
    cached = 0
    for run_dir in run_dirs:
        results_path = run_dir / "results.json"
        try:
            with open(results_path, "r") as f:
                res = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if has_cached_center_field_metrics(res):
            cached += 1
        elif (run_dir / "bs_opt.json").is_file():
            need_compute.append(run_dir)

    total = len(need_compute)
    print(
        f"Center-field metrics: {cached} cached, {total} to compute "
        f"(of {len(run_dirs)} candidate runs with results.json)"
    )
    if total == 0:
        return

    computed = 0
    failed = 0
    for idx, run_dir in enumerate(need_compute, start=1):
        ok, msg = update_one_run(run_dir, cli_field_on_axis=None, dry_run=False)
        if ok:
            computed += 1
            if idx == total or computed % CENTER_FIELD_PROGRESS_EVERY == 0:
                print(f"  [{idx}/{total}] {run_dir.name}: {msg}")
        else:
            failed += 1
            print(f"  [FAIL] [{idx}/{total}] {run_dir}: {msg}")

    print(f"Center-field pass finished: {computed} computed, {failed} failed.")


def load_records_for_npol(npol: int) -> list[dict]:
    """Load all valid results.json records for a single npol campaign."""
    pattern = f"stage_2_npol{npol}_ntor{NTOR_VALUE}_{EQ_NAME}*"
    records: list[dict] = []

    for campaign_dir in sorted(OUTPUTS_ROOT.glob(pattern)):
        if not campaign_dir.is_dir():
            continue
        if campaign_dir.name.startswith("stage_2") and campaign_dir.name.endswith("_sparse"):
            continue
        for run_dir in sorted(campaign_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            results_path = run_dir / "results.json"
            if not results_path.is_file():
                continue
            try:
                with open(results_path, "r") as f:
                    res = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            avg_bn = res.get("avg_Bnormal")
            center_field = _float_from_results(res, Y_METRIC_KEY)
            eq = res.get("eq_name")
            rec_npol = res.get("npoloidal")
            rec_ntor = res.get("ntoroidal")
            if (
                avg_bn is None
                or center_field is None
                or eq != EQ_NAME
                or rec_npol is None
                or rec_ntor is None
            ):
                continue
            if int(rec_npol) != npol or int(rec_ntor) != NTOR_VALUE:
                continue

            records.append(
                {
                    "run_dir": run_dir,
                    "avg_Bnormal": float(avg_bn),
                    Y_METRIC_KEY: center_field,
                }
            )
    return records


def make_multi_panel_plot(records_by_npol: dict[int, list[dict]]) -> Path:
    """Create and save the multi-panel figure."""
    PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_panels = len(NPOL_VALUES)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(8.6, 3.5),
        sharey=True,
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, npol in zip(axes, NPOL_VALUES):
        records = records_by_npol.get(npol, [])
        ax.set_title(r"$N_{\theta}$ =" + f" {npol}")
        ax.grid(True, alpha=0.25, zorder=0)

        if not records:
            ax.text(
                0.5,
                0.5,
                "No valid runs found",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xlabel(r"$\langle \mathbf{B}\cdot \mathbf{n} / |\mathbf{B}| \rangle $")
            continue

        fe = np.array([r[FIELD_ERROR_METRIC] for r in records], dtype=float)
        y_vals = np.array([r[Y_METRIC_KEY] for r in records], dtype=float)

        pareto_idx = compute_pareto_front(fe, y_vals)
        pareto_sorted = pareto_idx[np.argsort(fe[pareto_idx])]

        ax.scatter(
            fe,
            y_vals,
            c="tab:blue",
            s=44,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
            # label="All samples",
        )
        if len(pareto_sorted) > 0:
            ax.plot(
                fe[pareto_sorted],
                y_vals[pareto_sorted],
                "-o",
                color="k",
                linewidth=1.8,
                markersize=4.5,
                zorder=4,
                label="Pareto front",
            )
            pareto_under_threshold = [idx for idx in pareto_sorted if fe[idx] <= FIELD_ERROR_THRESHOLD]
            if pareto_under_threshold:
                init_idx = min(pareto_under_threshold, key=lambda idx: y_vals[idx])
                ax.scatter(
                    [fe[init_idx]],
                    [y_vals[init_idx]],
                    marker="*",
                    s=180,
                    c="limegreen",
                    edgecolors="k",
                    linewidths=0.7,
                    zorder=5,
                    label="Initial condition",
                )

        ax.axvline(
            FIELD_ERROR_THRESHOLD,
            color="tab:red",
            linestyle="--",
            linewidth=1.3,
            zorder=2,
            label="Error threshold",
        )
        ax.set_xlabel(r"$\langle \mathbf{B}\cdot \mathbf{n} / |\mathbf{B}| \rangle $")
        ax.set_xlim(2.5e-3, 9e-3)
        ax.set_ylim(0, 14.0)
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_scientific(True)
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

    axes[0].set_ylabel(r"$\max(B_{\mathrm{dipole,0}}) \,/\, B_T$")

    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for h, l in zip(ax_handles, ax_labels):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            framealpha=0.95,
            edgecolor="0.7",
            bbox_to_anchor=(0.37, 0.92),
        )

    out_path = (
        PLOT_OUT_DIR
        / f"stage2_multipanel_pareto_center_field_ntor{NTOR_VALUE}_{EQ_NAME}.png"
    )
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main():
    print(f"Outputs root: {OUTPUTS_ROOT}")
    print(f"Equilibrium (hardcoded): {EQ_NAME}")
    print(f"npol values (hardcoded): {NPOL_VALUES}")
    print(f"ntor (fixed): {NTOR_VALUE}")

    all_run_dirs: list[Path] = []
    for npol in NPOL_VALUES:
        all_run_dirs.extend(iter_candidate_run_dirs(npol))
    all_run_dirs = sorted(set(all_run_dirs))
    ensure_dipole_center_field_metrics(all_run_dirs)

    records_by_npol: dict[int, list[dict]] = {}
    for npol in NPOL_VALUES:
        recs = load_records_for_npol(npol)
        records_by_npol[npol] = recs
        print(f"npol={npol}: loaded {len(recs)} valid runs")

    out_path = make_multi_panel_plot(records_by_npol)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
