#!/usr/bin/env python3
"""
Create a GIF across true-epsilon runs in single-stage scan folders.

The script supports two sweep modes:
- Fix `fcp_threshold` and sweep increasing iota target.
- Fix iota target and sweep increasing `fcp_threshold`.

For each sweep value, the script keeps the highest (mpol, ntor)
resolution found on disk. If multiple runs exist at the same
resolution, the one with smallest final Boozer residual is used.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional fallback when matplotlib is unavailable
    plt = None


_MPOL_DIR_RE = re.compile(r"mpol(\d+)_ntor(\d+)$")
_TRUE_EPS_DIR_RE = re.compile(r"iota([\d.]+)_fcp([\d.]+)kA(?:_vt[\d.]+)?$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("../single_stage_true_epsilon_sequential"),
        help="Root directory containing single-stage true-epsilon scan runs.",
    )
    sweep_group = p.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument(
        "--fcp-threshold",
        type=float,
        help=(
            "Fix fcp threshold (in amps) to this value and sweep increasing iota target. "
            "Mutually exclusive with --iota-target."
        ),
    )
    sweep_group.add_argument(
        "--iota-target",
        type=float,
        help=(
            "Fix iota target to this value and sweep increasing fcp threshold. "
            "Mutually exclusive with --fcp-threshold."
        ),
    )
    p.add_argument(
        "--frame-file",
        type=str,
        default="x_section_optimized.png",
        help="Filename to load from each run directory for GIF frames.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GIF path (default: <scan-dir>/postprocess_plots/<mode-specific-name>.gif).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=1.2,
        help="Frames per second for the GIF.",
    )
    text_overlay_group = p.add_mutually_exclusive_group()
    text_overlay_group.add_argument(
        "--text-overlay",
        dest="text_overlay",
        action="store_true",
        help="Overlay frame annotation in the top-left corner (default: enabled).",
    )
    text_overlay_group.add_argument(
        "--no-text-overlay",
        dest="text_overlay",
        action="store_false",
        help="Disable frame annotation overlay.",
    )
    p.set_defaults(text_overlay=True)
    p.add_argument(
        "--overlay-mode",
        choices=["iota_only", "iota_and_fcp"],
        default="iota_only",
        help="Overlay text content style (default: iota_only).",
    )
    return p.parse_args()


def _extract_mpol_ntor(run_dir: Path) -> tuple[int, int] | None:
    for part in run_dir.parts:
        m = _MPOL_DIR_RE.fullmatch(part)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def _extract_true_epsilon_targets_from_path(run_dir: Path) -> tuple[float | None, float | None]:
    for part in run_dir.parts:
        m = _TRUE_EPS_DIR_RE.fullmatch(part)
        if m:
            return float(m.group(1)), float(m.group(2)) * 1000.0
    return None, None


def _load_true_epsilon_metadata(run_dir: Path) -> tuple[float | None, float | None, float]:
    path = run_dir / "iterations.json"
    iota_target, fcp_threshold = _extract_true_epsilon_targets_from_path(run_dir)
    last_boozer = math.inf
    if not path.is_file():
        return iota_target, fcp_threshold, last_boozer

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return iota_target, fcp_threshold, last_boozer

    cfg = data.get("config") if isinstance(data.get("config"), dict) else {}
    if iota_target is None:
        v = cfg.get("iota_target")
        if isinstance(v, (int, float)):
            iota_target = float(v)
    if fcp_threshold is None:
        v = cfg.get("f_cp_threshold")
        if isinstance(v, (int, float)):
            fcp_threshold = float(v)

    iterations = data.get("iterations")
    if isinstance(iterations, list) and iterations:
        last = iterations[-1]
        if isinstance(last, dict):
            boozer = last.get("boozer_residual")
            if isinstance(boozer, (int, float)):
                last_boozer = float(boozer)
    return iota_target, fcp_threshold, last_boozer


def _load_frame(path: Path, add_label: str | None = None) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    if add_label is None:
        return im.convert("RGB")

    def _render_label_rgba(text: str, target_width: int) -> Image.Image:
        if plt is not None:
            fontsize = 8
            fig = plt.figure(figsize=(8, 1.8), dpi=250, facecolor=(1, 1, 1, 0))
            fig.text(0.02, 0.5, text, fontsize=fontsize, family="serif", va="center", ha="left")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf).convert("RGBA")

    math_im = _render_label_rgba(add_label, im.size[0])
    panel_pad_x = max(14, int(0.012 * im.size[0]))
    panel_pad_y = max(10, int(0.012 * im.size[1]))
    box_w = math_im.size[0] + 2 * panel_pad_x
    box_h = math_im.size[1] + 2 * panel_pad_y
    margin = max(18, int(0.02 * min(im.size)))
    left, top = margin, margin

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (left, top, left + box_w, top + box_h),
        radius=max(12, int(0.01 * min(im.size))),
        fill=(255, 255, 255, 232),
        outline=(30, 30, 30, 220),
        width=3,
    )
    overlay.paste(math_im, (left + panel_pad_x, top + panel_pad_y), math_im)
    return Image.alpha_composite(im, overlay).convert("RGB")


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir.resolve()
    if not scan_dir.is_dir():
        raise SystemExit(f"scan dir not found: {scan_dir}")

    if args.out is None:
        out_dir = scan_dir / "postprocess_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.fcp_threshold is not None:
            out_path = out_dir / f"iota_increasing_fcp{args.fcp_threshold:g}.gif"
        else:
            out_path = out_dir / f"fcp_increasing_iota{args.iota_target:g}.gif"
    else:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keyed by the swept variable:
    # - fixed --fcp-threshold: key is iota target
    # - fixed --iota-target: key is fcp threshold
    candidates: dict[float, dict] = {}
    n_total = 0
    n_reject = {
        "missing_resolution": 0,
        "missing_iota": 0,
        "missing_fcp": 0,
        "iota_mismatch": 0,
        "fcp_mismatch": 0,
        "missing_frame": 0,
    }

    for run_dir in sorted(scan_dir.rglob("mpol*_ntor*")):
        if not run_dir.is_dir():
            continue
        n_total += 1

        res = _extract_mpol_ntor(run_dir)
        if res is None:
            n_reject["missing_resolution"] += 1
            continue
        mpol, ntor = res

        iota_tar, fcp_threshold, boozer = _load_true_epsilon_metadata(run_dir)
        if iota_tar is None:
            n_reject["missing_iota"] += 1
            continue
        if fcp_threshold is None:
            n_reject["missing_fcp"] += 1
            continue
        if args.fcp_threshold is not None:
            if not math.isclose(fcp_threshold, args.fcp_threshold, rel_tol=0.0, abs_tol=1e-12):
                n_reject["fcp_mismatch"] += 1
                continue
            sweep_key = iota_tar
        else:
            if not math.isclose(iota_tar, args.iota_target, rel_tol=0.0, abs_tol=1e-12):
                n_reject["iota_mismatch"] += 1
                continue
            sweep_key = fcp_threshold

        frame_path = run_dir / args.frame_file
        if not frame_path.is_file():
            n_reject["missing_frame"] += 1
            continue

        record = {
            "run_dir": run_dir,
            "frame_path": frame_path,
            "iota": iota_tar,
            "fcp": fcp_threshold,
            "mpol": mpol,
            "ntor": ntor,
            "resolution": (mpol, ntor),
            "boozer": math.inf if boozer is None else boozer,
        }
        prev = candidates.get(sweep_key)
        if (
            prev is None
            or record["resolution"] > prev["resolution"]
            or (record["resolution"] == prev["resolution"] and record["boozer"] < prev["boozer"])
        ):
            candidates[sweep_key] = record

    selected = [candidates[k] for k in sorted(candidates.keys())]
    if not selected:
        print(f"Scan root: {scan_dir}")
        print(f"Total mpol*_ntor* dirs seen: {n_total}")
        for key, val in n_reject.items():
            print(f"Rejected ({key}): {val}")
        raise SystemExit("No matching runs found after filtering.")

    frames = []
    for rec in selected:
        label = None
        if args.text_overlay:
            if args.iota_target is not None:
                # When sweeping fcp at fixed iota target, annotate frames by fcp.
                label = rf"$f_{{\mathrm{{CP}}}}={rec['fcp']:.6g}$"
            elif args.overlay_mode == "iota_and_fcp":
                label = rf"$\iota_{{\mathrm{{tar}}}}={rec['iota']:.6g}\quad f_{{\mathrm{{CP}}}}={rec['fcp']:.6g}$"
            else:
                label = rf"$\iota_{{\mathrm{{tar}}}}={rec['iota']:.6g}$"
        frames.append(_load_frame(rec["frame_path"], add_label=label))

    # Ensure equal shape for GIF encoder by padding/cropping to first frame size.
    w0, h0 = frames[0].size
    normalized_frames = []
    for im in frames:
        if im.size != (w0, h0):
            im = im.resize((w0, h0), Image.Resampling.LANCZOS)
        normalized_frames.append(im)

    duration_ms = int(round(1000.0 / args.fps))
    normalized_frames[0].save(
        out_path,
        save_all=True,
        append_images=normalized_frames[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )

    print(f"Scan root: {scan_dir}")
    print(f"Total mpol*_ntor* dirs seen: {n_total}")
    for key, val in n_reject.items():
        print(f"Rejected ({key}): {val}")
    print(f"Selected frames: {len(selected)}")
    print(f"Frame file: {args.frame_file}")
    print(f"Output GIF: {out_path}")
    if args.fcp_threshold is not None:
        print("Included runs (sorted by iota):")
        for rec in selected:
            print(
                f"  iota_tar={rec['iota']:.6g}  mpol={rec['mpol']} ntor={rec['ntor']}  "
                f"boozer={rec['boozer']:.3e}  {rec['run_dir']}"
            )
    else:
        print("Included runs (sorted by fcp):")
        for rec in selected:
            print(
                f"  fcp={rec['fcp']:.6g}  mpol={rec['mpol']} ntor={rec['ntor']}  "
                f"boozer={rec['boozer']:.3e}  {rec['run_dir']}"
            )


if __name__ == "__main__":
    main()

