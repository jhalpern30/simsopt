#!/usr/bin/env python3
"""
Create a GIF across increasing iota at fixed CURRENT_WEIGHT.

Filters applied by default:
- Exclude paths containing "sparse" or "sparsity" (case-insensitive).
- Use only iota folders without "_vol" suffix (implicit volume target 0.3).
- Use only run directories named mpol6_ntor*.

Each GIF frame is loaded from a file inside each run directory (default: modB_plot.png).
If multiple runs match the same iota target, the one with smallest final J_Boozer is used.
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


_IOTA_DIR_RE = re.compile(r"iota_tar([\d.]+)$")
_MPOL_DIR_RE = re.compile(r"mpol(\d+)_ntor(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        help="Root directory containing single-stage scan runs.",
    )
    sweep_group = p.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument(
        "--w-cp",
        type=float,
        help=(
            "Fix CURRENT_WEIGHT to this value and sweep increasing iota target. "
            "Mutually exclusive with --iota-target."
        ),
    )
    sweep_group.add_argument(
        "--iota-target",
        type=float,
        help=(
            "Fix iota target to this value and sweep increasing CURRENT_WEIGHT. "
            "Mutually exclusive with --w-cp."
        ),
    )
    p.add_argument(
        "--frame-file",
        type=str,
        default="modB_plot.png",
        help="Filename to load from each run directory for GIF frames.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GIF path (default: <scan-dir>/postprocess_plots/iota_increasing_wc<value>.gif).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=1.2,
        help="Frames per second for the GIF.",
    )
    p.add_argument(
        "--text-overlay",
        action="store_true",
        help="Overlay frame annotation in the top-left corner.",
    )
    p.add_argument(
        "--overlay-mode",
        choices=["iota_only", "iota_and_wc"],
        default="iota_only",
        help="Overlay text content style (default: iota_only).",
    )
    return p.parse_args()


def _extract_iota_target(run_dir: Path) -> float | None:
    for part in run_dir.parts:
        m = _IOTA_DIR_RE.fullmatch(part)
        if m:
            return float(m.group(1))
    return None


def _run_is_mpol6(run_dir: Path) -> bool:
    for part in run_dir.parts:
        m = _MPOL_DIR_RE.fullmatch(part)
        if not m:
            continue
        return int(m.group(1)) == 6
    return False


def _path_contains_sparse_token(run_dir: Path) -> bool:
    return any(("sparse" in part.lower()) or ("sparsity" in part.lower()) for part in run_dir.parts)


def _path_is_default_volume_iota_dir(run_dir: Path) -> bool:
    # Keep only iota_tar* directories without explicit _vol* suffix.
    for part in run_dir.parts:
        if part.startswith("iota_tar"):
            return "_vol" not in part
    return False


def _load_iterations(run_dir: Path) -> tuple[float | None, float | None]:
    path = run_dir / "iterations.json"
    if not path.is_file():
        return None, None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    cw = None
    weights = data.get("weights")
    if isinstance(weights, dict):
        v = weights.get("CURRENT_WEIGHT")
        if isinstance(v, (int, float)):
            cw = float(v)

    boozer = None
    iterations = data.get("iterations")
    if isinstance(iterations, list) and iterations:
        last = iterations[-1]
        if isinstance(last, dict):
            b = last.get("J_Boozer")
            if isinstance(b, (int, float)):
                boozer = float(b)
    return cw, boozer


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
        if args.w_cp is not None:
            out_path = out_dir / f"iota_increasing_wc{args.w_cp:g}.gif"
        else:
            out_path = out_dir / f"wc_increasing_iota{args.iota_target:g}.gif"
    else:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keyed by the swept variable:
    # - fixed --w-cp: key is iota target
    # - fixed --iota-target: key is current weight
    candidates: dict[float, dict] = {}
    n_total = 0
    n_reject = {
        "sparse": 0,
        "vol": 0,
        "mpol": 0,
        "missing_iota": 0,
        "iota_mismatch": 0,
        "cw_mismatch": 0,
        "missing_frame": 0,
    }

    for run_dir in sorted(scan_dir.rglob("mpol*_ntor*")):
        if not run_dir.is_dir():
            continue
        n_total += 1

        if _path_contains_sparse_token(run_dir):
            n_reject["sparse"] += 1
            continue
        if not _path_is_default_volume_iota_dir(run_dir):
            n_reject["vol"] += 1
            continue
        if not _run_is_mpol6(run_dir):
            n_reject["mpol"] += 1
            continue

        iota_tar = _extract_iota_target(run_dir)
        if iota_tar is None:
            n_reject["missing_iota"] += 1
            continue

        current_weight, boozer = _load_iterations(run_dir)
        if current_weight is None:
            n_reject["cw_mismatch"] += 1
            continue
        if args.w_cp is not None:
            if not math.isclose(current_weight, args.w_cp, rel_tol=0.0, abs_tol=1e-12):
                n_reject["cw_mismatch"] += 1
                continue
            sweep_key = iota_tar
        else:
            if not math.isclose(iota_tar, args.iota_target, rel_tol=0.0, abs_tol=1e-12):
                n_reject["iota_mismatch"] += 1
                continue
            sweep_key = current_weight

        frame_path = run_dir / args.frame_file
        if not frame_path.is_file():
            n_reject["missing_frame"] += 1
            continue

        record = {
            "run_dir": run_dir,
            "frame_path": frame_path,
            "iota": iota_tar,
            "cw": current_weight,
            "boozer": math.inf if boozer is None else boozer,
        }
        prev = candidates.get(sweep_key)
        if prev is None or record["boozer"] < prev["boozer"]:
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
                # When sweeping current weight at fixed iota target, annotate frames by w_cp.
                label = rf"$w_{{\mathrm{{CP}}}}={rec['cw']:.6g}$"
            elif args.overlay_mode == "iota_and_wc":
                label = rf"$\iota_{{\mathrm{{tar}}}}={rec['iota']:.6g}\quad w_{{\mathrm{{CP}}}}={rec['cw']:.6g}$"
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
    if args.w_cp is not None:
        print("Included runs (sorted by iota):")
        for rec in selected:
            print(f"  iota_tar={rec['iota']:.6g}  J_Boozer={rec['boozer']:.3e}  {rec['run_dir']}")
    else:
        print("Included runs (sorted by w_cp):")
        for rec in selected:
            print(f"  w_cp={rec['cw']:.6g}  J_Boozer={rec['boozer']:.3e}  {rec['run_dir']}")


if __name__ == "__main__":
    main()

