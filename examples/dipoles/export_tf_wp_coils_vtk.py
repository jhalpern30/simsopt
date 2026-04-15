#!/usr/bin/env python3
"""
Export TF and WP coils as separate VTK files for a run directory.

Given a path containing ``bs_opt.json`` (and usually ``results.json``), this script
splits coils into TF and WP sets and writes:
  - tf_coils_currents.vtu
  - wp_coils_currents.vtu

Only coil current data is written (no regularized force/torque fields).
"""

from __future__ import annotations

import argparse
import os

from simsopt._core.optimizable import load
from simsopt.field import coils_to_vtk


def infer_num_tf_coils(results: dict, ncoils_total: int) -> int:
    """Infer TF coil count from results metadata."""
    if "# TF coils" in results:
        return int(results["# TF coils"])

    if "ntf" in results and "surf_nfp" in results:
        return int(results["ntf"]) * 2 * int(results["surf_nfp"])

    if "num_tf_coils" in results:
        return int(results["num_tf_coils"])

    if "num_wps" in results:
        n_tf = ncoils_total - int(results["num_wps"])
        if n_tf >= 0:
            return n_tf

    raise ValueError(
        "Could not infer TF coil count from results.json. "
        "Use --num-tf-coils to set it explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        help="Path to run directory containing bs_opt.json.",
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=None,
        help="Override inferred TF coil count.",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Close each coil polyline by repeating the first point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    bs_path = os.path.join(run_dir, "bs_opt.json")
    results_path = os.path.join(run_dir, "results.json")

    if not os.path.isfile(bs_path):
        raise FileNotFoundError(f"Missing file: {bs_path}")

    bs = load(bs_path)
    coils = list(bs.coils)
    if len(coils) == 0:
        raise RuntimeError(f"No coils found in {bs_path}")

    if args.num_tf_coils is not None:
        num_tf_coils = args.num_tf_coils
    else:
        if not os.path.isfile(results_path):
            raise FileNotFoundError(
                f"Missing {results_path} and no --num-tf-coils provided."
            )
        results = load(results_path)
        num_tf_coils = infer_num_tf_coils(results, len(coils))

    if num_tf_coils < 0 or num_tf_coils > len(coils):
        raise ValueError(
            f"Invalid TF coil count {num_tf_coils}; total coils = {len(coils)}."
        )

    tf_coils = coils[:num_tf_coils]
    wp_coils = coils[num_tf_coils:]

    tf_out = os.path.join(run_dir, "tf_coils_currents")
    wp_out = os.path.join(run_dir, "wp_coils_currents")
    coils_to_vtk(tf_coils, filename=tf_out, close=args.close)
    coils_to_vtk(wp_coils, filename=wp_out, close=args.close)

    print(f"Wrote {len(tf_coils)} TF coils to {tf_out}.vtu")
    print(f"Wrote {len(wp_coils)} WP coils to {wp_out}.vtu")


if __name__ == "__main__":
    main()
