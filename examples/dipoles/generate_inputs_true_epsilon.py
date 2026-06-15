import argparse
import itertools
import numpy as np
import sys

def parse_fcp_thresholds(text):
    vals = [float(x) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Need at least one f_CP threshold.")
    return vals

def build_cases(iota_min, iota_max, iota_count, fcp_thresholds):
    iota_targets = np.linspace(iota_min, iota_max, iota_count)
    return list(itertools.product(iota_targets, fcp_thresholds))

def main():
    parser = argparse.ArgumentParser(description="Parameter generator for true epsilon-constraint single-stage scans.")
    parser.add_argument("--iota-min", type=float, default=0.10, help="Minimum iota target.")
    parser.add_argument("--iota-max", type=float, default=0.25, help="Maximum iota target.")
    parser.add_argument("--iota-count", type=int, default=16, help="Number of iota targets.")
    parser.add_argument(
        "--f-cp-thresholds",
        type=str,
        default="50000,100000,150000,200000",
        help="Comma-separated current thresholds in A.",
    )
    parser.add_argument("--f-b-threshold", type=float, default=1e-4, help="Flux-surface threshold.")
    parser.add_argument("--iota-threshold", type=float, default=0.0025, help="Absolute iota tolerance.")
    parser.add_argument("--mpol", type=int, default=6, help="Boozer surface mpol.")
    parser.add_argument("--ntor", type=int, default=6, help="Boozer surface ntor.")
    parser.add_argument("--maxiter", type=int, default=200, help="BFGS max iterations.")
    parser.add_argument("--volume-target", type=float, default=0.3, help="Boozer surface volume target.")
    parser.add_argument("--index", type=int, default=None, help="Case index.")
    parser.add_argument("--list", action="store_true", help="List all cases.")
    parser.add_argument("--total", action="store_true", help="Print total number of cases.")
    args = parser.parse_args()

    if args.iota_count < 1:
        raise ValueError("--iota-count must be >= 1.")
    if args.iota_max < args.iota_min:
        raise ValueError("--iota-max must be >= --iota-min.")

    fcp_thresholds = parse_fcp_thresholds(args.f_cp_thresholds)
    cases = build_cases(args.iota_min, args.iota_max, args.iota_count, fcp_thresholds)

    if args.total:
        print(len(cases))
        return

    if args.list:
        print(f"Total sweep points: {len(cases)}")
        print(f"{'idx':>4}  {'iota_target':>12}  {'f_cp_threshold[A]':>18}")
        for idx, (iota_target, fcp_threshold) in enumerate(cases):
            print(f"{idx:4d}  {iota_target:12.6f}  {fcp_threshold:18.1f}")
        return

    if args.index is None:
        print("Error: --index required unless using --list or --total.", file=sys.stderr)
        sys.exit(1)
    if args.index < 0 or args.index >= len(cases):
        print(f"Error: --index must be in [0, {len(cases)}).", file=sys.stderr)
        sys.exit(1)

    iota_target, fcp_threshold = cases[args.index]
    print(
        f"--iota-target {iota_target} "
        f"--f-cp-threshold {fcp_threshold} "
        f"--f-b-threshold {args.f_b_threshold} "
        f"--iota-threshold {args.iota_threshold} "
        f"--mpol {args.mpol} "
        f"--ntor {args.ntor} "
        f"--maxiter {args.maxiter} "
        f"--volume-target {args.volume_target}"
    )

if __name__ == "__main__":
    main()
