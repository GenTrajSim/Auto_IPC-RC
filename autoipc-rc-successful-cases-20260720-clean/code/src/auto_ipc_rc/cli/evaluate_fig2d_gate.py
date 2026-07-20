from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from auto_ipc_rc.fig2d_gate import compare_fig2d_to_reference, evaluate_fig2d_dual_mode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate whether Fig2d rows show a separated dual-mode distribution.")
    parser.add_argument("--rows", required=True, help="Predicted text rows with columns potential rho pc")
    parser.add_argument("--reference", default=None, help="Optional reference text rows with columns potential rho pc")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--prominence-fraction", type=float, default=0.08)
    parser.add_argument("--min-peak-distance-bins", type=int, default=10)
    parser.add_argument("--max-valley-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)

    rows = np.loadtxt(args.rows, dtype=np.float64)
    if args.reference:
        reference = np.loadtxt(args.reference, dtype=np.float64)
        result = compare_fig2d_to_reference(
            rows,
            reference,
            bins=args.bins,
            prominence_fraction=args.prominence_fraction,
            min_peak_distance_bins=args.min_peak_distance_bins,
            max_valley_fraction=args.max_valley_fraction,
        )
    else:
        result = evaluate_fig2d_dual_mode(
            rows,
            bins=args.bins,
            prominence_fraction=args.prominence_fraction,
            min_peak_distance_bins=args.min_peak_distance_bins,
            max_valley_fraction=args.max_valley_fraction,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
