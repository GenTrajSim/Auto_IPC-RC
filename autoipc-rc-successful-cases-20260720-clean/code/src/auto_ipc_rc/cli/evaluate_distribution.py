from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from auto_ipc_rc.distribution import compare_joint_distribution, compare_pc_distribution, load_reference_rows, validate_reference_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare predicted and reference AutoIPC PC distributions.")
    parser.add_argument("--predicted", required=True, help="Predicted text file with columns potential_i rho_i pc_i")
    parser.add_argument("--reference", required=True, help="Reference text file with columns potential_i rho_i pc_i")
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", required=True, help="Metrics JSON output path")
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args(argv)

    predicted = load_reference_rows(args.predicted, expected_rows=args.expected_rows)
    reference = load_reference_rows(args.reference, expected_rows=args.expected_rows)
    payload = {
        "predicted": validate_reference_rows(predicted, expected_rows=args.expected_rows),
        "reference": validate_reference_rows(reference, expected_rows=args.expected_rows),
        "pc": compare_pc_distribution(predicted[:, 2], reference[:, 2], bins=args.bins),
        "rho": compare_pc_distribution(predicted[:, 1], reference[:, 1], bins=args.bins),
        "potential": compare_pc_distribution(predicted[:, 0], reference[:, 0], bins=args.bins),
        "potential_pc_2d": compare_joint_distribution(predicted[:, [0, 2]], reference[:, [0, 2]], bins=args.bins),
        "rho_pc_2d": compare_joint_distribution(predicted[:, [1, 2]], reference[:, [1, 2]], bins=args.bins),
        "potential_rho_2d": compare_joint_distribution(predicted[:, [0, 1]], reference[:, [0, 1]], bins=args.bins),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
