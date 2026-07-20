#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from auto_ipc_rc.fig2d_gate import compare_fig2d_to_reference


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill reference-relative Fig2d dual-mode gates for completed eval epochs.")
    parser.add_argument("--eval-root", required=True, help="Directory containing epoch_* Fig2d evaluation folders")
    parser.add_argument("--ref-pci", required=True, help="Reference rows for PCI, columns potential rho pc")
    parser.add_argument("--ref-pcii", required=True, help="Reference rows for PCII, columns potential rho pc")
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--prominence-fraction", type=float, default=0.08)
    parser.add_argument("--min-peak-distance-bins", type=int, default=10)
    parser.add_argument("--max-valley-fraction", type=float, default=0.25)
    parser.add_argument("--force", action="store_true", help="Recompute gates that already exist")
    args = parser.parse_args(argv)

    eval_root = Path(args.eval_root)
    refs = {
        "pci": Path(args.ref_pci),
        "pcii": Path(args.ref_pcii),
    }
    reference_rows = {label: np.loadtxt(path, dtype=np.float64) for label, path in refs.items()}
    report: list[dict[str, Any]] = []

    for epoch_dir in sorted(eval_root.glob("epoch_*")):
        if not epoch_dir.is_dir():
            continue
        pred_dir = epoch_dir / "p1800_t188_predictions"
        epoch_report: dict[str, Any] = {"epoch_dir": str(epoch_dir), "labels": {}}
        for label, filename in [("pci", "PCI_predicted_rows.txt"), ("pcii", "PCII_predicted_rows.txt")]:
            rows_path = pred_dir / filename
            output_path = epoch_dir / f"{label}_fig2d_gate.json"
            if not rows_path.exists():
                epoch_report["labels"][label] = {"status": "missing_predictions"}
                continue
            if output_path.exists() and not args.force:
                result = json.loads(output_path.read_text(encoding="utf-8"))
                status = "existing"
            else:
                rows = np.loadtxt(rows_path, dtype=np.float64)
                result = compare_fig2d_to_reference(
                    rows,
                    reference_rows[label],
                    bins=args.bins,
                    prominence_fraction=args.prominence_fraction,
                    min_peak_distance_bins=args.min_peak_distance_bins,
                    max_valley_fraction=args.max_valley_fraction,
                )
                output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                status = "computed"
            epoch_report["labels"][label] = _compact_gate(status, result)
        _update_summary(epoch_dir)
        report.append(epoch_report)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _compact_gate(status: str, result: dict[str, Any]) -> dict[str, Any]:
    predicted = result.get("predicted", {})
    reference = result.get("reference", {})
    return {
        "status": status,
        "passed": result.get("passed"),
        "reason": result.get("reason"),
        "distance_ratio": result.get("distance_ratio"),
        "allowed_valley_fraction": result.get("allowed_valley_fraction"),
        "pred_peak_count": predicted.get("peak_count"),
        "pred_peak_distance_bins": predicted.get("best_peak_distance_bins"),
        "pred_valley_fraction": predicted.get("best_valley_fraction"),
        "ref_peak_distance_bins": reference.get("best_peak_distance_bins"),
        "ref_valley_fraction": reference.get("best_valley_fraction"),
    }


def _update_summary(epoch_dir: Path) -> None:
    summary_path = epoch_dir / "fig2d_metrics_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    changed = False
    for label in ["pci", "pcii"]:
        gate_path = epoch_dir / f"{label}_fig2d_gate.json"
        if not gate_path.exists():
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        summary.setdefault(label, {})
        compact = _compact_gate("computed", gate)
        summary[label].update({
            "dual_mode_passed": compact["passed"],
            "dual_mode_reason": compact["reason"],
            "distance_ratio": compact["distance_ratio"],
            "allowed_valley_fraction": compact["allowed_valley_fraction"],
            "pred_peak_count": compact["pred_peak_count"],
            "pred_peak_distance_bins": compact["pred_peak_distance_bins"],
            "pred_valley_fraction": compact["pred_valley_fraction"],
            "ref_peak_distance_bins": compact["ref_peak_distance_bins"],
            "ref_valley_fraction": compact["ref_valley_fraction"],
        })
        changed = True
    if changed:
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
