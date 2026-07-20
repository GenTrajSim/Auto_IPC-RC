from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Affine-calibrate selected predicted row columns to reference mean/std.")
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--columns", nargs="+", type=int, required=True, help="Zero-based columns to calibrate")
    args = parser.parse_args(argv)

    predicted = _load_rows(Path(args.predicted), name="predicted")
    reference = _load_rows(Path(args.reference), name="reference")
    if predicted.shape != reference.shape:
        raise ValueError(f"predicted/reference shapes must match, got {predicted.shape} and {reference.shape}")

    calibrated = predicted.copy()
    transforms = {}
    for column in args.columns:
        if column < 0 or column >= predicted.shape[1]:
            raise ValueError(f"column {column} out of range for shape {predicted.shape}")
        pred_mean = float(np.mean(predicted[:, column]))
        pred_std = float(np.std(predicted[:, column]))
        ref_mean = float(np.mean(reference[:, column]))
        ref_std = float(np.std(reference[:, column]))
        scale = ref_std / pred_std if pred_std > 0.0 else 1.0
        calibrated[:, column] = (predicted[:, column] - pred_mean) * scale + ref_mean
        transforms[str(column)] = {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "ref_mean": ref_mean,
            "ref_std": ref_std,
            "scale": float(scale),
            "offset": float(ref_mean - scale * pred_mean),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, calibrated, fmt="%.10g")
    summary = {
        "predicted": str(Path(args.predicted)),
        "reference": str(Path(args.reference)),
        "output": str(output),
        "rows": int(calibrated.shape[0]),
        "columns": list(args.columns),
        "transforms": transforms,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _load_rows(path: Path, *, name: str) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        raise ValueError(f"{name} rows must be 2D, got shape {rows.shape}")
    if not np.isfinite(rows).all():
        raise ValueError(f"{name} rows contain NaN or inf")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
