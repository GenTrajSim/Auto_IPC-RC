from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def evaluate_fig2d_dual_mode(
    rows: np.ndarray,
    *,
    bins: int = 120,
    prominence_fraction: float = 0.08,
    min_peak_distance_bins: int = 10,
    max_valley_fraction: float = 0.25,
) -> dict[str, Any]:
    data = _rows(rows)
    hist_raw, x_edges, y_edges = np.histogram2d(data[:, 1], data[:, 2], bins=int(bins))
    hist = gaussian_filter(hist_raw.astype(np.float64), sigma=1.25)
    if hist.size == 0 or float(hist.max()) <= 0.0:
        return _result(False, hist, [], None, reason="empty_histogram")

    neighborhood = max(5, int(min_peak_distance_bins) // 2 * 2 + 1)
    local_max = hist == maximum_filter(hist, size=neighborhood, mode="constant")
    effective_prominence = max(float(prominence_fraction), 0.50)
    threshold = float(hist.max()) * effective_prominence
    candidates = np.argwhere(local_max & (hist >= threshold))
    peaks = [
        {
            "x_bin": int(i),
            "y_bin": int(j),
            "height": float(hist[i, j]),
            "rho": float(0.5 * (x_edges[i] + x_edges[i + 1])),
            "pc": float(0.5 * (y_edges[j] + y_edges[j + 1])),
        }
        for i, j in candidates
    ]
    peaks.sort(key=lambda item: item["height"], reverse=True)

    best_pair = None
    for left_index, left in enumerate(peaks):
        for right in peaks[left_index + 1 :]:
            dist = math.hypot(left["x_bin"] - right["x_bin"], left["y_bin"] - right["y_bin"])
            if dist < float(min_peak_distance_bins):
                continue
            valley_fraction = _line_valley_fraction(hist, left, right)
            pair = {
                "left": left,
                "right": right,
                "distance_bins": float(dist),
                "valley_fraction": float(valley_fraction),
                "score": float(dist * (1.0 - min(valley_fraction, 1.0))),
            }
            if best_pair is None or pair["score"] > best_pair["score"]:
                best_pair = pair

    passed = bool(best_pair is not None and best_pair["valley_fraction"] <= float(max_valley_fraction))
    reason = "passed" if passed else ("no_separated_peak_pair" if best_pair is None else "valley_not_deep_enough")
    return _result(passed, hist, peaks, best_pair, reason=reason)


def compare_fig2d_to_reference(
    predicted_rows: np.ndarray,
    reference_rows: np.ndarray,
    *,
    bins: int = 120,
    prominence_fraction: float = 0.08,
    min_peak_distance_bins: int = 10,
    max_valley_fraction: float = 0.25,
    min_distance_ratio: float = 0.50,
    max_valley_ratio: float = 2.50,
) -> dict[str, Any]:
    reference = evaluate_fig2d_dual_mode(
        reference_rows,
        bins=bins,
        prominence_fraction=prominence_fraction,
        min_peak_distance_bins=min_peak_distance_bins,
        max_valley_fraction=max_valley_fraction,
    )
    predicted = evaluate_fig2d_dual_mode(
        predicted_rows,
        bins=bins,
        prominence_fraction=prominence_fraction,
        min_peak_distance_bins=min_peak_distance_bins,
        max_valley_fraction=max_valley_fraction,
    )
    ref_distance = float(reference["best_peak_distance_bins"])
    pred_distance = float(predicted["best_peak_distance_bins"])
    distance_ratio = pred_distance / ref_distance if ref_distance > 0.0 else 0.0
    ref_valley = float(reference["best_valley_fraction"])
    pred_valley = float(predicted["best_valley_fraction"])
    allowed_valley = min(0.85, max(float(max_valley_fraction), ref_valley * float(max_valley_ratio), ref_valley + 0.10))
    reference_has_dual_peaks = bool(reference["best_pair"] is not None)
    predicted_has_dual_peaks = bool(predicted["best_pair"] is not None)
    predicted_passed = bool(predicted_has_dual_peaks and pred_valley <= allowed_valley)
    passed = bool(reference_has_dual_peaks and predicted_passed and distance_ratio >= float(min_distance_ratio))
    if not reference_has_dual_peaks:
        reason = "reference_has_no_dual_peak_pair"
    elif not predicted_has_dual_peaks:
        reason = "predicted_has_no_dual_peak_pair"
    elif not predicted_passed:
        reason = "predicted_valley_too_shallow_vs_reference"
    elif distance_ratio < float(min_distance_ratio):
        reason = "predicted_peak_distance_too_small"
    else:
        reason = "passed"
    return {
        "passed": passed,
        "reason": reason,
        "distance_ratio": float(distance_ratio),
        "allowed_valley_fraction": float(allowed_valley),
        "predicted": predicted,
        "reference": reference,
    }


def _rows(rows: np.ndarray) -> np.ndarray:
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Fig2d rows must have shape [N,3], got {data.shape}")
    if data.shape[0] == 0:
        raise ValueError("Fig2d rows are empty")
    if not np.isfinite(data).all():
        raise ValueError("Fig2d rows contain NaN or inf")
    return data


def _line_valley_fraction(hist: np.ndarray, left: dict[str, Any], right: dict[str, Any]) -> float:
    x0, y0 = int(left["x_bin"]), int(left["y_bin"])
    x1, y1 = int(right["x_bin"]), int(right["y_bin"])
    steps = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(int)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(int)
    values = hist[xs, ys]
    if values.size <= 2:
        valley = float(np.min(values))
    else:
        valley = float(np.min(values[1:-1]))
    peak_floor = float(min(left["height"], right["height"]))
    return valley / peak_floor if peak_floor > 0.0 else 1.0


def _result(passed: bool, hist: np.ndarray, peaks: list[dict[str, Any]], best_pair: dict[str, Any] | None, *, reason: str) -> dict[str, Any]:
    return {
        "passed": passed,
        "reason": reason,
        "peak_count": int(len(peaks)),
        "top_peaks": peaks[:5],
        "best_pair": best_pair,
        "best_peak_distance_bins": float(best_pair["distance_bins"]) if best_pair else 0.0,
        "best_valley_fraction": float(best_pair["valley_fraction"]) if best_pair else 1.0,
        "hist_max": float(hist.max()) if hist.size else 0.0,
        "nonzero_bins": int(np.count_nonzero(hist)),
    }
