from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def load_reference_rows(path, *, expected_rows: int, expected_columns: int = 3) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    validate_reference_rows(rows, expected_rows=expected_rows, expected_columns=expected_columns)
    return rows


def validate_reference_rows(rows: np.ndarray, *, expected_rows: int, expected_columns: int = 3) -> dict[str, int | bool]:
    arr = np.asarray(rows)
    if arr.ndim != 2:
        raise ValueError(f"reference rows must be a 2D array, got shape {arr.shape}")
    summary = {
        "rows": int(arr.shape[0]),
        "columns": int(arr.shape[1]),
        "finite": bool(np.isfinite(arr).all()),
    }
    if arr.shape[0] != int(expected_rows):
        raise ValueError(f"expected {expected_rows} rows, got {arr.shape[0]}")
    if arr.shape[1] != int(expected_columns):
        raise ValueError(f"expected {expected_columns} columns, got {arr.shape[1]}")
    if not summary["finite"]:
        raise ValueError("reference rows contain NaN or inf")
    return summary


def compare_pc_distribution(predicted: np.ndarray, reference: np.ndarray, *, bins: int = 100) -> dict[str, float | int]:
    pred = _finite_vector(predicted, name="predicted")
    ref = _finite_vector(reference, name="reference")
    hist_pred, hist_ref = _aligned_histograms(pred, ref, bins=bins)
    js = float(jensenshannon(hist_pred, hist_ref, base=2.0) ** 2)
    if abs(js) < 1.0e-15:
        js = 0.0
    ks = ks_2samp(pred, ref, alternative="two-sided", mode="auto")
    return {
        "pred_count": int(pred.size),
        "ref_count": int(ref.size),
        "pred_mean": float(np.mean(pred)),
        "ref_mean": float(np.mean(ref)),
        "pred_std": float(np.std(pred)),
        "ref_std": float(np.std(ref)),
        "pred_q05": float(np.quantile(pred, 0.05)),
        "ref_q05": float(np.quantile(ref, 0.05)),
        "pred_q50": float(np.quantile(pred, 0.50)),
        "ref_q50": float(np.quantile(ref, 0.50)),
        "pred_q95": float(np.quantile(pred, 0.95)),
        "ref_q95": float(np.quantile(ref, 0.95)),
        "wasserstein": float(wasserstein_distance(pred, ref)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "js_divergence": js,
    }


def compare_joint_distribution(predicted: np.ndarray, reference: np.ndarray, *, bins: int = 100) -> dict[str, float | int]:
    pred = _finite_matrix(predicted, name="predicted", columns=2)
    ref = _finite_matrix(reference, name="reference", columns=2)
    pred_prob, ref_prob = _aligned_histograms_2d(pred, ref, bins=bins)
    js = float(jensenshannon(pred_prob.reshape(-1), ref_prob.reshape(-1), base=2.0) ** 2)
    if abs(js) < 1.0e-15:
        js = 0.0
    return {
        "pred_count": int(pred.shape[0]),
        "ref_count": int(ref.shape[0]),
        "pred_x_mean": float(np.mean(pred[:, 0])),
        "ref_x_mean": float(np.mean(ref[:, 0])),
        "pred_y_mean": float(np.mean(pred[:, 1])),
        "ref_y_mean": float(np.mean(ref[:, 1])),
        "js_divergence_2d": js,
    }


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} distribution is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} distribution contains NaN or inf")
    return arr


def _aligned_histograms(pred: np.ndarray, ref: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
    lower = float(min(np.min(pred), np.min(ref)))
    upper = float(max(np.max(pred), np.max(ref)))
    if math.isclose(lower, upper):
        upper = lower + 1.0
    edges = np.linspace(lower, upper, int(bins) + 1)
    pred_hist, _ = np.histogram(pred, bins=edges)
    ref_hist, _ = np.histogram(ref, bins=edges)
    pred_prob = pred_hist.astype(np.float64) + 1.0e-12
    ref_prob = ref_hist.astype(np.float64) + 1.0e-12
    pred_prob /= np.sum(pred_prob)
    ref_prob /= np.sum(ref_prob)
    return pred_prob, ref_prob

def _finite_matrix(values: np.ndarray, *, name: str, columns: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != columns:
        raise ValueError(f"{name} distribution must have shape [N,{columns}], got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} distribution is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} distribution contains NaN or inf")
    return arr


def _aligned_histograms_2d(pred: np.ndarray, ref: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = []
    for column in range(2):
        lower = float(min(np.min(pred[:, column]), np.min(ref[:, column])))
        upper = float(max(np.max(pred[:, column]), np.max(ref[:, column])))
        if math.isclose(lower, upper):
            upper = lower + 1.0
        edges.append(np.linspace(lower, upper, int(bins) + 1))
    pred_hist, _, _ = np.histogram2d(pred[:, 0], pred[:, 1], bins=edges)
    ref_hist, _, _ = np.histogram2d(ref[:, 0], ref[:, 1], bins=edges)
    pred_prob = pred_hist.astype(np.float64) + 1.0e-12
    ref_prob = ref_hist.astype(np.float64) + 1.0e-12
    pred_prob /= np.sum(pred_prob)
    ref_prob /= np.sum(ref_prob)
    return pred_prob, ref_prob

