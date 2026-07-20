from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class PCConstraintSpec:
    name: str
    alpha: float
    phi_pi_fraction: float

    @property
    def phi_radians(self) -> float:
        return math.pi * float(self.phi_pi_fraction)

    @property
    def slope_target(self) -> float:
        return math.tan(self.phi_radians)


def multi_head_pc_loss(
    rho_local: tf.Tensor,
    pc_heads: tf.Tensor,
    specs: Sequence[PCConstraintSpec],
    *,
    head_weights: Sequence[float] | None = None,
    alpha_weight: float = 100.0,
    phi_weight: float = 1.0,
    rank_weight: float = 0.0,
    spearman_weight: float = 1.0,
    epsilon: float = 1.0e-8,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    rho = tf.cast(rho_local, tf.float32)
    pc = tf.cast(pc_heads, tf.float32)
    if pc.shape.rank != 3:
        raise ValueError("pc_heads must have shape [K,B,A]")
    if len(specs) == 0:
        raise ValueError("at least one PC constraint spec is required")

    weights = _head_weights(len(specs), head_weights)
    total_losses = []
    alpha_losses = []
    phi_losses = []
    rank_losses = []
    correlations = []
    spearman_correlations = []
    slopes = []

    per_head_metrics: dict[str, tf.Tensor] = {}
    for index, spec in enumerate(specs):
        pc_i = pc[index]
        corr = _pearson_tf(rho, pc_i, epsilon=epsilon)
        pearson_alpha_loss = tf.square(corr - float(spec.alpha))
        s_corr = _spearman_tf(rho, pc_i, epsilon=epsilon)
        # v8_plus included Spearman in the scalar alpha loss through a
        # numpy/scipy rank op. That term is intentionally non-differentiable;
        # keeping it here restores the monitored objective without pretending it
        # supplies gradient to the network.
        spearman_alpha_loss = tf.stop_gradient(tf.square(s_corr - float(spec.alpha)))
        alpha_loss = pearson_alpha_loss + float(spearman_weight) * spearman_alpha_loss
        slope = _per_system_slope(rho, pc_i, corr, epsilon=epsilon)
        phi_loss = tf.reduce_mean(tf.square(slope - float(spec.slope_target)))
        rank_loss = _pairwise_rank_surrogate(rho, pc_i) if rank_weight else tf.constant(0.0, dtype=tf.float32)
        head_loss = alpha_weight * alpha_loss + phi_weight * phi_loss + rank_weight * rank_loss

        total_losses.append(float(weights[index]) * head_loss)
        alpha_losses.append(alpha_loss)
        phi_losses.append(phi_loss)
        rank_losses.append(rank_loss)
        correlations.append(corr)
        spearman_correlations.append(s_corr)
        slope_mean = tf.reduce_mean(slope)
        slopes.append(slope_mean)

        prefix = _metric_prefix(spec.name)
        per_head_metrics[f"{prefix}_alpha_loss"] = alpha_loss
        per_head_metrics[f"{prefix}_phi_loss"] = phi_loss
        per_head_metrics[f"{prefix}_rank_loss"] = rank_loss
        per_head_metrics[f"{prefix}_correlation"] = corr
        per_head_metrics[f"{prefix}_spearman_correlation"] = s_corr
        per_head_metrics[f"{prefix}_slope"] = slope_mean

    loss = tf.add_n(total_losses)
    metrics = {
        "pc_loss": loss,
        "alpha_loss_mean": tf.reduce_mean(tf.stack(alpha_losses)),
        "phi_loss_mean": tf.reduce_mean(tf.stack(phi_losses)),
        "rank_loss_mean": tf.reduce_mean(tf.stack(rank_losses)),
        "correlation_mean": tf.reduce_mean(tf.stack(correlations)),
        "spearman_correlation_mean": tf.reduce_mean(tf.stack(spearman_correlations)),
        "slope_mean": tf.reduce_mean(tf.stack(slopes)),
    }
    metrics.update(per_head_metrics)
    return loss, metrics


def _metric_prefix(name: str) -> str:
    prefix = "".join(char if char.isalnum() else "_" for char in name.strip())
    prefix = prefix.strip("_")
    if not prefix:
        raise ValueError("PC constraint spec name must contain at least one alphanumeric character")
    return prefix


def _head_weights(count: int, weights: Sequence[float] | None) -> tuple[float, ...]:
    if weights is None:
        return tuple(1.0 for _ in range(count))
    if len(weights) != count:
        raise ValueError("head_weights length must match specs length")
    raw = tuple(float(weight) for weight in weights)
    if any(weight < 0.0 for weight in raw):
        raise ValueError("head_weights must be non-negative")
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("at least one head_weight must be positive")
    return raw


def _pearson_tf(x: tf.Tensor, y: tf.Tensor, *, epsilon: float) -> tf.Tensor:
    x_flat = tf.reshape(tf.cast(x, tf.float32), (-1,))
    y_flat = tf.reshape(tf.cast(y, tf.float32), (-1,))
    x_centered = x_flat - tf.reduce_mean(x_flat)
    y_centered = y_flat - tf.reduce_mean(y_flat)
    denom = tf.sqrt(tf.reduce_sum(tf.square(x_centered)) + epsilon) * tf.sqrt(
        tf.reduce_sum(tf.square(y_centered)) + epsilon
    )
    return tf.reduce_sum(x_centered * y_centered) / denom


def _spearman_tf(x: tf.Tensor, y: tf.Tensor, *, epsilon: float) -> tf.Tensor:
    del epsilon  # Spearman follows the original v8_plus rank-difference formula.
    x_flat = tf.reshape(tf.cast(x, tf.float32), (-1,))
    y_flat = tf.reshape(tf.cast(y, tf.float32), (-1,))
    x_rank = tf.numpy_function(_rankdata_np, [x_flat], tf.float32)
    y_rank = tf.numpy_function(_rankdata_np, [y_flat], tf.float32)
    x_rank.set_shape([None])
    y_rank.set_shape([None])
    diff = x_rank - y_rank
    n = tf.cast(tf.shape(x_flat)[0], tf.float32)
    denom = n * (tf.square(n) - 1.0)
    return 1.0 - (6.0 * tf.reduce_sum(tf.square(diff))) / denom


def _rankdata_np(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.shape[0], dtype=np.float32)
    if flat.shape[0] == 0:
        return ranks
    sorted_values = flat[order]
    start = 0
    while start < flat.shape[0]:
        end = start + 1
        while end < flat.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        # scipy.stats.rankdata average-rank convention, 1-based ranks.
        rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _per_system_slope(rho: tf.Tensor, pc: tf.Tensor, corr: tf.Tensor, *, epsilon: float) -> tf.Tensor:
    rho_centered = rho - tf.reduce_mean(rho, axis=1, keepdims=True)
    pc_centered = pc - tf.reduce_mean(pc, axis=1, keepdims=True)
    rho_norm = tf.sqrt(tf.reduce_sum(tf.square(rho_centered), axis=1) + epsilon)
    pc_norm = tf.sqrt(tf.reduce_sum(tf.square(pc_centered), axis=1) + epsilon)
    return tf.cast(corr, tf.float32) * (pc_norm / rho_norm)


def _pairwise_rank_surrogate(rho: tf.Tensor, pc: tf.Tensor) -> tf.Tensor:
    order = tf.argsort(rho, axis=1, stable=True)
    pc_sorted = tf.gather(pc, order, batch_dims=1)
    adjacent_delta = pc_sorted[:, 1:] - pc_sorted[:, :-1]
    return tf.reduce_mean(tf.nn.softplus(-adjacent_delta))
