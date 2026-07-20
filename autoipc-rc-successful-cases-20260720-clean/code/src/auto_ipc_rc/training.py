from __future__ import annotations

from collections.abc import Callable, Sequence

import tensorflow as tf

from auto_ipc_rc.losses import PCConstraintSpec, multi_head_pc_loss
from auto_ipc_rc.models.multi_head_autoencoder import MultiHeadAutoencoder


def train_one_batch(
    model: MultiHeadAutoencoder,
    optimizer: tf.keras.optimizers.Optimizer,
    coord: tf.Tensor,
    rho_target: tf.Tensor,
    potential_target: tf.Tensor,
    specs: Sequence[PCConstraintSpec],
    *,
    decoder_optimizer: tf.keras.optimizers.Optimizer | None = None,
    rho_weight: float = 1.0,
    potential_weight: float = 1.0,
    alpha_weight: float = 100.0,
    phi_weight: float = 1.0,
    rank_weight: float = 0.0,
    head_weights: Sequence[float] | None = None,
) -> dict[str, tf.Tensor]:
    encoder_optimizer = optimizer
    decoder_optimizer = decoder_optimizer or optimizer
    _ensure_model_built(model)
    _ensure_dual_optimizers_built(model, encoder_optimizer, decoder_optimizer)
    return _train_one_batch_impl(
        model,
        encoder_optimizer,
        decoder_optimizer,
        coord,
        rho_target,
        potential_target,
        specs,
        rho_weight=rho_weight,
        potential_weight=potential_weight,
        alpha_weight=alpha_weight,
        phi_weight=phi_weight,
        rank_weight=rank_weight,
        head_weights=head_weights,
    )


def make_train_step(
    model: MultiHeadAutoencoder,
    optimizer: tf.keras.optimizers.Optimizer,
    specs: Sequence[PCConstraintSpec],
    *,
    decoder_optimizer: tf.keras.optimizers.Optimizer | None = None,
    rho_weight: float = 1.0,
    potential_weight: float = 1.0,
    alpha_weight: float = 100.0,
    phi_weight: float = 1.0,
    rank_weight: float = 0.0,
    head_weights: Sequence[float] | None = None,
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], dict[str, tf.Tensor]]:
    """Create one compiled training step for the manifest loop.

    The earlier real-data loop called the eager ``train_one_batch`` thousands of
    times and rebuilt optimizer bookkeeping checks on every step. Keeping a
    single traced graph mirrors the v8_plus training path and prevents Python/TF
    objects from accumulating across epochs.
    """
    encoder_optimizer = optimizer
    decoder_optimizer = decoder_optimizer or optimizer
    _ensure_model_built(model)
    _ensure_dual_optimizers_built(model, encoder_optimizer, decoder_optimizer)
    coord_signature = tf.TensorSpec(
        shape=(None, None, model.cfg.neighbors, model.cfg.feature_dim),
        dtype=tf.float32,
    )

    @tf.function(
        reduce_retracing=True,
        input_signature=[
            coord_signature,
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ],
    )
    def _step(coord: tf.Tensor, rho_target: tf.Tensor, potential_target: tf.Tensor) -> dict[str, tf.Tensor]:
        return _train_one_batch_impl(
            model,
            encoder_optimizer,
            decoder_optimizer,
            coord,
            rho_target,
            potential_target,
            specs,
            rho_weight=rho_weight,
            potential_weight=potential_weight,
            alpha_weight=alpha_weight,
            phi_weight=phi_weight,
            rank_weight=rank_weight,
            head_weights=head_weights,
        )

    return _step


def _train_one_batch_impl(
    model: MultiHeadAutoencoder,
    encoder_optimizer: tf.keras.optimizers.Optimizer,
    decoder_optimizer: tf.keras.optimizers.Optimizer,
    coord: tf.Tensor,
    rho_target: tf.Tensor,
    potential_target: tf.Tensor,
    specs: Sequence[PCConstraintSpec],
    *,
    rho_weight: float,
    potential_weight: float,
    alpha_weight: float,
    phi_weight: float,
    rank_weight: float,
    head_weights: Sequence[float] | None,
) -> dict[str, tf.Tensor]:
    with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
        outputs = model(coord, training=True)
        rho_loss = tf.reduce_mean(tf.square(outputs.rho_global - tf.cast(rho_target, tf.float32)))
        potential_loss = tf.reduce_mean(tf.square(outputs.potential_global - tf.cast(potential_target, tf.float32)))
        encoder_loss = float(rho_weight) * rho_loss + float(potential_weight) * potential_loss
        pc_loss, pc_metrics = multi_head_pc_loss(
            outputs.rho_local,
            outputs.pc_heads,
            specs,
            alpha_weight=alpha_weight,
            phi_weight=phi_weight,
            rank_weight=rank_weight,
            head_weights=head_weights,
        )

    encoder_vars = model.encoder.trainable_variables
    decoder_vars = _decoder_trainable_variables(model)
    encoder_gradients = encoder_tape.gradient(encoder_loss, encoder_vars)
    decoder_gradients = decoder_tape.gradient(pc_loss, decoder_vars)

    encoder_grads_and_vars = [(grad, var) for grad, var in zip(encoder_gradients, encoder_vars) if grad is not None]
    decoder_grads_and_vars = [(grad, var) for grad, var in zip(decoder_gradients, decoder_vars) if grad is not None]
    if encoder_grads_and_vars:
        encoder_optimizer.apply_gradients(encoder_grads_and_vars)
    if decoder_grads_and_vars:
        decoder_optimizer.apply_gradients(decoder_grads_and_vars)

    total_loss = encoder_loss + pc_loss
    metrics = {
        "total_loss": total_loss,
        "encoder_loss": encoder_loss,
        "rho_loss": rho_loss,
        "potential_loss": potential_loss,
        "pc_loss": pc_loss,
    }
    metrics.update(pc_metrics)
    return metrics


def _ensure_model_built(model: MultiHeadAutoencoder) -> None:
    if model.trainable_variables:
        return
    dummy = tf.zeros((1, 1, model.cfg.neighbors, model.cfg.feature_dim), dtype=tf.float32)
    model(dummy, training=False)


def _decoder_trainable_variables(model: MultiHeadAutoencoder):
    return [var for decoder in model.pc_decoders for var in decoder.trainable_variables]


def _ensure_dual_optimizers_built(
    model: MultiHeadAutoencoder,
    encoder_optimizer: tf.keras.optimizers.Optimizer,
    decoder_optimizer: tf.keras.optimizers.Optimizer,
) -> None:
    encoder_vars = model.encoder.trainable_variables
    decoder_vars = _decoder_trainable_variables(model)
    if decoder_optimizer is encoder_optimizer:
        _ensure_optimizer_built(encoder_optimizer, model.trainable_variables)
    else:
        _ensure_optimizer_built(encoder_optimizer, encoder_vars)
        _ensure_optimizer_built(decoder_optimizer, decoder_vars)


def _ensure_optimizer_built(optimizer: tf.keras.optimizers.Optimizer, variables) -> None:
    if hasattr(optimizer, "build"):
        optimizer.build(variables)
