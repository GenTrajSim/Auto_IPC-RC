from __future__ import annotations

import numpy as np
import tensorflow as tf

from auto_ipc_rc.losses import PCConstraintSpec
from auto_ipc_rc.models.multi_head_autoencoder import MultiHeadAutoencoder, MultiHeadModelConfig
from auto_ipc_rc.training import train_one_batch


def test_train_one_batch_returns_finite_loss_and_updates_weights() -> None:
    tf.keras.utils.set_random_seed(2026)
    cfg = MultiHeadModelConfig(neighbors=30, feature_dim=4, m1=100, m2=100, inner_dim=8, dropout=0.0)
    model = MultiHeadAutoencoder(cfg, num_heads=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
    coord = tf.constant(np.random.default_rng(2026).normal(size=(2, 5, 30, 4)).astype(np.float32))
    rho_target = tf.constant([-0.5, 0.5], dtype=tf.float32)
    potential_target = tf.constant([0.25, -0.25], dtype=tf.float32)
    specs = (
        PCConstraintSpec(name="PCI", alpha=0.4, phi_pi_fraction=0.490),
        PCConstraintSpec(name="PCII", alpha=0.2, phi_pi_fraction=0.455),
    )
    _ = model(coord, training=False)
    before = [weight.numpy().copy() for weight in model.trainable_variables]

    metrics = train_one_batch(model, optimizer, coord, rho_target, potential_target, specs, rank_weight=0.1)

    after = [weight.numpy() for weight in model.trainable_variables]
    assert np.isfinite(float(metrics["total_loss"].numpy()))
    assert np.isfinite(float(metrics["rho_loss"].numpy()))
    assert np.isfinite(float(metrics["potential_loss"].numpy()))
    assert any(np.any(a != b) for a, b in zip(after, before))


def test_train_one_batch_does_not_backprop_pc_loss_into_encoder() -> None:
    tf.keras.utils.set_random_seed(2027)
    cfg = MultiHeadModelConfig(neighbors=30, feature_dim=4, m1=100, m2=100, inner_dim=8, dropout=0.0)
    model = MultiHeadAutoencoder(cfg, num_heads=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
    coord = tf.constant(np.random.default_rng(2027).normal(size=(2, 5, 30, 4)).astype(np.float32))
    outputs = model(coord, training=False)
    specs = (
        PCConstraintSpec(name="PCI", alpha=0.4, phi_pi_fraction=0.490),
        PCConstraintSpec(name="PCII", alpha=0.2, phi_pi_fraction=0.455),
    )
    encoder_before = [weight.numpy().copy() for weight in model.encoder.trainable_variables]
    decoder_before = [weight.numpy().copy() for decoder in model.pc_decoders for weight in decoder.trainable_variables]

    train_one_batch(
        model,
        optimizer,
        coord,
        outputs.rho_global,
        outputs.potential_global,
        specs,
        rho_weight=0.0,
        potential_weight=0.0,
        rank_weight=0.0,
    )

    encoder_after = [weight.numpy() for weight in model.encoder.trainable_variables]
    decoder_after = [weight.numpy() for decoder in model.pc_decoders for weight in decoder.trainable_variables]
    assert all(np.array_equal(a, b) for a, b in zip(encoder_after, encoder_before))
    assert any(np.any(a != b) for a, b in zip(decoder_after, decoder_before))
