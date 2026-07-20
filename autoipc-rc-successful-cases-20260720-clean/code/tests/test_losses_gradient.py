from __future__ import annotations

import numpy as np
import tensorflow as tf

from auto_ipc_rc.losses import PCConstraintSpec, multi_head_pc_loss


def test_multi_head_pc_loss_has_finite_nonzero_gradient_for_pc_heads() -> None:
    rho_local = tf.constant([[-1.0, -0.2, 0.3, 1.0], [0.8, 0.1, -0.4, -0.9]], dtype=tf.float32)
    pc_heads = tf.Variable(
        np.array(
            [
                [[0.7, -0.1, 0.2, -0.6], [-0.5, 0.4, -0.3, 0.9]],
                [[-0.2, 0.6, -0.7, 0.1], [0.3, -0.8, 0.5, -0.4]],
            ],
            dtype=np.float32,
        )
    )
    specs = (
        PCConstraintSpec(name="PCI", alpha=0.4, phi_pi_fraction=0.490),
        PCConstraintSpec(name="PCII", alpha=0.2, phi_pi_fraction=0.455),
    )

    with tf.GradientTape() as tape:
        loss, metrics = multi_head_pc_loss(rho_local, pc_heads, specs, rank_weight=0.1)
    grad = tape.gradient(loss, pc_heads)

    assert loss.shape == ()
    assert np.isfinite(float(loss.numpy()))
    assert np.isfinite(grad.numpy()).all()
    assert float(tf.linalg.global_norm([grad]).numpy()) > 0.0
    assert metrics["alpha_loss_mean"].shape == ()
    assert metrics["phi_loss_mean"].shape == ()
    assert metrics["rank_loss_mean"].shape == ()
    for name in ("PCI", "PCII"):
        assert metrics[f"{name}_alpha_loss"].shape == ()
        assert metrics[f"{name}_phi_loss"].shape == ()
        assert metrics[f"{name}_rank_loss"].shape == ()
        assert metrics[f"{name}_correlation"].shape == ()
        assert metrics[f"{name}_slope"].shape == ()


def test_multi_head_pc_loss_uses_phi_pi_fraction_not_legacy_milliradian_code() -> None:
    spec = PCConstraintSpec(name="PCI", alpha=0.4, phi_pi_fraction=0.490)

    assert spec.phi_radians == np.pi * 0.490


def test_head_weights_are_absolute_multipliers() -> None:
    rho_local = tf.constant([[-1.0, -0.2, 0.3, 1.0], [0.8, 0.1, -0.4, -0.9]], dtype=tf.float32)
    pc_heads = tf.constant(
        np.array(
            [
                [[0.7, -0.1, 0.2, -0.6], [-0.5, 0.4, -0.3, 0.9]],
                [[-0.2, 0.6, -0.7, 0.1], [0.3, -0.8, 0.5, -0.4]],
            ],
            dtype=np.float32,
        )
    )
    specs = (
        PCConstraintSpec(name="PCI", alpha=0.4, phi_pi_fraction=0.490),
        PCConstraintSpec(name="PCII", alpha=0.2, phi_pi_fraction=0.455),
    )

    default_loss, _ = multi_head_pc_loss(rho_local, pc_heads, specs)
    one_one_loss, _ = multi_head_pc_loss(rho_local, pc_heads, specs, head_weights=(1.0, 1.0))
    half_half_loss, _ = multi_head_pc_loss(rho_local, pc_heads, specs, head_weights=(0.5, 0.5))

    np.testing.assert_allclose(default_loss.numpy(), one_one_loss.numpy(), rtol=1e-6)
    np.testing.assert_allclose(half_half_loss.numpy(), 0.5 * one_one_loss.numpy(), rtol=1e-6)
