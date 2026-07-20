from __future__ import annotations

import numpy as np
import tensorflow as tf

from auto_ipc_rc.models.multi_head_autoencoder import MultiHeadAutoencoder, MultiHeadModelConfig


def test_multi_head_model_uses_approved_embedding_dimensions_and_shapes() -> None:
    tf.keras.utils.set_random_seed(2026)
    cfg = MultiHeadModelConfig(neighbors=30, feature_dim=4, m1=100, m2=100, inner_dim=16, dropout=0.0)
    model = MultiHeadAutoencoder(cfg, num_heads=2)
    coord = tf.constant(np.ones((2, 5, 30, 4), dtype=np.float32))

    outputs = model(coord, training=False)

    assert model.encoder.embedding.WM1.shape == (30, 100)
    assert model.encoder.embedding.WM2.shape == (30, 100)
    assert outputs.rho_global.shape == (2,)
    assert outputs.potential_global.shape == (2,)
    assert outputs.rho_local.shape == (2, 5)
    assert outputs.potential_local.shape == (2, 5)
    assert outputs.pc_heads.shape == (2, 2, 5)


def test_multi_head_model_rejects_non_four_channel_inputs() -> None:
    cfg = MultiHeadModelConfig(neighbors=30, feature_dim=4, m1=100, m2=100, inner_dim=16, dropout=0.0)
    model = MultiHeadAutoencoder(cfg, num_heads=2)
    bad_coord = tf.zeros((1, 5, 30, 5), dtype=tf.float32)

    try:
        model(bad_coord, training=False)
    except ValueError as exc:
        assert "feature dimension" in str(exc)
    else:
        raise AssertionError("expected ValueError for bad feature dimension")
