from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class MultiHeadModelConfig:
    neighbors: int = 30
    feature_dim: int = 4
    m1: int = 100
    m2: int = 100
    inner_dim: int = 250
    out_dim: int = 1
    dropout: float = 0.0
    descriptor_dropout: float = 0.0


@dataclass(frozen=True)
class MultiHeadOutputs:
    rho_global: tf.Tensor
    potential_global: tf.Tensor
    rho_local: tf.Tensor
    potential_local: tf.Tensor
    pc_heads: tf.Tensor


class EmbeddingDescriptor(tf.keras.layers.Layer):
    def __init__(self, cfg: MultiHeadModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def build(self, input_shape):
        self.WM1 = self.add_weight(
            shape=(self.cfg.neighbors, self.cfg.m1),
            initializer="random_normal",
            trainable=True,
            name="WM1",
        )
        self.WM2 = self.add_weight(
            shape=(self.cfg.neighbors, self.cfg.m2),
            initializer="random_normal",
            trainable=True,
            name="WM2",
        )
        super().build(input_shape)

    def call(self, inputs):
        _validate_input_shape(inputs, self.cfg)
        r_t = tf.transpose(inputs, perm=[0, 1, 3, 2])
        d_2 = tf.matmul(r_t, self.WM2)
        d_1 = tf.matmul(self.WM1, inputs, transpose_a=True)
        return tf.matmul(d_1, d_2)


class SharedFeatureDropout(tf.keras.layers.Layer):
    def __init__(self, rate: float, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def call(self, inputs, training: bool = False):
        if not training or self.rate <= 0.0:
            return inputs
        feature_count = tf.shape(inputs)[-1]
        noise_shape = tf.concat([tf.ones(tf.rank(inputs) - 1, dtype=tf.int32), [feature_count]], axis=0)
        return tf.nn.dropout(inputs, rate=self.rate, noise_shape=noise_shape)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, cfg: MultiHeadModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.embedding = EmbeddingDescriptor(cfg, name="embedding")
        self.rho_dense1 = tf.keras.layers.Dense(cfg.inner_dim)
        self.rho_dense2 = tf.keras.layers.Dense(cfg.inner_dim)
        self.rho_dense3 = tf.keras.layers.Dense(cfg.out_dim)
        self.pot_dense1 = tf.keras.layers.Dense(cfg.inner_dim)
        self.pot_dense2 = tf.keras.layers.Dense(cfg.inner_dim)
        self.pot_dense3 = tf.keras.layers.Dense(cfg.out_dim)
        self.descriptor_dropout = tf.keras.layers.Dropout(cfg.descriptor_dropout)
        self.rho_dropout1 = tf.keras.layers.Dropout(cfg.dropout)
        self.rho_dropout2 = tf.keras.layers.Dropout(cfg.dropout)
        self.pot_dropout1 = tf.keras.layers.Dropout(cfg.dropout)
        self.pot_dropout2 = tf.keras.layers.Dropout(cfg.dropout)
        self.activation = tf.keras.layers.Activation("gelu")

    def call(self, inputs, training: bool = False):
        _validate_input_shape(inputs, self.cfg)
        batch_size = tf.shape(inputs)[0]
        atom_count = tf.shape(inputs)[1]

        descriptor = self.embedding(inputs)
        descriptor = tf.reshape(descriptor, (batch_size * atom_count, self.cfg.m1 * self.cfg.m2))
        descriptor = self.descriptor_dropout(descriptor, training=training)
        x1 = self.rho_dense1(descriptor)
        x = self.activation(self.rho_dropout1(x1, training=training))
        x2 = self.rho_dense2(x)
        x = self.activation(self.rho_dropout2(x2, training=training))
        x3 = self.rho_dense3(x)
        rho_local = tf.reshape(x3, (batch_size, atom_count))
        rho_global = tf.reduce_mean(rho_local, axis=-1)

        y1 = self.pot_dense1(descriptor)
        y = self.activation(self.pot_dropout1(y1, training=training))
        y2 = self.pot_dense2(y)
        y = self.activation(self.pot_dropout2(y2, training=training))
        y3 = self.pot_dense3(y)
        potential_local = tf.reshape(y3, (batch_size, atom_count))
        potential_global = tf.reduce_sum(potential_local, axis=-1)

        return rho_global, rho_local, x3, x2, x1, potential_global, potential_local, y3, y2, y1


class DecoderHead(tf.keras.layers.Layer):
    def __init__(self, cfg: MultiHeadModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        reg = tf.keras.regularizers.L2(0.01)
        self.rho_dense1 = tf.keras.layers.Dense(cfg.inner_dim, kernel_regularizer=reg, bias_regularizer=reg)
        self.rho_dense2 = tf.keras.layers.Dense(cfg.inner_dim, kernel_regularizer=reg, bias_regularizer=reg)
        self.pot_dense1 = tf.keras.layers.Dense(cfg.inner_dim, kernel_regularizer=reg, bias_regularizer=reg)
        self.pot_dense2 = tf.keras.layers.Dense(cfg.inner_dim, kernel_regularizer=reg, bias_regularizer=reg)
        self.fuse_dense1 = tf.keras.layers.Dense(cfg.inner_dim * 2, kernel_regularizer=reg, bias_regularizer=reg)
        self.fuse_dense2 = tf.keras.layers.Dense(cfg.inner_dim * 2, kernel_regularizer=reg, bias_regularizer=reg)
        self.out_dense = tf.keras.layers.Dense(cfg.out_dim, kernel_regularizer=reg, bias_regularizer=reg)
        self.rho_dropout1 = tf.keras.layers.Dropout(cfg.dropout)
        self.rho_dropout2 = tf.keras.layers.Dropout(cfg.dropout)
        self.pot_dropout1 = tf.keras.layers.Dropout(cfg.dropout)
        self.pot_dropout2 = tf.keras.layers.Dropout(cfg.dropout)
        self.fuse_dropout1 = tf.keras.layers.Dropout(cfg.dropout)
        self.fuse_dropout2 = tf.keras.layers.Dropout(cfg.dropout)
        self.activation = tf.keras.layers.Activation("gelu")

    def call(self, x3, x2, x1, y3, y2, y1, batch_size, atom_count, training: bool = False):
        x = self.rho_dense1(x3) + x1
        x = self.activation(self.rho_dropout1(x, training=training))
        x = self.rho_dense2(x) + x2
        x = self.activation(self.rho_dropout2(x, training=training))

        y = self.pot_dense1(y3) + y1
        y = self.activation(self.pot_dropout1(y, training=training))
        y = self.pot_dense2(y) + y2
        y = self.activation(self.pot_dropout2(y, training=training))

        fused = tf.concat([x, y], axis=-1)
        fused = self.activation(self.fuse_dropout1(self.fuse_dense1(fused), training=training))
        fused = self.activation(self.fuse_dropout2(self.fuse_dense2(fused), training=training))
        pc_flat = self.out_dense(fused)
        return tf.reshape(pc_flat, (batch_size, atom_count))


class MultiHeadAutoencoder(tf.keras.Model):
    def __init__(self, cfg: MultiHeadModelConfig, num_heads: int):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        self.cfg = cfg
        self.encoder = Encoder(cfg, name="encoder")
        self.pc_decoders = [DecoderHead(cfg, name=f"decoder_pc{i + 1}") for i in range(num_heads)]

    def call(self, inputs, training: bool = False) -> MultiHeadOutputs:
        _validate_input_shape(inputs, self.cfg)
        batch_size = tf.shape(inputs)[0]
        atom_count = tf.shape(inputs)[1]
        rho_global, rho_local, x3, x2, x1, potential_global, potential_local, y3, y2, y1 = self.encoder(
            inputs,
            training=training,
        )
        pc_heads = [
            decoder(x3, x2, x1, y3, y2, y1, batch_size, atom_count, training=training)
            for decoder in self.pc_decoders
        ]
        return MultiHeadOutputs(
            rho_global=rho_global,
            potential_global=potential_global,
            rho_local=rho_local,
            potential_local=potential_local,
            pc_heads=tf.stack(pc_heads, axis=0),
        )


def _validate_input_shape(inputs, cfg: MultiHeadModelConfig) -> None:
    shape = inputs.shape
    if shape.rank != 4:
        raise ValueError(f"coord input must have rank 4 [B,A,N,D], got rank {shape.rank}")
    if shape[-2] is not None and int(shape[-2]) != cfg.neighbors:
        raise ValueError(f"neighbor dimension must be {cfg.neighbors}, got {shape[-2]}")
    if shape[-1] is not None and int(shape[-1]) != cfg.feature_dim:
        raise ValueError(f"feature dimension must be {cfg.feature_dim}, got {shape[-1]}")
