from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from auto_ipc_rc.data_manifest import DataManifest, FrameRecord
from auto_ipc_rc.dataset import build_record_index, load_frame_arrays_from_index
from auto_ipc_rc.losses import PCConstraintSpec
from auto_ipc_rc.models.multi_head_autoencoder import MultiHeadAutoencoder, MultiHeadModelConfig
from auto_ipc_rc.normalization import TargetNormalizer, fit_target_normalizer, write_normalizer
from auto_ipc_rc.splits import frame_key
from auto_ipc_rc.training import make_train_step, train_one_batch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AutoIPC-RC training entry point.")
    parser.add_argument("--smoke-synthetic", action="store_true", help="Run one synthetic batch; does not read real data")
    parser.add_argument("--manifest-jsonl", help="frames.jsonl produced by build_manifest")
    parser.add_argument("--output-dir", help="Training output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--drop-remainder", action="store_true", help="Drop final incomplete batch to match the original/v8_plus full-batch generator")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for smoke/debug runs")
    parser.add_argument("--sample-frames-per-epoch", type=int, default=None, help="Sample this many unique frames each epoch before optional repeats; matches the paper generator when set to 20000")
    parser.add_argument("--repeat-size", type=int, default=1, help="Repeat sampled/full epoch keys before shuffling and batching")
    parser.add_argument("--shuffle-buffer", type=int, default=None, help="Use bounded buffer shuffle after repeat; set to 400 to match the paper tf.data shuffle(400)")
    parser.add_argument("--normalization-json", default=None, help="Use a precomputed target normalization JSON instead of fitting from the manifest")
    parser.add_argument("--inner-dim", type=int, default=250)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for hidden encoder/decoder layers")
    parser.add_argument("--descriptor-dropout", type=float, default=0.0, help="Dropout rate applied to the flattened descriptor before rho/potential branches")
    parser.add_argument("--m1", type=int, default=100)
    parser.add_argument("--m2", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--head-weights", default=None, help="Comma-separated absolute PC loss multipliers")
    parser.add_argument("--pc-spec", action="append", default=None, help="PC spec NAME:ALPHA:PHI_FRACTION, e.g. PC1:0.2:0.455. Repeat for multiple heads. Defaults to PCI/PCII.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save epoch checkpoint every N epochs; 0 saves only the final epoch of this invocation")
    parser.add_argument("--checkpoint-keep", type=int, default=0, help="Keep only the newest N epoch checkpoints; 0 keeps all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume-weights", default=None, help="Optional weights file to resume from")
    parser.add_argument("--resume-checkpoint", default=None, help="Optional tf.train.Checkpoint prefix to resume model and optimizer state from")
    parser.add_argument("--initial-epoch", type=int, default=0, help="Last completed epoch before this run")
    args = parser.parse_args(argv)

    if args.smoke_synthetic:
        _run_synthetic_smoke(inner_dim=args.inner_dim, m1=args.m1, m2=args.m2, dropout=args.dropout, descriptor_dropout=args.descriptor_dropout)
        return 0

    if not args.manifest_jsonl or not args.output_dir:
        return 2

    specs = _parse_pc_specs(args.pc_spec)

    _run_manifest_training(
        manifest_jsonl=Path(args.manifest_jsonl),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        sample_frames_per_epoch=args.sample_frames_per_epoch,
        repeat_size=args.repeat_size,
        shuffle_buffer=args.shuffle_buffer,
        drop_remainder=args.drop_remainder,
        inner_dim=args.inner_dim,
        dropout=args.dropout,
        descriptor_dropout=args.descriptor_dropout,
        m1=args.m1,
        m2=args.m2,
        learning_rate=args.learning_rate,
        rank_weight=args.rank_weight,
        specs=specs,
        head_weights=_parse_head_weights(args.head_weights, expected_count=len(specs)),
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        seed=args.seed,
        normalization_json=Path(args.normalization_json) if args.normalization_json else None,
        resume_weights=Path(args.resume_weights) if args.resume_weights else None,
        resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        initial_epoch=args.initial_epoch,
    )
    return 0


def _run_synthetic_smoke(*, inner_dim: int, m1: int, m2: int, dropout: float, descriptor_dropout: float) -> None:
    tf.keras.utils.set_random_seed(2026)
    cfg = MultiHeadModelConfig(neighbors=30, feature_dim=4, m1=m1, m2=m2, inner_dim=inner_dim, dropout=dropout, descriptor_dropout=descriptor_dropout)
    model = MultiHeadAutoencoder(cfg, num_heads=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
    decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)
    coord = tf.constant(np.random.default_rng(2026).normal(size=(2, 5, 30, 4)).astype(np.float32))
    rho_target = tf.constant([-0.5, 0.5], dtype=tf.float32)
    potential_target = tf.constant([0.25, -0.25], dtype=tf.float32)
    specs = _default_specs()
    metrics = train_one_batch(model, optimizer, coord, rho_target, potential_target, specs, decoder_optimizer=decoder_optimizer, rank_weight=0.1)
    loss = float(metrics["total_loss"].numpy())
    if not np.isfinite(loss):
        raise RuntimeError("synthetic smoke produced non-finite loss")


def _run_manifest_training(
    *,
    manifest_jsonl: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_steps: int | None,
    sample_frames_per_epoch: int | None,
    repeat_size: int,
    shuffle_buffer: int | None,
    drop_remainder: bool,
    inner_dim: int,
    dropout: float,
    descriptor_dropout: float,
    m1: int,
    m2: int,
    learning_rate: float,
    rank_weight: float,
    specs: tuple[PCConstraintSpec, ...],
    head_weights: tuple[float, ...] | None,
    log_every: int,
    checkpoint_every: int,
    checkpoint_keep: int,
    seed: int,
    normalization_json: Path | None,
    resume_weights: Path | None,
    resume_checkpoint: Path | None,
    initial_epoch: int,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if m1 <= 0 or m2 <= 0:
        raise ValueError("m1 and m2 must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not 0.0 <= descriptor_dropout < 1.0:
        raise ValueError("descriptor_dropout must be in [0, 1)")
    if sample_frames_per_epoch is not None and sample_frames_per_epoch <= 0:
        raise ValueError("sample_frames_per_epoch must be positive when set")
    if repeat_size <= 0:
        raise ValueError("repeat_size must be positive")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if checkpoint_keep < 0:
        raise ValueError("checkpoint_keep must be non-negative")
    if shuffle_buffer is not None and shuffle_buffer <= 0:
        raise ValueError("shuffle_buffer must be positive when set")
    if resume_weights is not None and resume_checkpoint is not None:
        raise ValueError("--resume-weights and --resume-checkpoint are mutually exclusive")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    _configure_gpu()
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    manifest = _load_manifest_jsonl(manifest_jsonl)
    all_keys = tuple(frame_key(frame) for frame in manifest.frames)
    record_index = build_record_index(manifest)
    normalizer = _read_normalizer(normalization_json) if normalization_json is not None else fit_target_normalizer(manifest, all_keys)
    write_normalizer(normalizer, output_dir / "normalization.json")

    cfg = MultiHeadModelConfig(neighbors=30, feature_dim=4, m1=m1, m2=m2, inner_dim=inner_dim, dropout=dropout, descriptor_dropout=descriptor_dropout)
    model = MultiHeadAutoencoder(cfg, num_heads=len(specs))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    _write_resolved_config(
        output_dir / "resolved_config.json",
        manifest_jsonl=manifest_jsonl,
        frame_count=len(all_keys),
        epochs=epochs,
        batch_size=batch_size,
        sample_frames_per_epoch=sample_frames_per_epoch,
        repeat_size=repeat_size,
        drop_remainder=drop_remainder,
        shuffle_buffer=shuffle_buffer,
        normalization_json=str(normalization_json) if normalization_json else None,
        inner_dim=inner_dim,
        dropout=dropout,
        descriptor_dropout=descriptor_dropout,
        learning_rate=learning_rate,
        rank_weight=rank_weight,
        head_weights=head_weights,
        checkpoint_every=checkpoint_every,
        checkpoint_keep=checkpoint_keep,
        seed=seed,
        resume_weights=str(resume_weights) if resume_weights else None,
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        initial_epoch=initial_epoch,
        model_config=cfg,
        specs=specs,
    )

    model(tf.zeros((1, 300, 30, 4), dtype=tf.float32), training=False)
    if resume_weights is not None:
        model.load_weights(resume_weights)
    train_step = make_train_step(
        model,
        optimizer,
        specs,
        decoder_optimizer=decoder_optimizer,
        rank_weight=rank_weight,
        head_weights=head_weights,
    )
    checkpoint_epoch = tf.Variable(initial_epoch, dtype=tf.int64, trainable=False)
    checkpoint_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    training_checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        decoder_optimizer=decoder_optimizer,
        epoch=checkpoint_epoch,
        global_step=checkpoint_step,
    )
    if resume_checkpoint is not None:
        _restore_training_checkpoint(
            training_checkpoint,
            resume_checkpoint,
            optimizer=optimizer,
            decoder_optimizer=decoder_optimizer,
            learning_rate=learning_rate,
        )
        print(
            json.dumps(
                {
                    "event": "resume_checkpoint_restored",
                    "checkpoint": str(resume_checkpoint),
                    "epoch": int(checkpoint_epoch.numpy()),
                    "global_step": int(checkpoint_step.numpy()),
                    "optimizer_iterations": int(optimizer.iterations.numpy()),
                    "decoder_optimizer_iterations": int(decoder_optimizer.iterations.numpy()),
                    "learning_rate": float(optimizer.learning_rate.numpy()),
                    "decoder_learning_rate": float(decoder_optimizer.learning_rate.numpy()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metrics_path = output_dir / "metrics.csv"
    fieldnames = _metrics_fieldnames(specs)
    _ensure_metrics_header(metrics_path, fieldnames)
    write_header = not metrics_path.exists()
    global_step = _last_metric_step(metrics_path)
    checkpoint_lock_file = os.environ.get("TRAIN_EVAL_LOCK_FILE")
    barrier_dir = os.environ.get("TRAIN_EVAL_BARRIER_DIR")
    barrier_epochs = _parse_epoch_set(os.environ.get("TRAIN_EVAL_BARRIER_EPOCHS"))
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for epoch in range(initial_epoch + 1, initial_epoch + epochs + 1):
            keys = _epoch_keys(all_keys, rng, sample_frames_per_epoch=sample_frames_per_epoch, repeat_size=repeat_size, shuffle_buffer=shuffle_buffer)
            if drop_remainder:
                steps_this_epoch = len(keys) // batch_size
                usable_frame_count = steps_this_epoch * batch_size
            else:
                steps_this_epoch = (len(keys) + batch_size - 1) // batch_size
                usable_frame_count = len(keys)
            if steps_this_epoch <= 0:
                raise ValueError("batch_size is larger than the epoch frame count and --drop-remainder was set")
            if max_steps is not None:
                steps_this_epoch = min(steps_this_epoch, max_steps)
                usable_frame_count = min(usable_frame_count, steps_this_epoch * batch_size)
            for step_index in range(steps_this_epoch):
                batch_keys = tuple(keys[step_index * batch_size : (step_index + 1) * batch_size].tolist())
                arrays = load_frame_arrays_from_index(record_index, batch_keys, normalizer=normalizer)
                metrics = train_step(
                    tf.convert_to_tensor(arrays.coord, dtype=tf.float32),
                    tf.convert_to_tensor(arrays.rho, dtype=tf.float32),
                    tf.convert_to_tensor(arrays.potential, dtype=tf.float32),
                )
                global_step += 1
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "seen_frames": min((step_index + 1) * batch_size, usable_frame_count),
                    "total_loss": float(metrics["total_loss"].numpy()),
                    "rho_loss": float(metrics["rho_loss"].numpy()),
                    "potential_loss": float(metrics["potential_loss"].numpy()),
                    "pc_loss": float(metrics["pc_loss"].numpy()),
                }
                for metric_name in fieldnames:
                    if metric_name in row or metric_name in {"epoch", "step", "seen_frames"}:
                        continue
                    if metric_name in metrics:
                        row[metric_name] = float(metrics[metric_name].numpy())
                writer.writerow(row)
                if log_every > 0 and (global_step == 1 or global_step % log_every == 0):
                    print(json.dumps(row, sort_keys=True), flush=True)
            final_epoch = initial_epoch + epochs
            if _should_save_checkpoint(epoch, final_epoch, checkpoint_every):
                # The evaluator can run on CPU while training continues. Lock only
                # the checkpoint transaction so an evaluator never copies a partial
                # TensorFlow/HDF5 checkpoint or races with pruning.
                with _checkpoint_lock(checkpoint_lock_file):
                    checkpoint_epoch.assign(epoch)
                    checkpoint_step.assign(global_step)
                    training_checkpoint.write(str(checkpoints_dir / "tf_latest"))
                    training_checkpoint.write(str(checkpoints_dir / f"tf_epoch_{epoch:04d}"))
                    _save_weights_atomically(model, checkpoints_dir / "latest.weights.h5")
                    _save_weights_atomically(model, checkpoints_dir / f"epoch_{epoch:04d}.weights.h5")
                    _prune_epoch_checkpoints(checkpoints_dir, checkpoint_keep)
                    _prune_tf_checkpoints(checkpoints_dir, checkpoint_keep)
                _wait_for_eval_barrier(epoch, barrier_dir, barrier_epochs)


@contextmanager
def _checkpoint_lock(path: str | None):
    if not path:
        yield
        return
    import fcntl

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parse_epoch_set(value: str | None) -> frozenset[int]:
    if not value or not value.strip():
        return frozenset()
    try:
        epochs = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("TRAIN_EVAL_BARRIER_EPOCHS must be comma-separated integers") from exc
    if any(epoch <= 0 for epoch in epochs):
        raise ValueError("TRAIN_EVAL_BARRIER_EPOCHS must contain positive integers")
    return frozenset(epochs)


def _wait_for_eval_barrier(epoch: int, barrier_dir: str | None, barrier_epochs: frozenset[int]) -> None:
    if not barrier_dir or epoch not in barrier_epochs:
        return
    eval_dir = Path(barrier_dir) / f"epoch_{epoch}"
    done = eval_dir / "evaluation.done"
    failed = eval_dir / "evaluation.failed"
    print(json.dumps({"event": "waiting_for_eval", "epoch": epoch, "marker": str(done)}, sort_keys=True), flush=True)
    while True:
        if done.is_file() and done.stat().st_size >= 0:
            print(json.dumps({"event": "evaluation_released", "epoch": epoch}, sort_keys=True), flush=True)
            return
        if failed.exists():
            raise RuntimeError(f"Fig2d evaluation failed at epoch {epoch}: {failed}")
        time.sleep(5.0)




def _save_weights_atomically(model: tf.keras.Model, path: Path) -> None:
    # Keras requires filenames ending in .weights.h5; write a hidden temp file, then replace.
    tmp_path = path.with_name(f".{path.name}.tmp.weights.h5")
    tmp_path.unlink(missing_ok=True)
    model.save_weights(tmp_path)
    tmp_path.replace(path)


def _restore_training_checkpoint(
    training_checkpoint: tf.train.Checkpoint,
    checkpoint_path: Path,
    *,
    optimizer: tf.keras.optimizers.Optimizer,
    decoder_optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float,
) -> tf.train.CheckpointLoadStatus:
    status = training_checkpoint.restore(str(checkpoint_path))
    status.expect_partial()
    optimizer.learning_rate.assign(learning_rate)
    decoder_optimizer.learning_rate.assign(learning_rate)
    return status


def _should_save_checkpoint(epoch: int, final_epoch: int, checkpoint_every: int) -> bool:
    if epoch == final_epoch:
        return True
    if checkpoint_every <= 0:
        return False
    return epoch % checkpoint_every == 0


def _prune_epoch_checkpoints(checkpoints_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    epoch_files = sorted(checkpoints_dir.glob("epoch_*.weights.h5"))
    for old in epoch_files[:-keep]:
        old.unlink(missing_ok=True)


def _prune_tf_checkpoints(checkpoints_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    epoch_indexes = sorted(checkpoints_dir.glob("tf_epoch_*.index"))
    old_prefixes = [path.with_suffix("") for path in epoch_indexes[:-keep]]
    for prefix in old_prefixes:
        for path in checkpoints_dir.glob(prefix.name + ".*"):
            path.unlink(missing_ok=True)



def _parse_head_weights(value: str | None, *, expected_count: int | None = None) -> tuple[float, ...] | None:
    if value is None or not value.strip():
        return None
    try:
        weights = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("--head-weights must be comma-separated numbers") from exc
    if expected_count is not None and len(weights) != expected_count:
        raise ValueError(f"--head-weights must contain exactly {expected_count} value(s)")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("--head-weights values must be positive")
    return weights


def _parse_pc_specs(values: Sequence[str] | None) -> tuple[PCConstraintSpec, ...]:
    if not values:
        return _default_specs()
    specs: list[PCConstraintSpec] = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError("--pc-spec must use NAME:ALPHA:PHI_FRACTION")
        name = parts[0].strip()
        if not name:
            raise ValueError("--pc-spec name must not be empty")
        try:
            alpha = float(parts[1])
            phi = float(parts[2])
        except ValueError as exc:
            raise ValueError("--pc-spec alpha and phi must be numeric") from exc
        specs.append(PCConstraintSpec(name=name, alpha=alpha, phi_pi_fraction=phi))
    return tuple(specs)


def _epoch_keys(
    all_keys: tuple[str, ...],
    rng: np.random.Generator,
    *,
    sample_frames_per_epoch: int | None,
    repeat_size: int,
    shuffle_buffer: int | None,
) -> np.ndarray:
    keys = np.asarray(all_keys, dtype=object)
    repeats = []
    for _ in range(repeat_size):
        if sample_frames_per_epoch is not None:
            sample_size = min(sample_frames_per_epoch, len(keys))
            repeats.append(rng.choice(keys, size=sample_size, replace=False))
        else:
            repeats.append(keys.copy())
    selected = np.concatenate(repeats)
    if shuffle_buffer is None:
        rng.shuffle(selected)
        return selected
    return _bounded_shuffle(selected, rng, buffer_size=shuffle_buffer)


def _bounded_shuffle(values: np.ndarray, rng: np.random.Generator, *, buffer_size: int) -> np.ndarray:
    if len(values) <= buffer_size:
        shuffled = values.copy()
        rng.shuffle(shuffled)
        return shuffled
    buffer = list(values[:buffer_size])
    out = []
    for value in values[buffer_size:]:
        index = int(rng.integers(0, len(buffer)))
        out.append(buffer[index])
        buffer[index] = value
    rng.shuffle(buffer)
    out.extend(buffer)
    return np.asarray(out, dtype=object)


def _read_normalizer(path: Path) -> TargetNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {name: payload[name] for name in TargetNormalizer.__dataclass_fields__ if name in payload}
    return TargetNormalizer(**allowed)

def _metrics_fieldnames(specs: Sequence[PCConstraintSpec]) -> list[str]:
    fields = [
        "epoch",
        "step",
        "seen_frames",
        "total_loss",
        "rho_loss",
        "potential_loss",
        "pc_loss",
        "alpha_loss_mean",
        "phi_loss_mean",
        "rank_loss_mean",
        "correlation_mean",
        "spearman_correlation_mean",
        "slope_mean",
    ]
    for spec in specs:
        prefix = _metric_prefix(spec.name)
        fields.extend(
            [
                f"{prefix}_alpha_loss",
                f"{prefix}_phi_loss",
                f"{prefix}_rank_loss",
                f"{prefix}_correlation",
                f"{prefix}_spearman_correlation",
                f"{prefix}_slope",
            ]
        )
    return fields


def _metric_prefix(name: str) -> str:
    prefix = "".join(char if char.isalnum() else "_" for char in name.strip()).strip("_")
    if not prefix:
        raise ValueError("PC constraint spec name must contain at least one alphanumeric character")
    return prefix


def _ensure_metrics_header(path: Path, fieldnames: Sequence[str]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing = reader.fieldnames or []
        if list(existing) == list(fieldnames):
            return
        rows = list(reader)
    merged = list(fieldnames)
    for name in existing:
        if name not in merged:
            merged.append(name)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in merged})
    tmp_path.replace(path)


def _last_metric_step(path: Path) -> int:
    if not path.exists():
        return 0
    last = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("step"):
                last = int(row["step"])
    return last


def _load_manifest_jsonl(path: Path) -> DataManifest:
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frames.append(FrameRecord(**json.loads(line)))
    if not frames:
        raise ValueError(f"manifest contains no frames: {path}")
    return DataManifest(frames=frames, rejections=[])


def _default_specs() -> tuple[PCConstraintSpec, PCConstraintSpec]:
    return (
        PCConstraintSpec(name="PC1", alpha=0.2, phi_pi_fraction=0.455),
        PCConstraintSpec(name="PC2", alpha=0.4, phi_pi_fraction=0.490),
    )


def _configure_gpu(*, allow_cpu: bool = False) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if allow_cpu:
            return
        raise RuntimeError("GPU training requested but no TensorFlow GPU is visible")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _write_resolved_config(path: Path, **payload) -> None:
    converted = dict(payload)
    converted["manifest_jsonl"] = str(converted["manifest_jsonl"])
    converted["model_config"] = asdict(converted["model_config"])
    converted["specs"] = [asdict(spec) for spec in converted["specs"]]
    path.write_text(json.dumps(converted, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
