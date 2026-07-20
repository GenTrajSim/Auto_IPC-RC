from __future__ import annotations

from auto_ipc_rc.cli.train import main


def test_train_cli_runs_synthetic_smoke_without_real_training() -> None:
    exit_code = main(["--smoke-synthetic", "--inner-dim", "8"])

    assert exit_code == 0


def test_train_cli_refuses_non_smoke_training_until_explicitly_implemented() -> None:
    exit_code = main([])

    assert exit_code == 2

import csv
import json
from pathlib import Path

import numpy as np
import pytest


def test_train_cli_runs_one_real_data_step_from_manifest(tmp_path: Path) -> None:
    from auto_ipc_rc.cli.train import main

    root = tmp_path / "Dataset" / "P1700_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    frames_path = tmp_path / "frames.jsonl"
    rows = []
    for frame_id in ["1", "2"]:
        coord_path = root / "new_coord" / f"{frame_id}.npy"
        box_path = root / "box" / f"{frame_id}.npy"
        np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
        np.save(box_path, np.array([1, 2, 3, 4, 5, float(frame_id), -float(frame_id)], dtype=np.float64))
        rows.append(
            {
                "dataset": "Dataset",
                "condition": "P1700_T188",
                "data_dir": "data1",
                "frame_id": frame_id,
                "coord_path": str(coord_path),
                "box_path": str(box_path),
                "coord_sha256": frame_id.zfill(64),
                "box_sha256": str(int(frame_id) + 1).zfill(64),
                "rho": float(frame_id),
                "potential": -float(frame_id),
            }
        )
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    exit_code = main(
        [
            "--manifest-jsonl",
            str(frames_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "1",
            "--inner-dim",
            "8",
            "--m1",
            "100",
            "--m2",
            "60",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "out" / "checkpoints" / "latest.weights.h5").exists()
    metrics_path = tmp_path / "out" / "metrics.csv"
    assert metrics_path.exists()
    row = next(csv.DictReader(metrics_path.open()))
    assert "PC1_alpha_loss" in row
    assert "PC1_phi_loss" in row
    assert "PC2_alpha_loss" in row
    assert "PC2_phi_loss" in row
    config = json.loads((tmp_path / "out" / "resolved_config.json").read_text())
    assert config["model_config"]["m1"] == 100
    assert config["model_config"]["m2"] == 60


def test_train_cli_can_resume_from_weights(tmp_path: Path) -> None:
    from auto_ipc_rc.cli.train import main

    root = tmp_path / "Dataset" / "P1700_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    frames_path = tmp_path / "frames.jsonl"
    rows = []
    for frame_id in ["1", "2"]:
        coord_path = root / "new_coord" / f"{frame_id}.npy"
        box_path = root / "box" / f"{frame_id}.npy"
        np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
        np.save(box_path, np.array([1, 2, 3, 4, 5, float(frame_id), -float(frame_id)], dtype=np.float64))
        rows.append(
            {
                "dataset": "Dataset",
                "condition": "P1700_T188",
                "data_dir": "data1",
                "frame_id": frame_id,
                "coord_path": str(coord_path),
                "box_path": str(box_path),
                "coord_sha256": frame_id.zfill(64),
                "box_sha256": str(int(frame_id) + 1).zfill(64),
                "rho": float(frame_id),
                "potential": -float(frame_id),
            }
        )
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "out"
    common = ["--manifest-jsonl", str(frames_path), "--output-dir", str(out), "--epochs", "1", "--batch-size", "1", "--max-steps", "1", "--inner-dim", "8"]

    assert main(common) == 0
    assert main([*common, "--resume-weights", str(out / "checkpoints" / "latest.weights.h5"), "--initial-epoch", "1"]) == 0

    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert rows[-1]["epoch"] == "2"

def test_train_cli_upgrades_existing_metrics_header(tmp_path: Path) -> None:
    from auto_ipc_rc.cli.train import _ensure_metrics_header, _metrics_fieldnames, _default_specs

    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "epoch,step,seen_frames,total_loss,rho_loss,potential_loss,pc_loss\n"
        "1,1,1,4.0,0.1,0.2,3.7\n",
        encoding="utf-8",
    )

    fieldnames = _metrics_fieldnames(_default_specs())
    _ensure_metrics_header(metrics_path, fieldnames)

    rows = list(csv.DictReader(metrics_path.open()))
    assert rows[0]["pc_loss"] == "3.7"
    assert "PC1_alpha_loss" in rows[0]
    assert rows[0]["PC1_alpha_loss"] == ""
    assert "PC2_phi_loss" in rows[0]



def test_parse_head_weights() -> None:
    from auto_ipc_rc.cli.train import _parse_head_weights

    assert _parse_head_weights(None) is None
    assert _parse_head_weights("1,2") == (1.0, 2.0)


def test_default_specs_use_explicit_pc1_pc2_mapping() -> None:
    from auto_ipc_rc.cli.train import _default_specs

    specs = _default_specs()

    assert [(spec.name, spec.alpha, spec.phi_pi_fraction) for spec in specs] == [
        ("PC1", 0.2, 0.455),
        ("PC2", 0.4, 0.490),
    ]


def test_train_cli_records_head_weights_in_config(tmp_path: Path) -> None:
    from auto_ipc_rc.cli.train import main

    root = tmp_path / "Dataset" / "P1700_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    frames_path = tmp_path / "frames.jsonl"
    rows = []
    for frame_id in ["1", "2"]:
        coord_path = root / "new_coord" / f"{frame_id}.npy"
        box_path = root / "box" / f"{frame_id}.npy"
        np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
        np.save(box_path, np.array([1, 2, 3, 4, 5, float(frame_id), -float(frame_id)], dtype=np.float64))
        rows.append(
            {
                "dataset": "Dataset",
                "condition": "P1700_T188",
                "data_dir": "data1",
                "frame_id": frame_id,
                "coord_path": str(coord_path),
                "box_path": str(box_path),
                "coord_sha256": frame_id.zfill(64),
                "box_sha256": str(int(frame_id) + 1).zfill(64),
                "rho": float(frame_id),
                "potential": -float(frame_id),
            }
        )
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    exit_code = main(
        [
            "--manifest-jsonl",
            str(frames_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "1",
            "--inner-dim",
            "8",
            "--head-weights",
            "1,2",
        ]
    )

    assert exit_code == 0
    config = json.loads((tmp_path / "out" / "resolved_config.json").read_text())
    assert config["head_weights"] == [1.0, 2.0]


def test_train_cli_drop_remainder_skips_partial_batch(tmp_path: Path) -> None:
    from auto_ipc_rc.cli.train import main

    root = tmp_path / "Dataset" / "P1700_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    frames_path = tmp_path / "frames.jsonl"
    rows = []
    for frame_id in ["1", "2", "3"]:
        coord_path = root / "new_coord" / f"{frame_id}.npy"
        box_path = root / "box" / f"{frame_id}.npy"
        np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
        np.save(box_path, np.array([1, 2, 3, 4, 5, float(frame_id), -float(frame_id)], dtype=np.float64))
        rows.append(
            {
                "dataset": "Dataset",
                "condition": "P1700_T188",
                "data_dir": "data1",
                "frame_id": frame_id,
                "coord_path": str(coord_path),
                "box_path": str(box_path),
                "coord_sha256": frame_id.zfill(64),
                "box_sha256": str(int(frame_id) + 1).zfill(64),
                "rho": float(frame_id),
                "potential": -float(frame_id),
            }
        )
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert main([
        "--manifest-jsonl", str(frames_path),
        "--output-dir", str(tmp_path / "out"),
        "--epochs", "1",
        "--batch-size", "2",
        "--drop-remainder",
        "--inner-dim", "8",
    ]) == 0

    rows = list(csv.DictReader((tmp_path / "out" / "metrics.csv").open()))
    assert len(rows) == 1
    assert rows[0]["seen_frames"] == "2"


def test_train_cli_accepts_single_pc_spec(tmp_path: Path) -> None:
    from auto_ipc_rc.cli.train import main

    root = tmp_path / "Dataset" / "P1700_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    frames_path = tmp_path / "frames.jsonl"
    rows = []
    for frame_id in ["1", "2"]:
        coord_path = root / "new_coord" / f"{frame_id}.npy"
        box_path = root / "box" / f"{frame_id}.npy"
        np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
        np.save(box_path, np.array([1, 2, 3, 4, 5, float(frame_id), -float(frame_id)], dtype=np.float64))
        rows.append(
            {
                "dataset": "Dataset",
                "condition": "P1700_T188",
                "data_dir": "data1",
                "frame_id": frame_id,
                "coord_path": str(coord_path),
                "box_path": str(box_path),
                "coord_sha256": frame_id.zfill(64),
                "box_sha256": str(int(frame_id) + 1).zfill(64),
                "rho": float(frame_id),
                "potential": -float(frame_id),
            }
        )
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert main([
        "--manifest-jsonl", str(frames_path),
        "--output-dir", str(tmp_path / "out"),
        "--epochs", "1",
        "--batch-size", "1",
        "--max-steps", "1",
        "--inner-dim", "8",
        "--pc-spec", "PC1:0.2:0.455",
        "--head-weights", "1",
    ]) == 0

    config = json.loads((tmp_path / "out" / "resolved_config.json").read_text())
    assert config["specs"] == [{"alpha": 0.2, "name": "PC1", "phi_pi_fraction": 0.455}]
    row = next(csv.DictReader((tmp_path / "out" / "metrics.csv").open()))
    assert "PC1_alpha_loss" in row
    assert "PC2_alpha_loss" not in row


def test_train_cli_can_use_external_normalization_json(tmp_path: Path) -> None:
    from auto_ipc_rc.cli.train import main
    from auto_ipc_rc.normalization import TargetNormalizer, write_normalizer

    root = tmp_path / "Dataset" / "P1700_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    frames_path = tmp_path / "frames.jsonl"
    rows = []
    for frame_id, rho, potential in [("1", 10.0, -100.0), ("2", 20.0, -200.0)]:
        coord_path = root / "new_coord" / f"{frame_id}.npy"
        box_path = root / "box" / f"{frame_id}.npy"
        np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
        np.save(box_path, np.array([1, 2, 3, 4, 5, rho, potential], dtype=np.float64))
        rows.append(
            {
                "dataset": "Dataset",
                "condition": "P1700_T188",
                "data_dir": "data1",
                "frame_id": frame_id,
                "coord_path": str(coord_path),
                "box_path": str(box_path),
                "coord_sha256": frame_id.zfill(64),
                "box_sha256": str(int(frame_id) + 1).zfill(64),
                "rho": rho,
                "potential": potential,
            }
        )
    frames_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    normalization_path = tmp_path / "paper_norm.json"
    write_normalizer(TargetNormalizer(1.0, 3.0, -30.0, -10.0, 47), normalization_path)
    out = tmp_path / "out"

    exit_code = main(
        [
            "--manifest-jsonl",
            str(frames_path),
            "--output-dir",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "1",
            "--inner-dim",
            "8",
            "--normalization-json",
            str(normalization_path),
            "--shuffle-buffer",
            "1",
        ]
    )

    assert exit_code == 0
    written = json.loads((out / "normalization.json").read_text())
    assert written["rho_min"] == 1.0
    assert written["rho_max"] == 3.0
    assert written["fit_frame_count"] == 47
    config = json.loads((out / "resolved_config.json").read_text())
    assert config["normalization_json"] == str(normalization_path)
    assert config["shuffle_buffer"] == 1


def test_epoch_keys_draws_each_repeat_independently() -> None:
    from auto_ipc_rc.cli.train import _epoch_keys

    class RecordingRng:
        def __init__(self) -> None:
            self.choice_calls = 0

        def choice(self, values, *, size: int, replace: bool):
            assert replace is False
            start = self.choice_calls * size
            self.choice_calls += 1
            return np.asarray(values[start : start + size], dtype=object)

        def shuffle(self, values) -> None:
            return None

    rng = RecordingRng()
    keys = _epoch_keys(
        tuple(str(index) for index in range(8)),
        rng,
        sample_frames_per_epoch=4,
        repeat_size=2,
        shuffle_buffer=None,
    )

    assert rng.choice_calls == 2
    assert keys.tolist() == [str(index) for index in range(8)]


def test_restore_checkpoint_preserves_optimizer_state_and_overrides_learning_rate(tmp_path: Path) -> None:
    import tensorflow as tf

    from auto_ipc_rc.cli.train import _restore_training_checkpoint

    source_encoder = tf.Variable(1.0)
    source_decoder = tf.Variable(2.0)
    source_optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
    source_decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
    source_optimizer.build([source_encoder])
    source_decoder_optimizer.build([source_decoder])
    source_optimizer.apply_gradients([(tf.constant(0.5), source_encoder)])
    source_decoder_optimizer.apply_gradients([(tf.constant(0.25), source_decoder)])
    source_checkpoint = tf.train.Checkpoint(
        encoder=source_encoder,
        decoder=source_decoder,
        optimizer=source_optimizer,
        decoder_optimizer=source_decoder_optimizer,
    )
    checkpoint_prefix = source_checkpoint.write(str(tmp_path / "source"))

    restored_encoder = tf.Variable(0.0)
    restored_decoder = tf.Variable(0.0)
    restored_optimizer = tf.keras.optimizers.Adam(learning_rate=8.0e-5)
    restored_decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=8.0e-5)
    restored_optimizer.build([restored_encoder])
    restored_decoder_optimizer.build([restored_decoder])
    restored_checkpoint = tf.train.Checkpoint(
        encoder=restored_encoder,
        decoder=restored_decoder,
        optimizer=restored_optimizer,
        decoder_optimizer=restored_decoder_optimizer,
    )

    _restore_training_checkpoint(
        restored_checkpoint,
        Path(checkpoint_prefix),
        optimizer=restored_optimizer,
        decoder_optimizer=restored_decoder_optimizer,
        learning_rate=8.0e-5,
    )

    assert int(restored_optimizer.iterations.numpy()) == 1
    assert int(restored_decoder_optimizer.iterations.numpy()) == 1
    assert float(restored_optimizer.learning_rate.numpy()) == pytest.approx(8.0e-5)
    assert float(restored_decoder_optimizer.learning_rate.numpy()) == pytest.approx(8.0e-5)
    assert float(restored_encoder.numpy()) == pytest.approx(float(source_encoder.numpy()))
    assert float(restored_decoder.numpy()) == pytest.approx(float(source_decoder.numpy()))
