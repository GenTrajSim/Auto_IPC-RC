from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auto_ipc_rc.cli.predict_distribution import main
from auto_ipc_rc.models.multi_head_autoencoder import MultiHeadAutoencoder, MultiHeadModelConfig
from auto_ipc_rc.normalization import TargetNormalizer, write_normalizer


def test_predict_distribution_cli_writes_one_file_per_head(tmp_path: Path) -> None:
    root = tmp_path / "Dataset" / "P1800_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    coord_path = root / "new_coord" / "1.npy"
    box_path = root / "box" / "1.npy"
    np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
    np.save(box_path, np.array([1, 2, 3, 4, 5, 1000.0, -17000.0], dtype=np.float64))
    frames_path = tmp_path / "frames.jsonl"
    frame = {
        "dataset": "Dataset",
        "condition": "P1800_T188",
        "data_dir": "data1",
        "frame_id": "1",
        "coord_path": str(coord_path),
        "box_path": str(box_path),
        "coord_sha256": "1".zfill(64),
        "box_sha256": "2".zfill(64),
        "rho": 1000.0,
        "potential": -17000.0,
    }
    frames_path.write_text(json.dumps(frame) + "\n", encoding="utf-8")
    normalizer = TargetNormalizer(900.0, 1200.0, -19000.0, -14000.0, 1)
    normalizer_path = tmp_path / "normalization.json"
    write_normalizer(normalizer, normalizer_path)
    weights_path = tmp_path / "weights.h5"
    cfg = MultiHeadModelConfig(inner_dim=8, m1=100, m2=60)
    model = MultiHeadAutoencoder(cfg, num_heads=2)
    model(np.zeros((1, 300, 30, 4), dtype=np.float32), training=False)
    model.save_weights(weights_path)
    out = tmp_path / "pred"

    assert main([
        "--manifest-jsonl", str(frames_path),
        "--weights", str(weights_path),
        "--normalization", str(normalizer_path),
        "--output-dir", str(out),
        "--condition", "P1800_T188",
        "--batch-size", "1",
        "--inner-dim", "8",
        "--m1", "100",
        "--m2", "60",
    ]) == 0

    for name in ["PCI", "PCII"]:
        rows = np.loadtxt(out / f"{name}_predicted_rows.txt")
        assert rows.shape == (300, 3)
        assert np.isfinite(rows).all()
    summary = json.loads((out / "prediction_summary.json").read_text())
    assert summary["frames"] == 1
    assert summary["rows_per_head"] == 300
    assert summary["model_config"]["m1"] == 100
    assert summary["model_config"]["m2"] == 60


def test_predict_distribution_cli_supports_single_head_mc_dropout(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "Dataset" / "P1800_T188" / "data1"
    (root / "new_coord").mkdir(parents=True)
    (root / "box").mkdir()
    coord_path = root / "new_coord" / "1.npy"
    box_path = root / "box" / "1.npy"
    np.save(coord_path, np.ones((300, 30, 4), dtype=np.float32))
    np.save(box_path, np.array([1, 2, 3, 4, 5, 1000.0, -17000.0], dtype=np.float64))
    frames_path = tmp_path / "frames.jsonl"
    frame = {
        "dataset": "Dataset",
        "condition": "P1800_T188",
        "data_dir": "data1",
        "frame_id": "1",
        "coord_path": str(coord_path),
        "box_path": str(box_path),
        "coord_sha256": "1".zfill(64),
        "box_sha256": "2".zfill(64),
        "rho": 1000.0,
        "potential": -17000.0,
    }
    frames_path.write_text(json.dumps(frame) + "\n", encoding="utf-8")
    normalizer = TargetNormalizer(900.0, 1200.0, -19000.0, -14000.0, 1)
    normalizer_path = tmp_path / "normalization.json"
    write_normalizer(normalizer, normalizer_path)
    weights_path = tmp_path / "weights.h5"
    cfg = MultiHeadModelConfig(inner_dim=8, m1=100, m2=100)
    model = MultiHeadAutoencoder(cfg, num_heads=1)
    model(np.zeros((1, 300, 30, 4), dtype=np.float32), training=False)
    model.save_weights(weights_path)
    out = tmp_path / "pred"
    training_flags = []
    original_call = MultiHeadAutoencoder.call

    def recording_call(self, inputs, training=False):
        training_flags.append(bool(training))
        return original_call(self, inputs, training=training)

    monkeypatch.setattr(MultiHeadAutoencoder, "call", recording_call)

    assert main([
        "--manifest-jsonl", str(frames_path),
        "--weights", str(weights_path),
        "--normalization", str(normalizer_path),
        "--output-dir", str(out),
        "--condition", "P1800_T188",
        "--batch-size", "1",
        "--inner-dim", "8",
        "--m1", "100",
        "--m2", "100",
        "--head-names", "PC1",
        "--mc-dropout",
    ]) == 0

    assert (out / "PC1_predicted_rows.txt").exists()
    assert not (out / "PCII_predicted_rows.txt").exists()
    summary = json.loads((out / "prediction_summary.json").read_text())
    assert summary["heads"] == ["PC1"]
    assert summary["mc_dropout"] is True
    assert training_flags[-1] is True
