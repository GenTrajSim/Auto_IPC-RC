from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from auto_ipc_rc.cli.train import _configure_gpu, _load_manifest_jsonl
from auto_ipc_rc.dataset import build_record_index, load_frame_arrays_from_index
from auto_ipc_rc.models.multi_head_autoencoder import MultiHeadAutoencoder, MultiHeadModelConfig
from auto_ipc_rc.normalization import TargetNormalizer
from auto_ipc_rc.splits import frame_key

DEFAULT_HEAD_NAMES = ("PCI", "PCII")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate potential/rho/PC rows from a trained AutoIPC-RC checkpoint.")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", default=None, help="Optional condition filter, e.g. P1800_T188")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-dim", type=int, default=250)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--descriptor-dropout", type=float, default=0.0)
    parser.add_argument("--m1", type=int, default=100)
    parser.add_argument("--m2", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--head-names", default=",".join(DEFAULT_HEAD_NAMES), help="Comma-separated output head names; count defines decoder head count")
    parser.add_argument("--mc-dropout", action="store_true", help="Run prediction with dropout active to match the paper Fig2d evaluation")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow prediction without a visible TensorFlow GPU")
    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    head_names = _parse_head_names(args.head_names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_gpu(allow_cpu=args.allow_cpu)
    manifest = _load_manifest_jsonl(Path(args.manifest_jsonl))
    records = manifest.frames
    if args.condition:
        records = [record for record in records if record.condition == args.condition]
    records = sorted(records, key=lambda record: (record.dataset, record.condition, record.data_dir, int(record.frame_id)))
    if args.max_frames is not None:
        records = records[: args.max_frames]
    if not records:
        raise ValueError("no frames matched prediction filters")

    normalizer = _read_normalizer(Path(args.normalization))
    cfg = MultiHeadModelConfig(neighbors=30, feature_dim=4, m1=args.m1, m2=args.m2, inner_dim=args.inner_dim, dropout=args.dropout, descriptor_dropout=args.descriptor_dropout)
    model = MultiHeadAutoencoder(cfg, num_heads=len(head_names))
    model(tf.zeros((1, 300, 30, 4), dtype=tf.float32), training=False)
    model.load_weights(Path(args.weights))

    index = build_record_index(manifest)
    keys = [frame_key(record) for record in records]
    handles = {name: (output_dir / f"{name}_predicted_rows.txt").open("w", encoding="utf-8") for name in head_names}
    rows_per_head = 0
    try:
        for start in range(0, len(keys), args.batch_size):
            batch_keys = keys[start : start + args.batch_size]
            arrays = load_frame_arrays_from_index(index, batch_keys, normalizer=normalizer)
            outputs = model(tf.constant(arrays.coord), training=args.mc_dropout)
            rho_local = outputs.rho_local.numpy()
            potential_local = outputs.potential_local.numpy()
            pc_heads = outputs.pc_heads.numpy()
            for head_index, name in enumerate(head_names):
                rows = np.stack(
                    [
                        potential_local.reshape(-1),
                        rho_local.reshape(-1),
                        pc_heads[head_index].reshape(-1),
                    ],
                    axis=1,
                )
                np.savetxt(handles[name], rows, fmt="%.10g")
            rows_per_head += int(rho_local.size)
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "condition": args.condition,
        "frames": len(records),
        "rows_per_head": rows_per_head,
        "weights": str(Path(args.weights)),
        "normalization": str(Path(args.normalization)),
        "model_config": asdict(cfg),
        "heads": list(head_names),
        "mc_dropout": bool(args.mc_dropout),
        "allow_cpu": bool(args.allow_cpu),
    }
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _parse_head_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("--head-names must contain at least one name")
    if len(set(names)) != len(names):
        raise ValueError("--head-names must be unique")
    return names


def _read_normalizer(path: Path) -> TargetNormalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field: payload[field] for field in ["rho_min", "rho_max", "potential_min", "potential_max", "fit_frame_count"]}
    return TargetNormalizer(**allowed)


if __name__ == "__main__":
    raise SystemExit(main())
