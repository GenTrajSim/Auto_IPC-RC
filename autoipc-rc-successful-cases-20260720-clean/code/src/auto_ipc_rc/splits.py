from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

from auto_ipc_rc.data_manifest import DataManifest, FrameRecord


@dataclass(frozen=True)
class HoldoutSplit:
    train: tuple[str, ...]
    holdout: tuple[str, ...]


def create_holdout_split(
    manifest: DataManifest,
    *,
    holdout_fraction: float = 0.10,
    seed: int = 2026,
    exclude_condition: str | None = None,
) -> HoldoutSplit:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")

    eligible_by_condition: dict[str, list[FrameRecord]] = {}
    for frame in manifest.frames:
        if exclude_condition is not None and frame.condition == exclude_condition:
            continue
        eligible_by_condition.setdefault(frame.condition, []).append(frame)

    holdout_keys: set[str] = set()
    for condition in sorted(eligible_by_condition):
        frames = eligible_by_condition[condition]
        target = int(math.floor(len(frames) * holdout_fraction))
        if target == 0 and holdout_fraction > 0.0 and frames:
            target = 1
        ranked = sorted(frames, key=lambda frame: _score(frame, seed))
        holdout_keys.update(frame_key(frame) for frame in ranked[:target])

    all_keys = [frame_key(frame) for frame in sorted(manifest.frames, key=frame_key)]
    train = tuple(key for key in all_keys if key not in holdout_keys)
    holdout = tuple(key for key in all_keys if key in holdout_keys)
    return HoldoutSplit(train=train, holdout=holdout)


def frame_key(frame: FrameRecord) -> str:
    return f"{frame.dataset}/{frame.condition}/{frame.data_dir}/{frame.frame_id}"


def _score(frame: FrameRecord, seed: int) -> str:
    payload = "|".join(
        [
            str(seed),
            frame_key(frame),
            frame.coord_sha256,
            frame.box_sha256,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_split_artifact(split: HoldoutSplit, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": list(split.train),
        "holdout": list(split.holdout),
        "train_count": len(split.train),
        "holdout_count": len(split.holdout),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
