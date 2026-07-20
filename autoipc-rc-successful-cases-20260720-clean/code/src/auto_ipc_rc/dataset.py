from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from auto_ipc_rc.data_manifest import DataManifest, FrameRecord
from auto_ipc_rc.normalization import TargetNormalizer
from auto_ipc_rc.splits import frame_key


@dataclass(frozen=True)
class FrameArrays:
    coord: np.ndarray
    rho: np.ndarray
    potential: np.ndarray
    frame_keys: tuple[str, ...]


def build_record_index(manifest: DataManifest) -> dict[str, FrameRecord]:
    return {frame_key(record): record for record in manifest.frames}


def load_frame_arrays(
    manifest: DataManifest,
    frame_keys: Sequence[str],
    *,
    normalizer: TargetNormalizer | None,
) -> FrameArrays:
    return load_frame_arrays_from_index(build_record_index(manifest), frame_keys, normalizer=normalizer)


def load_frame_arrays_from_index(
    records_by_key: dict[str, FrameRecord],
    frame_keys: Sequence[str],
    *,
    normalizer: TargetNormalizer | None,
) -> FrameArrays:
    records = [records_by_key[key] for key in frame_keys]
    coord = np.stack([np.load(record.coord_path).astype(np.float32) for record in records], axis=0)
    rho = np.asarray([record.rho for record in records], dtype=np.float32)
    potential = np.asarray([record.potential for record in records], dtype=np.float32)
    if normalizer is not None:
        rho = normalizer.normalize_rho(rho)
        potential = normalizer.normalize_potential(potential)
    return FrameArrays(coord=coord, rho=rho, potential=potential, frame_keys=tuple(frame_keys))


def make_numpy_batches(
    manifest: DataManifest,
    frame_keys: Sequence[str],
    *,
    batch_size: int,
    normalizer: TargetNormalizer | None,
) -> Iterator[FrameArrays]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    index = build_record_index(manifest)
    for start in range(0, len(frame_keys), batch_size):
        yield load_frame_arrays_from_index(index, frame_keys[start : start + batch_size], normalizer=normalizer)
