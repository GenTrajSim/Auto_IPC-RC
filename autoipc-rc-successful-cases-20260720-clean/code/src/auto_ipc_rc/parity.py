from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from auto_ipc_rc.data_manifest import FrameRecord
from auto_ipc_rc.splits import frame_key


@dataclass(frozen=True)
class ParityFixture:
    coord: np.ndarray
    rho: np.ndarray
    potential: np.ndarray
    frame_keys: np.ndarray
    coord_sha256: np.ndarray
    box_sha256: np.ndarray


def capture_parity_fixture(frames: Iterable[FrameRecord], output_path: str | Path, *, limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = sorted(frames, key=frame_key)[:limit]
    if not selected:
        raise ValueError("cannot capture a parity fixture from an empty frame list")

    coord = np.stack([np.load(frame.coord_path).astype(np.float32) for frame in selected], axis=0)
    rho = np.asarray([frame.rho for frame in selected], dtype=np.float64)
    potential = np.asarray([frame.potential for frame in selected], dtype=np.float64)
    frame_keys = np.asarray([frame_key(frame) for frame in selected])
    coord_sha256 = np.asarray([frame.coord_sha256 for frame in selected])
    box_sha256 = np.asarray([frame.box_sha256 for frame in selected])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        coord=coord,
        rho=rho,
        potential=potential,
        frame_keys=frame_keys,
        coord_sha256=coord_sha256,
        box_sha256=box_sha256,
    )


def load_parity_fixture(path: str | Path) -> ParityFixture:
    data = np.load(path)
    return ParityFixture(
        coord=data["coord"],
        rho=data["rho"],
        potential=data["potential"],
        frame_keys=data["frame_keys"],
        coord_sha256=data["coord_sha256"],
        box_sha256=data["box_sha256"],
    )
