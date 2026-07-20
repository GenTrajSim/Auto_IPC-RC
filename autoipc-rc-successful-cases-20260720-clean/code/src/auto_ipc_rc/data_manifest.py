from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import hashlib
import json

import numpy as np


DEFAULT_DESCRIPTOR_SHAPE = (300, 30, 4)
RHO_COLUMN = 5
POTENTIAL_COLUMN = 6


class ManifestValidationError(RuntimeError):
    """Raised when manifest construction finds rejected frames in fail-closed mode."""


@dataclass(frozen=True)
class FrameRecord:
    dataset: str
    condition: str
    data_dir: str
    frame_id: str
    coord_path: str
    box_path: str
    coord_sha256: str
    box_sha256: str
    rho: float
    potential: float


@dataclass(frozen=True)
class RejectionRecord:
    dataset: str
    condition: str
    data_dir: str
    frame_id: str
    coord_path: str
    box_path: str | None
    reason: str
    detail: str


@dataclass(frozen=True)
class DataManifest:
    frames: list[FrameRecord]
    rejections: list[RejectionRecord]

    def counts_by_dataset(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for frame in self.frames:
            counts[frame.dataset] = counts.get(frame.dataset, 0) + 1
        return counts

    def counts_by_condition(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for frame in self.frames:
            key = f"{frame.dataset}/{frame.condition}"
            counts[key] = counts.get(key, 0) + 1
        return counts


def build_manifest(
    roots: Iterable[str | Path],
    *,
    expected_descriptor_shape: Sequence[int] = DEFAULT_DESCRIPTOR_SHAPE,
    rho_column: int = RHO_COLUMN,
    potential_column: int = POTENTIAL_COLUMN,
    fail_closed: bool = True,
) -> DataManifest:
    """Build a deterministic complete-frame manifest from descriptor roots.

    The scanner treats each ``new_coord/*.npy`` file as one frame and requires a
    same-named ``box/*.npy`` file under the same data directory. Invalid frames
    are always recorded; in fail-closed mode any rejection aborts the build.
    """
    expected_shape = tuple(int(v) for v in expected_descriptor_shape)
    frames: list[FrameRecord] = []
    rejections: list[RejectionRecord] = []

    for root in sorted(Path(r) for r in roots):
        dataset = root.name
        for coord_path in sorted(root.glob("*/*/new_coord/*.npy")):
            condition = coord_path.parents[2].name
            data_dir = coord_path.parents[1].name
            frame_id = coord_path.stem
            box_path = coord_path.parents[1] / "box" / coord_path.name

            rejection = _validate_frame(
                dataset=dataset,
                condition=condition,
                data_dir=data_dir,
                frame_id=frame_id,
                coord_path=coord_path,
                box_path=box_path,
                expected_shape=expected_shape,
                rho_column=int(rho_column),
                potential_column=int(potential_column),
            )
            if rejection is not None:
                rejections.append(rejection)
                continue

            box = np.load(box_path)
            frames.append(
                FrameRecord(
                    dataset=dataset,
                    condition=condition,
                    data_dir=data_dir,
                    frame_id=frame_id,
                    coord_path=str(coord_path),
                    box_path=str(box_path),
                    coord_sha256=_sha256_file(coord_path),
                    box_sha256=_sha256_file(box_path),
                    rho=float(box[int(rho_column)]),
                    potential=float(box[int(potential_column)]),
                )
            )

    manifest = DataManifest(frames=frames, rejections=rejections)
    if fail_closed and rejections:
        reasons = ", ".join(sorted({r.reason for r in rejections}))
        raise ManifestValidationError(f"manifest rejected {len(rejections)} frame(s): {reasons}")
    return manifest


def _validate_frame(
    *,
    dataset: str,
    condition: str,
    data_dir: str,
    frame_id: str,
    coord_path: Path,
    box_path: Path,
    expected_shape: tuple[int, ...],
    rho_column: int,
    potential_column: int,
) -> RejectionRecord | None:
    if not box_path.exists():
        return _reject(dataset, condition, data_dir, frame_id, coord_path, None, "missing_box", "paired box file is absent")

    try:
        coord = np.load(coord_path)
    except Exception as exc:  # noqa: BLE001 - record bad external data without swallowing it silently.
        return _reject(dataset, condition, data_dir, frame_id, coord_path, box_path, "bad_coord_load", str(exc))

    if tuple(coord.shape) != expected_shape:
        return _reject(
            dataset,
            condition,
            data_dir,
            frame_id,
            coord_path,
            box_path,
            "bad_coord_shape",
            f"expected {expected_shape}, got {tuple(coord.shape)}",
        )
    if not np.isfinite(coord).all():
        return _reject(dataset, condition, data_dir, frame_id, coord_path, box_path, "nonfinite_coord", "coord contains NaN or inf")

    try:
        box = np.load(box_path)
    except Exception as exc:  # noqa: BLE001
        return _reject(dataset, condition, data_dir, frame_id, coord_path, box_path, "bad_box_load", str(exc))

    required_len = max(rho_column, potential_column) + 1
    if box.ndim != 1 or box.shape[0] < required_len:
        return _reject(
            dataset,
            condition,
            data_dir,
            frame_id,
            coord_path,
            box_path,
            "bad_box_shape",
            f"expected 1D box with at least {required_len} values, got {tuple(box.shape)}",
        )
    if not np.isfinite(box).all():
        return _reject(dataset, condition, data_dir, frame_id, coord_path, box_path, "nonfinite_box", "box contains NaN or inf")
    return None


def _reject(
    dataset: str,
    condition: str,
    data_dir: str,
    frame_id: str,
    coord_path: Path,
    box_path: Path | None,
    reason: str,
    detail: str,
) -> RejectionRecord:
    return RejectionRecord(
        dataset=dataset,
        condition=condition,
        data_dir=data_dir,
        frame_id=frame_id,
        coord_path=str(coord_path),
        box_path=str(box_path) if box_path is not None else None,
        reason=reason,
        detail=detail,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_artifacts(manifest: DataManifest, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path / "frames.jsonl", [asdict(frame) for frame in manifest.frames])
    _write_jsonl(output_path / "rejections.jsonl", [asdict(rejection) for rejection in manifest.rejections])
    summary = {
        "accepted_frames": len(manifest.frames),
        "rejected_frames": len(manifest.rejections),
        "counts_by_dataset": manifest.counts_by_dataset(),
        "counts_by_condition": manifest.counts_by_condition(),
        "rejection_reasons": _rejection_counts(manifest),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rejection_counts(manifest: DataManifest) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rejection in manifest.rejections:
        counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
    return counts
