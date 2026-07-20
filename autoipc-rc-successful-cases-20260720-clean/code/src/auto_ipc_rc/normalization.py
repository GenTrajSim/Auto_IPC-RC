from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from auto_ipc_rc.data_manifest import DataManifest
from auto_ipc_rc.splits import frame_key

SCHEME = "paper_midpoint_minmax"
FORMULA = "2*(x-(max_train+min_train)/2)/(max_train-min_train)"


@dataclass(frozen=True)
class TargetNormalizer:
    rho_min: float
    rho_max: float
    potential_min: float
    potential_max: float
    fit_frame_count: int
    scheme: str = SCHEME
    formula: str = FORMULA

    def normalize_rho(self, values: np.ndarray) -> np.ndarray:
        return _normalize(values, self.rho_min, self.rho_max)

    def denormalize_rho(self, values: np.ndarray) -> np.ndarray:
        return _denormalize(values, self.rho_min, self.rho_max)

    def normalize_potential(self, values: np.ndarray) -> np.ndarray:
        return _normalize(values, self.potential_min, self.potential_max)

    def denormalize_potential(self, values: np.ndarray) -> np.ndarray:
        return _denormalize(values, self.potential_min, self.potential_max)


def fit_target_normalizer(manifest: DataManifest, train_keys: tuple[str, ...] | list[str] | set[str]) -> TargetNormalizer:
    train_key_set = set(train_keys)
    selected = [frame for frame in manifest.frames if frame_key(frame) in train_key_set]
    if not selected:
        raise ValueError("cannot fit target normalizer without training frames")
    rho = np.asarray([frame.rho for frame in selected], dtype=np.float64)
    potential = np.asarray([frame.potential for frame in selected], dtype=np.float64)
    return TargetNormalizer(
        rho_min=float(np.min(rho)),
        rho_max=float(np.max(rho)),
        potential_min=float(np.min(potential)),
        potential_max=float(np.max(potential)),
        fit_frame_count=len(selected),
    )


def write_normalizer(normalizer: TargetNormalizer, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(normalizer), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    span = _safe_span(vmin, vmax)
    midpoint = 0.5 * (float(vmax) + float(vmin))
    return np.asarray(2.0 * (values - midpoint) / span, dtype=np.float32)


def _denormalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    span = _safe_span(vmin, vmax)
    midpoint = 0.5 * (float(vmax) + float(vmin))
    return np.asarray(values * span / 2.0 + midpoint, dtype=np.float32)


def _safe_span(vmin: float, vmax: float, eps: float = 1.0e-12) -> float:
    span = float(vmax) - float(vmin)
    return span if abs(span) > eps else eps
