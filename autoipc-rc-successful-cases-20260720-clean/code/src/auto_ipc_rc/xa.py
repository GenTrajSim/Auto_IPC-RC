from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ABPlane:
    c1: float
    c2: float
    c3: float
    m_cut: float


@dataclass(frozen=True)
class XAResult:
    mean_rc: np.ndarray
    mean_q1: np.ndarray
    mean_q2: np.ndarray
    mean_q3: np.ndarray
    xa: np.ndarray

    def as_rows(self) -> np.ndarray:
        return np.stack([self.mean_rc, self.mean_q1, self.mean_q2, self.mean_q3, self.xa], axis=1)


def compute_frame_xa(
    pcii_rows: np.ndarray,
    pci_rows: np.ndarray,
    *,
    plane: ABPlane,
    particles_per_frame: int = 300,
) -> XAResult:
    pcii = _rows(pcii_rows, name="pcii_rows")
    pci = _rows(pci_rows, name="pci_rows")
    if pcii.shape[0] != pci.shape[0]:
        raise ValueError(f"PCI and PCII row counts must match, got {pci.shape[0]} and {pcii.shape[0]}")
    if pcii.shape[0] % particles_per_frame != 0:
        raise ValueError(f"row count {pcii.shape[0]} is not divisible by particles_per_frame={particles_per_frame}")

    q1 = pcii[:, 2]
    q2 = pci[:, 1]
    q3 = pci[:, 2]
    denom = float(np.sqrt(plane.c1 * plane.c1 + plane.c2 * plane.c2 + plane.c3 * plane.c3))
    if denom == 0.0:
        raise ValueError("A/B plane coefficients must not all be zero")
    rc = (plane.c1 * q1 + plane.c2 * q2 + plane.c3 * q3 - plane.m_cut) / denom

    frames = pcii.shape[0] // particles_per_frame
    rc_f = rc.reshape(frames, particles_per_frame)
    q1_f = q1.reshape(frames, particles_per_frame)
    q2_f = q2.reshape(frames, particles_per_frame)
    q3_f = q3.reshape(frames, particles_per_frame)
    xa = np.mean(rc_f > 0.0, axis=1)
    return XAResult(
        mean_rc=np.mean(rc_f, axis=1),
        mean_q1=np.mean(q1_f, axis=1),
        mean_q2=np.mean(q2_f, axis=1),
        mean_q3=np.mean(q3_f, axis=1),
        xa=xa,
    )


def summarize_xa(result: XAResult, *, plane: ABPlane, particles_per_frame: int) -> dict[str, float | int | dict[str, float]]:
    return {
        "frames": int(result.xa.size),
        "particles_per_frame": int(particles_per_frame),
        "xA_mean": float(np.mean(result.xa)),
        "xA_std": float(np.std(result.xa)),
        "xA_min": float(np.min(result.xa)),
        "xA_max": float(np.max(result.xa)),
        "mean_RC_mean": float(np.mean(result.mean_rc)),
        "mean_RC_std": float(np.std(result.mean_rc)),
        "plane": asdict(plane),
    }


def _rows(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N,3], got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    return arr
