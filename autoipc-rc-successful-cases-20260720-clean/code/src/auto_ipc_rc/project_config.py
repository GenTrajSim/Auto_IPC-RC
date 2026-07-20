from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelDimensions:
    m1: int
    m2: int
    n_atoms: int
    n_neighbors: int
    descriptor_dim: int


def load_model_dimensions(path: str | Path) -> ModelDimensions:
    data = _load_yaml(path)
    model = data["model"]
    embedding = model["embedding"]
    input_cfg = model["input"]
    return ModelDimensions(
        m1=int(embedding["m1"]),
        m2=int(embedding["m2"]),
        n_atoms=int(input_cfg["n_atoms"]),
        n_neighbors=int(input_cfg["n_neighbors"]),
        descriptor_dim=int(input_cfg["descriptor_dim"]),
    )


def load_contract_model_dimensions(path: str | Path) -> ModelDimensions:
    data = _load_yaml(path)
    model = data["model"]
    return ModelDimensions(
        m1=int(model["m1"]),
        m2=int(model["m2"]),
        n_atoms=300,
        n_neighbors=30,
        descriptor_dim=4,
    )


def assert_approved_dimensions(model_dims: ModelDimensions, contract_dims: ModelDimensions) -> None:
    if model_dims != contract_dims:
        raise ValueError(f"model dimensions do not match approved contract: {model_dims} != {contract_dims}")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def load_multi_head_model_config(path: str | Path, *, inner_dim: int | None = None, dropout: float | None = None):
    from auto_ipc_rc.models.multi_head_autoencoder import MultiHeadModelConfig

    data = _load_yaml(path)
    model = data["model"]
    embedding = model["embedding"]
    input_cfg = model["input"]
    rho_branch = model["encoder"]["rho_branch"]
    regularization = model.get("regularization", {})
    hidden_layers = rho_branch.get("hidden_layers", [250])
    return MultiHeadModelConfig(
        neighbors=int(input_cfg["n_neighbors"]),
        feature_dim=int(input_cfg["descriptor_dim"]),
        m1=int(embedding["m1"]),
        m2=int(embedding["m2"]),
        inner_dim=int(inner_dim if inner_dim is not None else hidden_layers[0]),
        out_dim=int(rho_branch.get("output_dim", 1)),
        dropout=float(dropout if dropout is not None else regularization.get("dropout", 0.0)),
    )


def load_loss_specs(path: str | Path):
    from auto_ipc_rc.losses import PCConstraintSpec

    data = _load_yaml(path)
    heads = data["loss"]["alpha_phi_per_head"]
    ordered = sorted(heads, key=lambda item: int(item["head_index"]))
    return tuple(
        PCConstraintSpec(
            name=str(item["head_name"]),
            alpha=float(item["alpha_target"]),
            phi_pi_fraction=float(item["phi_target_pi_fraction"]),
        )
        for item in ordered
    )
