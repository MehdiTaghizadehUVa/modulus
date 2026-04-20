"""Checkpoint helpers for FloodForecaster's PhysicsNeMo-native workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

import physicsnemo


BEST_CHECKPOINT_FILENAME = "best_checkpoint.json"


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP and compiled wrappers before resolving checkpoint filenames."""
    if hasattr(model, "module"):
        model = model.module
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        model = model._orig_mod
    return model


def _unique_model_entries(
    models: Union[torch.nn.Module, Sequence[torch.nn.Module], None]
) -> List[tuple[str, torch.nn.Module]]:
    """Mirror PhysicsNeMo's class-name-based model naming."""
    if models is None:
        return []
    if not isinstance(models, (list, tuple)):
        models = [models]

    grouped: Dict[str, List[torch.nn.Module]] = {}
    for model in models:
        model = _unwrap_model(model)
        name = model.__class__.__name__
        grouped.setdefault(name, []).append(model)

    entries: List[tuple[str, torch.nn.Module]] = []
    for base_name, grouped_models in grouped.items():
        if len(grouped_models) == 1:
            entries.append((base_name, grouped_models[0]))
        else:
            for index, model in enumerate(grouped_models):
                entries.append((f"{base_name}{index}", model))
    return entries


def _model_file_name(path: Union[str, Path], name: str, model: torch.nn.Module, epoch: int) -> Path:
    """Build the expected model file path for a given epoch."""
    model = _unwrap_model(model)
    extension = ".mdlus" if isinstance(model, physicsnemo.Module) else ".pt"
    return Path(path) / f"{name}.0.{epoch}{extension}"


def expected_model_files(
    path: Union[str, Path], models: Union[torch.nn.Module, Sequence[torch.nn.Module], None], epoch: int
) -> List[str]:
    """Return the list of expected model files for validation and sidecars."""
    return [
        _model_file_name(path, name, model, epoch).name
        for name, model in _unique_model_entries(models)
    ]


def expected_training_state_file(path: Union[str, Path], epoch: int) -> str:
    """Return the expected training-state checkpoint filename for an epoch."""
    return (Path(path) / f"checkpoint.0.{epoch}.pt").name


def resolve_checkpoint_epoch(path: Union[str, Path], mode: Union[str, int, None]) -> int:
    """Resolve the checkpoint epoch from an explicit epoch, latest file, or best sidecar."""
    checkpoint_dir = Path(path)
    if isinstance(mode, int):
        return mode
    if mode is None:
        mode = "latest"
    if isinstance(mode, str) and mode.isdigit():
        return int(mode)

    mode = str(mode).lower()
    if mode == "best":
        sidecar_path = checkpoint_dir / BEST_CHECKPOINT_FILENAME
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"Best-checkpoint sidecar not found at {sidecar_path}. "
                "Set checkpoint.inference_epoch=latest or provide a valid best checkpoint."
            )
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "epoch" not in payload:
            raise ValueError(f"Best-checkpoint sidecar {sidecar_path} is missing an epoch field.")
        return int(payload["epoch"])

    if mode != "latest":
        raise ValueError(f"Unsupported checkpoint resolution mode: {mode}")

    checkpoint_files = list(checkpoint_dir.glob("checkpoint.0.*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No PhysicsNeMo training checkpoints found in {checkpoint_dir}")

    epochs: List[int] = []
    for checkpoint_file in checkpoint_files:
        parts = checkpoint_file.name.split(".")
        if len(parts) < 4:
            continue
        try:
            epochs.append(int(parts[2]))
        except ValueError:
            continue
    if not epochs:
        raise FileNotFoundError(f"Could not resolve a checkpoint epoch from files in {checkpoint_dir}")
    return max(epochs)


def validate_checkpoint_files(
    path: Union[str, Path],
    models: Union[torch.nn.Module, Sequence[torch.nn.Module], None],
    epoch: int,
    *,
    require_training_state: bool = True,
) -> Dict[str, Any]:
    """Fail fast when a checkpoint directory is missing expected files."""
    checkpoint_dir = Path(path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint path must be a directory: {checkpoint_dir}")

    missing_files: List[str] = []
    model_files = expected_model_files(checkpoint_dir, models, epoch)
    for file_name in model_files:
        if not (checkpoint_dir / file_name).exists():
            missing_files.append(file_name)

    training_state_file = expected_training_state_file(checkpoint_dir, epoch)
    if require_training_state and not (checkpoint_dir / training_state_file).exists():
        missing_files.append(training_state_file)

    if missing_files:
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} is missing required files for epoch {epoch}: {missing_files}"
        )

    return {
        "epoch": int(epoch),
        "model_files": model_files,
        "training_state_file": training_state_file,
    }


def write_best_checkpoint_metadata(
    path: Union[str, Path],
    *,
    stage: str,
    epoch: int,
    metric_name: str,
    metric_value: float,
    models: Union[torch.nn.Module, Sequence[torch.nn.Module], None],
) -> Path:
    """Write the best-checkpoint sidecar used by inference and manual inspection."""
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = checkpoint_dir / BEST_CHECKPOINT_FILENAME
    payload = {
        "stage": stage,
        "epoch": int(epoch),
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "model_files": expected_model_files(checkpoint_dir, models, epoch),
        "training_state_file": expected_training_state_file(checkpoint_dir, epoch),
    }
    with sidecar_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return sidecar_path


def resolve_legacy_neuralop_checkpoint_name(
    path: Union[str, Path],
    mode: Union[str, int, None],
) -> Optional[str]:
    """Resolve the legacy neuralop checkpoint basename when present."""
    checkpoint_dir = Path(path)
    if isinstance(mode, int) or (isinstance(mode, str) and mode.isdigit()):
        return None

    mode = "latest" if mode is None else str(mode).lower()
    candidates: List[str]
    if mode == "best":
        candidates = ["best_model", "model"]
    elif mode == "latest":
        candidates = ["model", "best_model"]
    else:
        raise ValueError(f"Unsupported legacy checkpoint resolution mode: {mode}")

    for candidate in candidates:
        if (checkpoint_dir / f"{candidate}_state_dict.pt").exists():
            return candidate
    return None
