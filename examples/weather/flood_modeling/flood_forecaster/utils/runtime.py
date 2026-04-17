"""Runtime helpers for deterministic data loading and distributed execution."""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, random_split

from physicsnemo.distributed import DistributedManager


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for deterministic example behavior."""
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


def make_torch_generator(seed: int, offset: int = 0) -> torch.Generator:
    """Create a CPU generator with a deterministic seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(offset))
    return generator


def split_dataset(
    dataset,
    lengths: Sequence[int],
    seed: int,
    offset: int = 0,
):
    """Split a dataset deterministically with an explicit torch Generator."""
    return random_split(
        dataset,
        lengths,
        generator=make_torch_generator(seed=seed, offset=offset),
    )


def create_data_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader that shards data with DistributedSampler when needed."""
    sampler = None
    if DistributedManager.is_initialized():
        dist_manager = DistributedManager()
        if dist_manager.distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**loader_kwargs)


def create_loader_from_config(
    dataset,
    data_config,
    *,
    shuffle: bool,
    drop_last: bool = False,
    collate_fn=None,
) -> DataLoader:
    """Create a DataLoader using the example's config schema."""
    resolved_collate_fn = collate_fn
    if resolved_collate_fn is None and hasattr(dataset, "make_collate_fn"):
        resolved_collate_fn = dataset.make_collate_fn()
    return create_data_loader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(getattr(data_config, "num_workers", 0)),
        pin_memory=bool(getattr(data_config, "pin_memory", False)),
        persistent_workers=bool(getattr(data_config, "persistent_workers", False)),
        collate_fn=resolved_collate_fn,
    )


def _config_get(config: Any, key: str, default=None):
    """Read a config value from dict-like or attribute-based config objects."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def resolve_eval_interval(config: Any, default: int = 1) -> int:
    """Resolve validation cadence, preferring training config over wandb compatibility keys."""
    training_cfg = _config_get(config, "training", None)
    raw_value = _config_get(training_cfg, "eval_interval", None)
    if raw_value is None:
        wandb_cfg = _config_get(config, "wandb", None)
        raw_value = _config_get(wandb_cfg, "eval_interval", default)

    interval = int(raw_value)
    if interval <= 0:
        raise ValueError("training.eval_interval must be a positive integer.")
    return interval


def set_loader_epoch(loader: Optional[DataLoader], epoch: int) -> None:
    """Propagate the epoch to DistributedSampler instances when present."""
    if loader is None:
        return
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def resolve_amp_autocast_enabled(
    requested: bool,
    *,
    device,
    spatial_shape: Optional[Sequence[int]] = None,
    logger: Optional[Any] = None,
    context: str = "training",
) -> bool:
    r"""
    Resolve AMP autocast support for the current device and spatial shape.

    FloodForecaster's GINO/FNO path uses CUDA FFT kernels. Half-precision cuFFT
    rejects non-power-of-two signal sizes, so we proactively disable AMP for
    shapes like 48x48 and emit a clear warning instead of failing at the first
    forward pass.
    """
    enabled = bool(requested)
    device_type = torch.device(device).type if not isinstance(device, torch.device) else device.type
    if not enabled or device_type != "cuda":
        return enabled

    if spatial_shape is None:
        return enabled

    normalized_shape = [int(dim) for dim in spatial_shape if dim is not None]
    if normalized_shape and any(not _is_power_of_two(dim) for dim in normalized_shape):
        message = (
            f"Disabling AMP autocast for {context}: CUDA half-precision FFT requires "
            f"power-of-two spatial dimensions, got {normalized_shape}."
        )
        if logger is not None and hasattr(logger, "warning"):
            logger.warning(message)
        else:
            print(message)
        return False
    return enabled
