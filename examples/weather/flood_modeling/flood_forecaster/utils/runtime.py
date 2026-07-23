"""Runtime helpers for deterministic data loading and distributed execution."""

from __future__ import annotations

import random
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    Sampler,
    Subset,
    random_split,
)

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


def split_dataset_by_run(
    dataset,
    *,
    train_fraction: float,
    seed: int,
    offset: int = 0,
) -> Tuple[Subset, Subset]:
    r"""Split flood windows without placing one hydrograph in both partitions.

    Datasets containing multiple hydrographs are partitioned by run ID. A dataset
    containing only one run uses an ordered temporal split and removes
    ``dataset.n_history`` windows between partitions so their physical timesteps
    do not overlap.
    """
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1, got {train_fraction}."
        )

    sample_index = getattr(dataset, "sample_index", None)
    if sample_index is None:
        raise TypeError(
            "Run-level splitting requires a dataset exposing sample_index entries "
            "as (run_id, target_timestep) pairs."
        )
    if len(sample_index) != len(dataset):
        raise ValueError(
            "dataset.sample_index must contain exactly one entry per dataset sample."
        )

    run_to_indices: Dict[Any, List[int]] = {}
    for sample_idx, entry in enumerate(sample_index):
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            raise ValueError(
                "Each sample_index entry must be a (run_id, target_timestep) pair; "
                f"entry {sample_idx} is {entry!r}."
            )
        run_id = entry[0]
        try:
            run_to_indices.setdefault(run_id, []).append(sample_idx)
        except TypeError as exc:
            raise TypeError(f"Run ID at sample {sample_idx} must be hashable.") from exc

    run_ids = list(run_to_indices)
    if not run_ids:
        raise ValueError("Cannot split a flood dataset with no eligible samples.")

    if len(run_ids) == 1:
        purge_windows = int(getattr(dataset, "n_history", 0))
        if purge_windows < 0:
            raise ValueError(f"dataset.n_history must be non-negative, got {purge_windows}.")

        ordered_indices = sorted(
            run_to_indices[run_ids[0]],
            key=lambda idx: int(sample_index[idx][1]),
        )
        usable_samples = len(ordered_indices) - purge_windows
        if usable_samples < 2:
            raise ValueError(
                "A one-run dataset needs at least two retained samples plus "
                f"n_history={purge_windows} purge windows for train/validation splitting."
            )

        train_count = int(float(train_fraction) * usable_samples)
        train_count = min(max(train_count, 1), usable_samples - 1)
        val_start = train_count + purge_windows
        warnings.warn(
            "Run-level splitting received only one eligible hydrograph; using a "
            f"chronological split with {purge_windows} purged windows. Validation "
            "measures later-time forecasting on the same run, not unseen-run generalization.",
            UserWarning,
            stacklevel=2,
        )
        return Subset(dataset, ordered_indices[:train_count]), Subset(
            dataset,
            ordered_indices[val_start:],
        )

    permutation = torch.randperm(
        len(run_ids),
        generator=make_torch_generator(seed=seed, offset=offset),
    ).tolist()
    train_run_count = int(float(train_fraction) * len(run_ids))
    train_run_count = min(max(train_run_count, 1), len(run_ids) - 1)
    train_run_ids = {run_ids[idx] for idx in permutation[:train_run_count]}

    train_indices = [
        sample_idx
        for sample_idx, entry in enumerate(sample_index)
        if entry[0] in train_run_ids
    ]
    val_indices = [
        sample_idx
        for sample_idx, entry in enumerate(sample_index)
        if entry[0] not in train_run_ids
    ]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _resolve_run_groups(dataset) -> Dict[Any, List[int]]:
    r"""Map run IDs to indices in the outer dataset consumed by a DataLoader."""
    current = dataset
    base_indices = list(range(len(dataset)))
    visited = set()

    while True:
        current_id = id(current)
        if current_id in visited:
            raise ValueError("Dataset wrappers contain a cycle; cannot resolve run IDs.")
        visited.add(current_id)

        if isinstance(current, Subset):
            base_indices = [int(current.indices[idx]) for idx in base_indices]
            current = current.dataset
            continue

        wrapped_dataset = getattr(current, "base_dataset", None)
        if wrapped_dataset is not None:
            if len(current) != len(wrapped_dataset):
                raise ValueError(
                    "Run-aware sampling requires one-to-one dataset wrappers; "
                    f"got lengths {len(current)} and {len(wrapped_dataset)}."
                )
            current = wrapped_dataset
            continue
        break

    sample_index = getattr(current, "sample_index", None)
    if sample_index is None:
        raise TypeError(
            "Run-aware sampling requires an underlying dataset exposing "
            "sample_index entries as (run_id, target_timestep) pairs."
        )

    run_groups: Dict[Any, List[int]] = {}
    for outer_idx, base_idx in enumerate(base_indices):
        if base_idx < 0 or base_idx >= len(sample_index):
            raise IndexError(
                f"Resolved sample index {base_idx} is outside sample_index with "
                f"length {len(sample_index)}."
            )
        entry = sample_index[base_idx]
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            raise ValueError(
                "Each sample_index entry must be a (run_id, target_timestep) pair; "
                f"entry {base_idx} is {entry!r}."
            )
        try:
            run_groups.setdefault(entry[0], []).append(outer_idx)
        except TypeError as exc:
            raise TypeError(f"Run ID at sample {base_idx} must be hashable.") from exc

    if not run_groups:
        raise ValueError("Run-aware sampling requires at least one training sample.")
    return run_groups


class RunAwareSampler(Sampler[int]):
    r"""Shuffle flood runs and windows while bounding the active run working set.

    The sampler keeps at most ``active_pool_size`` runs open at a time. A run is
    replaced only after all of its windows have been yielded. In distributed mode,
    complete runs are assigned to one rank when the number of runs permits it.
    """

    def __init__(
        self,
        dataset,
        *,
        active_pool_size: int = 4,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if int(active_pool_size) <= 0:
            raise ValueError(
                f"active_pool_size must be a positive integer, got {active_pool_size}."
            )
        if int(num_replicas) <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if not 0 <= int(rank) < int(num_replicas):
            raise ValueError(
                f"rank must be in [0, {int(num_replicas) - 1}], got {rank}."
            )

        self.dataset = dataset
        self.active_pool_size = int(active_pool_size)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.run_groups = _resolve_run_groups(dataset)
        self.run_ids = list(self.run_groups)
        self._sample_level_sharding = len(self.run_ids) < self.num_replicas

        if self._sample_level_sharding:
            self.rank_run_ids = list(self.run_ids)
            self.num_samples = (len(dataset) + self.num_replicas - 1) // self.num_replicas
        else:
            rank_run_ids: List[List[Any]] = [[] for _ in range(self.num_replicas)]
            rank_sample_counts = [0] * self.num_replicas
            run_positions = {run_id: pos for pos, run_id in enumerate(self.run_ids)}
            largest_runs_first = sorted(
                self.run_ids,
                key=lambda run_id: (-len(self.run_groups[run_id]), run_positions[run_id]),
            )
            for run_id in largest_runs_first:
                target_rank = min(
                    range(self.num_replicas),
                    key=lambda rank_idx: (rank_sample_counts[rank_idx], rank_idx),
                )
                rank_run_ids[target_rank].append(run_id)
                rank_sample_counts[target_rank] += len(self.run_groups[run_id])

            self.rank_run_ids = rank_run_ids[self.rank]
            self.num_samples = max(rank_sample_counts)

    def _ordered_indices(
        self,
        run_ids: Sequence[Any],
        generator: torch.Generator,
    ) -> List[int]:
        if not run_ids:
            return []

        run_order = torch.randperm(len(run_ids), generator=generator).tolist()
        shuffled_runs = [run_ids[idx] for idx in run_order]
        windows_by_run: Dict[Any, List[int]] = {}
        for run_id in shuffled_runs:
            run_indices = self.run_groups[run_id]
            window_order = torch.randperm(len(run_indices), generator=generator).tolist()
            windows_by_run[run_id] = [run_indices[idx] for idx in window_order]

        pending_runs = iter(shuffled_runs)
        active_runs: List[Tuple[Any, int]] = []
        for _ in range(min(self.active_pool_size, len(shuffled_runs))):
            active_runs.append((next(pending_runs), 0))

        ordered_indices: List[int] = []
        while active_runs:
            active_slot = int(
                torch.randint(len(active_runs), (1,), generator=generator).item()
            )
            run_id, window_offset = active_runs[active_slot]
            ordered_indices.append(windows_by_run[run_id][window_offset])
            window_offset += 1

            if window_offset < len(windows_by_run[run_id]):
                active_runs[active_slot] = (run_id, window_offset)
                continue

            try:
                active_runs[active_slot] = (next(pending_runs), 0)
            except StopIteration:
                active_runs.pop(active_slot)

        return ordered_indices

    @staticmethod
    def _pad_indices(indices: List[int], total_size: int) -> List[int]:
        if not indices:
            raise ValueError("Cannot pad an empty run-aware sample partition.")
        if len(indices) >= total_size:
            return indices[:total_size]
        padding_size = total_size - len(indices)
        repeats = (padding_size + len(indices) - 1) // len(indices)
        return indices + (indices * repeats)[:padding_size]

    def __iter__(self) -> Iterator[int]:
        generator = make_torch_generator(seed=self.seed, offset=self.epoch)
        if self._sample_level_sharding:
            global_indices = self._ordered_indices(self.run_ids, generator)
            total_size = self.num_samples * self.num_replicas
            global_indices = self._pad_indices(global_indices, total_size)
            return iter(global_indices[self.rank:total_size:self.num_replicas])

        local_indices = self._ordered_indices(self.rank_run_ids, generator)
        return iter(self._pad_indices(local_indices, self.num_samples))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Select the deterministic shuffle for a training epoch."""
        self.epoch = int(epoch)


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
    run_aware: bool = False,
    active_pool_size: int = 4,
    seed: int = 0,
) -> DataLoader:
    """Create a DataLoader with standard or locality-aware distributed sampling."""
    sampler = None
    dist_manager = None
    if DistributedManager.is_initialized():
        dist_manager = DistributedManager()

    if run_aware:
        if not shuffle:
            raise ValueError("Run-aware sampling is a shuffled training-loader mode.")
        num_replicas = (
            dist_manager.world_size if dist_manager and dist_manager.distributed else 1
        )
        rank = dist_manager.rank if dist_manager and dist_manager.distributed else 0
        sampler = RunAwareSampler(
            dataset,
            active_pool_size=active_pool_size,
            seed=seed,
            num_replicas=num_replicas,
            rank=rank,
        )
        shuffle = False
    elif dist_manager and dist_manager.distributed:
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
    run_aware: bool = False,
    active_pool_size: int = 4,
    seed: int = 0,
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
        run_aware=run_aware,
        active_pool_size=active_pool_size,
        seed=seed,
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
    """Propagate the epoch to stateful DataLoader samplers when present."""
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
