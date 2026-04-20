# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Normalization utilities for flood prediction datasets."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


class FeatureGaussianNormalizer(nn.Module):
    r"""Feature-wise Gaussian normalizer with a neuralop-compatible API."""

    def __init__(
        self,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        mean_tensor = torch.as_tensor(mean if mean is not None else [], dtype=torch.float32)
        std_tensor = torch.as_tensor(std if std is not None else [], dtype=torch.float32)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)
        self.eps = float(eps)

    def _reshape_stats(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mean.numel() == 0 or self.std.numel() == 0:
            raise RuntimeError("Normalizer statistics have not been initialized.")
        view_shape = [1] * max(tensor.ndim - 1, 0) + [self.mean.shape[0]]
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return mean.view(*view_shape), std.view(*view_shape)

    def fit(self, tensor: torch.Tensor) -> "FeatureGaussianNormalizer":
        flat = tensor.reshape(-1, tensor.shape[-1]).to(dtype=torch.float64)
        mean = flat.mean(dim=0)
        var = flat.var(dim=0, unbiased=False)
        self.mean = mean.to(dtype=torch.float32)
        self.std = torch.sqrt(var.clamp_min(0.0)).to(dtype=torch.float32)
        return self

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self._reshape_stats(tensor)
        return (tensor - mean) / (std + self.eps)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        mean, std = self._reshape_stats(tensor)
        return tensor * (std + self.eps) + mean


class _FeatureStatsAccumulator:
    r"""Online accumulator for per-feature mean and variance statistics."""

    def __init__(self) -> None:
        self.count = 0
        self.sum: Optional[torch.Tensor] = None
        self.sumsq: Optional[torch.Tensor] = None

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.reshape(-1, tensor.shape[-1]).to(dtype=torch.float64)
        if flat.numel() == 0:
            return
        if self.sum is None or self.sumsq is None:
            self.sum = torch.zeros(flat.shape[-1], dtype=torch.float64)
            self.sumsq = torch.zeros(flat.shape[-1], dtype=torch.float64)
        self.sum += flat.sum(dim=0)
        self.sumsq += flat.square().sum(dim=0)
        self.count += flat.shape[0]

    def build(self) -> FeatureGaussianNormalizer:
        if self.count <= 0 or self.sum is None or self.sumsq is None:
            raise ValueError("Cannot build normalizer from empty statistics.")
        mean = self.sum / self.count
        var = (self.sumsq / self.count) - mean.square()
        std = torch.sqrt(var.clamp_min(0.0))
        return FeatureGaussianNormalizer(mean=mean.to(dtype=torch.float32), std=std.to(dtype=torch.float32))


def _resolve_subset_indices(dataset) -> Tuple[object, Iterable[int]]:
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        return dataset.dataset, list(dataset.indices)
    return dataset, range(len(dataset))


def collect_all_fields(
    dataset,
    expect_target: bool = True
) -> Tuple[
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[Optional[torch.Tensor]],
    List[Optional[torch.Tensor]],
]:
    r"""
    Collect all fields from a dataset into lists.

    Parameters
    ----------
    dataset : Dataset
        Dataset to collect fields from.
    expect_target : bool, optional, default=True
        Whether to expect target field.

    Returns
    -------
        Tuple[List[torch.Tensor], ...]
        Tuple of lists:
        ``(geometry, static, boundary, dynamic, target, cell_area)``.
        ``cell_area`` is always returned and contains ``None`` entries when absent.

    Raises
    ------
    KeyError
        If required fields are missing.
    """
    geometry_list = []
    static_list = []
    boundary_list = []
    dynamic_list = []
    target_list = []
    cell_area_list = []

    for i in range(len(dataset)):
        sample = dataset[i]
        required_fields = ["geometry", "static", "boundary", "dynamic"]
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            raise KeyError(f"Sample {i} missing required fields: {missing_fields}")

        geometry_list.append(sample["geometry"])
        static_list.append(sample["static"])
        boundary_list.append(sample["boundary"])
        dynamic_list.append(sample["dynamic"])
        target_list.append(sample.get("target", None) if expect_target else sample.get("target", None))
        cell_area_list.append(sample.get("cell_area", None))

    return geometry_list, static_list, boundary_list, dynamic_list, target_list, cell_area_list


def fit_normalizers_from_dataset(dataset) -> Dict[str, FeatureGaussianNormalizer]:
    r"""
    Fit FloodForecaster normalizers incrementally without stacking the full dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset or subset yielding FloodForecaster sample dictionaries.

    Returns
    -------
    Dict[str, FeatureGaussianNormalizer]
        Normalizer dictionary keyed by ``static``, ``boundary``, ``dynamic``, and ``target``.
    """
    base_dataset, _ = _resolve_subset_indices(dataset)
    if hasattr(base_dataset, "sample_index") and hasattr(base_dataset, "get_sample_components"):
        return fit_normalizers_from_sample_index(dataset)

    static_stats = _FeatureStatsAccumulator()
    boundary_stats = _FeatureStatsAccumulator()
    state_stats = _FeatureStatsAccumulator()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if "static" not in sample or "boundary" not in sample or "dynamic" not in sample:
            raise KeyError(f"Sample {idx} is missing one of the required fields: static, boundary, dynamic")
        static_stats.update(sample["static"])
        boundary_stats.update(sample["boundary"])
        state_stats.update(sample["dynamic"])
        target = sample.get("target")
        if target is not None:
            state_stats.update(target)

    static_norm = static_stats.build()
    boundary_norm = boundary_stats.build()
    state_norm = state_stats.build()
    return {
        "static": static_norm,
        "boundary": boundary_norm,
        "dynamic": state_norm,
        "target": state_norm,
    }


def fit_normalizers_from_sample_index(dataset_or_subset) -> Dict[str, FeatureGaussianNormalizer]:
    r"""
    Fit normalizers by grouping selected sample indices by run ID.

    This avoids reloading the same run for every overlapping history window while
    fitting statistics on clean physical states.
    """

    base_dataset, selected_indices = _resolve_subset_indices(dataset_or_subset)
    if not hasattr(base_dataset, "sample_index") or not hasattr(base_dataset, "get_sample_components"):
        raise TypeError(
            "fit_normalizers_from_sample_index requires a dataset exposing "
            "'sample_index' and 'get_sample_components'."
        )

    static_stats = _FeatureStatsAccumulator()
    boundary_stats = _FeatureStatsAccumulator()
    state_stats = _FeatureStatsAccumulator()

    sample_groups = defaultdict(list)
    for sample_idx in selected_indices:
        run_id, target_t = base_dataset.sample_index[sample_idx]
        sample_groups[run_id].append(int(target_t))

    static_stats.update(base_dataset.static_data)

    for run_id, target_timesteps in sample_groups.items():
        for target_t in sorted(target_timesteps):
            sample = base_dataset.get_sample_components(run_id, target_t, apply_noise=False)
            boundary_stats.update(sample["boundary"])
            state_stats.update(sample["dynamic"])
            state_stats.update(sample["target"])

    static_norm = static_stats.build()
    boundary_norm = boundary_stats.build()
    state_norm = state_stats.build()
    return {
        "static": static_norm,
        "boundary": boundary_norm,
        "dynamic": state_norm,
        "target": state_norm,
    }


def stack_and_fit_transform(
    geom_list: List[torch.Tensor],
    static_list: List[torch.Tensor],
    boundary_list: List[torch.Tensor],
    dyn_list: List[torch.Tensor],
    tgt_list: List[Optional[torch.Tensor]],
    normalizers: Optional[Dict[str, FeatureGaussianNormalizer]] = None,
    fit_normalizers: bool = True
) -> Tuple[Dict[str, FeatureGaussianNormalizer], Dict[str, torch.Tensor]]:
    r"""
    Stack field lists into tensors and apply normalization.

    Parameters
    ----------
    geom_list : List[torch.Tensor]
        List of geometry tensors.
    static_list : List[torch.Tensor]
        List of static feature tensors.
    boundary_list : List[torch.Tensor]
        List of boundary condition tensors.
    dyn_list : List[torch.Tensor]
        List of dynamic feature tensors.
    tgt_list : List[Optional[torch.Tensor]]
        List of target tensors.
    normalizers : Dict[str, FeatureGaussianNormalizer], optional
        Dict of existing normalizers (if fit_normalizers=False).
    fit_normalizers : bool, optional, default=True
        Whether to fit new normalizers.

    Returns
    -------
    Tuple[Dict[str, FeatureGaussianNormalizer], Dict[str, torch.Tensor]]
        Tuple of (normalizers dict, big_tensors dict).

    Raises
    ------
    ValueError
        If lists are empty or have incompatible shapes.
    """
    geom_list = [g for g in geom_list if g is not None] if geom_list else []
    static_list = [s for s in static_list if s is not None] if static_list else []
    boundary_list = [b for b in boundary_list if b is not None] if boundary_list else []
    dyn_list = [d for d in dyn_list if d is not None] if dyn_list else []
    tgt_list = [t for t in tgt_list if t is not None] if tgt_list else []

    geometry_big = torch.stack(geom_list, dim=0) if geom_list else None
    static_big = torch.stack(static_list, dim=0) if static_list else None
    boundary_big = torch.stack(boundary_list, dim=0) if boundary_list else None
    dynamic_big = torch.stack(dyn_list, dim=0) if dyn_list else None
    target_big = torch.stack(tgt_list, dim=0) if tgt_list else None

    if normalizers is None:
        normalizers = {}

    if static_big is not None:
        if fit_normalizers:
            static_norm = FeatureGaussianNormalizer()
            static_norm.fit(static_big)
            static_big = static_norm.transform(static_big)
            normalizers["static"] = static_norm
        else:
            static_big = normalizers["static"].transform(static_big)

    if boundary_big is not None:
        if fit_normalizers:
            boundary_norm = FeatureGaussianNormalizer()
            boundary_norm.fit(boundary_big)
            boundary_big = boundary_norm.transform(boundary_big)
            normalizers["boundary"] = boundary_norm
        else:
            boundary_big = normalizers["boundary"].transform(boundary_big)

    state_norm = None
    if fit_normalizers:
        state_tensors = []
        if dynamic_big is not None:
            state_tensors.append(dynamic_big.reshape(-1, dynamic_big.shape[-1]))
        if target_big is not None:
            state_tensors.append(target_big.reshape(-1, target_big.shape[-1]))
        if state_tensors:
            state_norm = FeatureGaussianNormalizer()
            state_norm.fit(torch.cat(state_tensors, dim=0))
            normalizers["dynamic"] = state_norm
            normalizers["target"] = state_norm
    else:
        state_norm = normalizers.get("target") or normalizers.get("dynamic")

    if target_big is not None and state_norm is not None:
        target_big = state_norm.transform(target_big)
        normalizers["target"] = state_norm

    if dynamic_big is not None and state_norm is not None:
        dynamic_big = state_norm.transform(dynamic_big)
        normalizers["dynamic"] = state_norm

    big_tensors = {
        "geometry": geometry_big,
        "static": static_big,
        "boundary": boundary_big,
        "dynamic": dynamic_big,
        "target": target_big,
    }
    return normalizers, big_tensors


def transform_with_existing_normalizers(
    geom_list: List[torch.Tensor],
    static_list: List[torch.Tensor],
    boundary_list: List[torch.Tensor],
    dyn_list: List[torch.Tensor],
    normalizers: Dict[str, FeatureGaussianNormalizer]
) -> Dict[str, torch.Tensor]:
    r"""
    Transform data lists using existing normalizers.

    Parameters
    ----------
    geom_list : List[torch.Tensor]
        List of geometry tensors.
    static_list : List[torch.Tensor]
        List of static feature tensors.
    boundary_list : List[torch.Tensor]
        List of boundary condition tensors.
    dyn_list : List[torch.Tensor]
        List of dynamic feature tensors.
    normalizers : Dict[str, FeatureGaussianNormalizer]
        Dict of normalizers to use.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dict of transformed tensors.

    Raises
    ------
    KeyError
        If required normalizers are missing.
    ValueError
        If lists are empty.
    """
    if not normalizers:
        raise ValueError("normalizers dict cannot be empty")
    transformed = {}
    data_map = {"geometry": geom_list, "static": static_list, "boundary": boundary_list, "dynamic": dyn_list}

    for key, data_list in data_map.items():
        if not data_list:
            continue
        big_tensor = torch.stack(data_list, dim=0)
        if key == "geometry":
            transformed[key] = big_tensor
        elif key in normalizers:
            transformed[key] = normalizers[key].transform(big_tensor)

    return transformed
