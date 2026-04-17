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

r"""Normalized dataset wrappers for flood prediction."""

import numpy as np
import torch
from torch.utils.data import Dataset


def _build_query_points(geometry: torch.Tensor, query_res) -> torch.Tensor:
    if geometry is None or geometry.numel() == 0:
        return torch.zeros((query_res[0], query_res[1], 2), dtype=torch.float32)

    geom_sample = geometry.cpu().numpy()
    x_vals = geom_sample[:, 0]
    y_vals = geom_sample[:, 1]
    tx = np.linspace(x_vals.min(), x_vals.max(), query_res[0], dtype=np.float32)
    ty = np.linspace(y_vals.min(), y_vals.max(), query_res[1], dtype=np.float32)
    grid_x, grid_y = np.meshgrid(tx, ty, indexing="ij")
    return torch.tensor(np.stack([grid_x, grid_y], axis=-1), dtype=torch.float32)


def _resolve_query_geometry(dataset) -> torch.Tensor:
    current = dataset
    while hasattr(current, "dataset") and hasattr(current, "indices"):
        current = current.dataset

    geometry = getattr(current, "xy_coords", None)
    if geometry is not None:
        return geometry

    geometry = getattr(current, "geometry", None)
    if geometry is None:
        return None
    if geometry.ndim == 2:
        return geometry
    if geometry.ndim >= 3:
        return geometry[0]
    return geometry


def _resolve_static_tensor(dataset) -> torch.Tensor:
    current = dataset
    while hasattr(current, "dataset") and hasattr(current, "indices"):
        current = current.dataset

    static = getattr(current, "static_data", None)
    if static is not None:
        return static

    static = getattr(current, "static", None)
    if static is None:
        return None
    if static.ndim == 2:
        return static
    if static.ndim >= 3:
        return static[0]
    return static


def _resolve_base_dataset_and_index(dataset, idx):
    current = dataset
    resolved_idx = idx
    while hasattr(current, "dataset") and hasattr(current, "indices"):
        resolved_idx = current.indices[resolved_idx]
        current = current.dataset
    return current, resolved_idx


class SharedFloodBatchCollator:
    r"""Collate FloodForecaster batches while attaching shared tensors only once."""

    def __init__(
        self,
        *,
        geometry: torch.Tensor,
        static: torch.Tensor,
        query_points: torch.Tensor,
    ) -> None:
        self.geometry = geometry
        self.static = static
        self.query_points = query_points

    def __call__(self, batch):
        collated = {
            "geometry": self.geometry,
            "static": self.static,
            "query_points": self.query_points,
        }

        for key in ("boundary", "dynamic", "target"):
            values = [item[key] for item in batch if key in item]
            if values:
                collated[key] = torch.stack(values, dim=0)

        if "cell_area" in batch[0]:
            collated["cell_area"] = batch[0]["cell_area"]
        if "run_id" in batch[0]:
            collated["run_id"] = [item["run_id"] for item in batch]
        if "time_index" in batch[0]:
            collated["time_index"] = torch.as_tensor(
                [item["time_index"] for item in batch],
                dtype=torch.long,
            )
        return collated


class NormalizedDataset(Dataset):
    r"""Dataset wrapper that provides normalized data with query points."""

    def __init__(self, geometry, static, boundary, dynamic, target=None, query_res=None, cell_area=None):
        self.geometry = geometry
        self.static = static
        self.boundary = boundary
        self.dynamic = dynamic
        self.target = target
        self.query_res = query_res if query_res is not None else [64, 64]
        self.cell_area = cell_area

        if self.geometry is not None and self.geometry.shape[0] > 0:
            geometry_for_query = self.geometry[0]
        else:
            geometry_for_query = None
        self.query_points = _build_query_points(geometry_for_query, self.query_res)

    def __len__(self):
        return self.geometry.shape[0] if self.geometry is not None else 0

    def __getitem__(self, idx):
        sample = {
            "geometry": self.geometry[idx],
            "static": self.static[idx],
            "boundary": self.boundary[idx],
            "dynamic": self.dynamic[idx],
            "query_points": self.query_points,
        }
        if self.target is not None:
            sample["target"] = self.target[idx]
        if self.cell_area is not None:
            sample["cell_area"] = self.cell_area[idx]
        return sample


class LazyNormalizedDataset(Dataset):
    r"""Dataset wrapper that normalizes FloodForecaster samples on demand."""

    def __init__(self, base_dataset, normalizers, query_res=None, apply_noise: bool = False):
        self.base_dataset = base_dataset
        self.normalizers = normalizers
        self.query_res = query_res if query_res is not None else [64, 64]
        self.apply_noise = bool(apply_noise)
        self.shared_geometry = _resolve_query_geometry(base_dataset)
        self.shared_query_points = _build_query_points(self.shared_geometry, self.query_res)

        shared_static = _resolve_static_tensor(base_dataset)
        if shared_static is None:
            self.shared_static = None
        elif "static" in self.normalizers:
            self.shared_static = self.normalizers["static"].transform(shared_static)
        else:
            self.shared_static = shared_static

    def make_collate_fn(self) -> SharedFloodBatchCollator:
        return SharedFloodBatchCollator(
            geometry=self.shared_geometry,
            static=self.shared_static,
            query_points=self.shared_query_points,
        )

    def _load_raw_sample(self, idx):
        base_dataset, base_idx = _resolve_base_dataset_and_index(self.base_dataset, idx)
        if hasattr(base_dataset, "sample_index") and hasattr(base_dataset, "get_sample_components"):
            run_id, target_t = base_dataset.sample_index[base_idx]
            sample = base_dataset.get_sample_components(
                run_id,
                target_t,
                apply_noise=self.apply_noise,
            )
            sample["run_id"] = run_id
            sample["time_index"] = target_t
            return sample
        return self.base_dataset[idx]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        raw_sample = self._load_raw_sample(idx)
        sample = {
            "geometry": self.shared_geometry,
            "query_points": self.shared_query_points,
        }
        if self.shared_static is not None:
            sample["static"] = self.shared_static

        for key in ["boundary", "dynamic", "target"]:
            value = raw_sample.get(key)
            if value is None:
                continue
            if key in self.normalizers:
                sample[key] = self.normalizers[key].transform(value)
            else:
                sample[key] = value

        for key in ["cell_area", "run_id", "time_index"]:
            if key in raw_sample:
                sample[key] = raw_sample[key]
        return sample


class LazyNormalizedRolloutDataset(LazyNormalizedDataset):
    r"""Lazy normalization wrapper for full-sequence rollout samples."""


class NormalizedRolloutTestDataset(Dataset):
    r"""Dataset wrapper for normalized rollout test samples."""

    def __init__(self, normalized_samples, query_res=None):
        self.normalized_samples = normalized_samples
        self.query_res = query_res if query_res is not None else [64, 64]
        geometry = None
        if len(self.normalized_samples) > 0:
            geometry = self.normalized_samples[0].get("geometry")
        self.query_points = _build_query_points(geometry, self.query_res)

    def __len__(self):
        return len(self.normalized_samples)

    def __getitem__(self, idx):
        sample = self.normalized_samples[idx].copy()
        sample["query_points"] = self.query_points
        return sample
