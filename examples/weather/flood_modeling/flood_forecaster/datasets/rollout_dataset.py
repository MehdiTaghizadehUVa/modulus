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

r"""Rollout dataset for flood prediction."""

import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from .cache_backend import create_run_store


class FloodRolloutTestDatasetNew(Dataset):
    r"""Dataset for rollout evaluation with channel order ``[WD, VX, VY]``."""

    def __init__(
        self,
        rollout_data_root,
        n_history,
        rollout_length,
        xy_file=None,
        query_res=None,
        static_files=None,
        dynamic_patterns=None,
        boundary_patterns=None,
        raise_on_smaller=True,
        skip_before_timestep=0,
        list_file_name="test.txt",
        backend="auto",
        cache_dir_name=".flood_cache",
        rebuild_cache=False,
        run_cache_size=4,
    ):
        super().__init__()
        self.data_root = Path(rollout_data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Rollout data root not found: {self.data_root}")

        self.n_history = n_history
        self.rollout_length = rollout_length
        self.xy_file = xy_file
        self.query_res = query_res if query_res else [64, 64]
        self.static_files = static_files if static_files else []
        self.dynamic_patterns = dynamic_patterns if dynamic_patterns else {}
        self.boundary_patterns = boundary_patterns if boundary_patterns else {}
        self.raise_on_smaller = raise_on_smaller
        self.skip_before_timestep = skip_before_timestep
        self.list_file_name = list_file_name
        self.backend = backend
        self.cache_dir_name = cache_dir_name
        self.rebuild_cache = rebuild_cache
        self.run_cache_size = int(run_cache_size)

        (
            self.run_store,
            self.manifest,
            self.xy_coords,
            self.static_data,
            self.cell_area,
        ) = create_run_store(
            self.data_root,
            list_file_name=self.list_file_name,
            xy_file=self.xy_file,
            static_files=self.static_files,
            dynamic_patterns=self.dynamic_patterns,
            boundary_patterns=self.boundary_patterns,
            raise_on_smaller=self.raise_on_smaller,
            backend=self.backend,
            cache_dir_name=self.cache_dir_name,
            rebuild_cache=self.rebuild_cache,
        )

        self.run_ids = list(self.manifest.get("run_ids", []))
        self.dynamic_keys = list(self.manifest.get("dynamic_keys", []))
        self.boundary_keys = list(self.manifest.get("boundary_keys", []))
        self.required_dynamic_keys = (
            ["WD", "VX", "VY"]
            if all(key in self.dynamic_keys for key in ["WD", "VX", "VY"])
            else list(self.dynamic_keys)
        )
        self.required_boundary_keys = (
            ["inflow"] if "inflow" in self.boundary_keys else list(self.boundary_keys)
        )
        self.reference_cell_count = int(self.manifest["reference_cell_count"])
        self.run_metadata = dict(self.manifest.get("run_metadata", {}))
        self.sequence_lengths = dict(self.manifest.get("sequence_lengths", {}))
        self._run_cache: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()

        self.valid_run_ids = []
        for run_id in self.run_ids:
            metadata = self.run_metadata.get(run_id, {})
            available_dynamic = metadata.get("available_dynamic_keys", [])
            available_boundary = metadata.get("available_boundary_keys", [])
            if not all(key in available_dynamic for key in self.required_dynamic_keys) or not all(
                key in available_boundary for key in self.required_boundary_keys
            ):
                warnings.warn(
                    f"Run ID {run_id} missing variables: {available_dynamic}, bc: {available_boundary}. Skipping."
                )
                continue

            sequence_length = int(self.sequence_lengths.get(run_id, 0))
            if sequence_length >= self.skip_before_timestep + self.n_history + self.rollout_length:
                self.valid_run_ids.append(run_id)

        if not self.valid_run_ids:
            raise ValueError("No hydrographs have enough time steps for rollout evaluation.")

        self.geometry = self.xy_coords
        self.static = self.static_data

    def _get_run_data(self, run_id: str) -> Dict[str, torch.Tensor]:
        if run_id in self._run_cache:
            self._run_cache.move_to_end(run_id)
            return self._run_cache[run_id]

        run_data = self.run_store.load_run(run_id)
        if self.run_cache_size > 0:
            self._run_cache[run_id] = run_data
            self._run_cache.move_to_end(run_id)
            while len(self._run_cache) > self.run_cache_size:
                self._run_cache.popitem(last=False)
        return run_data

    def get_run_components(self, run_id: str) -> Dict[str, torch.Tensor]:
        run_data = self._get_run_data(run_id)
        sample = {
            "run_id": run_id,
            "dynamic": run_data["dynamic"],
            "boundary": run_data["boundary"],
            "geometry": self.xy_coords,
            "static": self.static_data,
        }
        if self.cell_area is not None:
            sample["cell_area"] = self.cell_area
        return sample

    def __len__(self):
        return len(self.valid_run_ids)

    def __getitem__(self, idx):
        run_id = self.valid_run_ids[idx]
        return self.get_run_components(run_id)
