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

r"""Training dataset for flood prediction with query points."""

import math
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .cache_backend import create_run_store


class FloodDatasetWithQueryPoints(Dataset):
    r"""
    Dataset for training and one-step testing with channel order ``[WD, VX, VY]``.

    The dataset keeps geometry/static tensors in memory and loads per-run dynamic and
    boundary tensors on demand through a shared backend.
    """

    def __init__(
        self,
        data_root,
        n_history,
        query_res=None,
        xy_file=None,
        static_files=None,
        dynamic_patterns=None,
        boundary_patterns=None,
        raise_on_smaller=True,
        skip_before_timestep=0,
        noise_type="none",
        noise_std=None,
        list_file_name="train.txt",
        backend="auto",
        cache_dir_name=".flood_cache",
        rebuild_cache=False,
        run_cache_size=4,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        self.n_history = n_history
        self.query_res = query_res if query_res else [64, 64]
        self.xy_file = xy_file
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

        if noise_std is None or (isinstance(noise_std, (list, tuple)) and len(noise_std) == 0):
            self.noise_type = "none"
            self.noise_std = [0.0, 0.0, 0.0]
        else:
            self.noise_type = noise_type.lower() if noise_type else "none"
            self.noise_std = list(noise_std)
            if len(self.noise_std) != 3:
                raise ValueError("noise_std must be a list of exactly 3 floats for WD, VX, VY.")

        (
            self.run_store,
            self.manifest,
            self.xy_coords,
            self.static_data,
            _,
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
        self.sample_index = []
        self._run_cache: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()

        self._build_sample_indices()

    def _run_has_required_fields(self, run_id: str) -> bool:
        metadata = self.run_metadata.get(run_id, {})
        available_dynamic = metadata.get("available_dynamic_keys", [])
        available_boundary = metadata.get("available_boundary_keys", [])
        return all(key in available_dynamic for key in self.required_dynamic_keys) and all(
            key in available_boundary for key in self.required_boundary_keys
        )

    def _build_sample_indices(self) -> None:
        for run_id in self.run_ids:
            if not self._run_has_required_fields(run_id):
                warnings.warn(f"Run ID {run_id} missing required dynamic/boundary variables. Skipping.")
                continue
            sequence_length = int(self.sequence_lengths.get(run_id, 0))
            start_t = max(self.n_history, self.skip_before_timestep)
            for target_t in range(start_t, sequence_length):
                self.sample_index.append((run_id, target_t))

    def __len__(self):
        return len(self.sample_index)

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

    def _build_sample_components(
        self,
        run_id: str,
        target_t: int,
        *,
        apply_noise: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        run_data = self._get_run_data(run_id)
        dynamic = run_data["dynamic"]
        boundary = run_data["boundary"]

        t0 = target_t - self.n_history
        dynamic_hist = dynamic[t0:target_t].clone()
        if apply_noise:
            dynamic_hist = self._apply_noise(dynamic_hist)
        boundary_hist = boundary[t0:target_t].clone()
        target = dynamic[target_t].clone()
        return dynamic_hist, boundary_hist, target

    def get_sample_components(
        self,
        run_id: str,
        target_t: int,
        *,
        apply_noise: bool = False,
    ) -> Dict[str, torch.Tensor]:
        dynamic_hist, boundary_hist, target = self._build_sample_components(
            run_id,
            target_t,
            apply_noise=apply_noise,
        )
        return {
            "geometry": self.xy_coords,
            "static": self.static_data,
            "boundary": boundary_hist,
            "dynamic": dynamic_hist,
            "target": target,
        }

    def _apply_noise(self, dynamic_hist: torch.Tensor):
        if self.noise_type == "none" or all(std <= 0.0 for std in self.noise_std):
            return dynamic_hist

        n_history, num_cells, channels = dynamic_hist.shape
        device = dynamic_hist.device

        def make_noise(n_steps: int, n_cells: int) -> torch.Tensor:
            base = torch.randn((n_steps, n_cells, channels), device=device)
            std_tensor = torch.tensor(self.noise_std, device=device).view(1, 1, channels)
            return base * std_tensor

        if self.noise_type == "only_last":
            dynamic_hist[-1] += make_noise(1, num_cells)[0]
        elif self.noise_type == "correlated":
            shared_noise = make_noise(1, num_cells)[0]
            for t in range(n_history):
                dynamic_hist[t] += shared_noise
        elif self.noise_type == "uncorrelated":
            dynamic_hist += make_noise(n_history, num_cells)
        elif self.noise_type == "random_walk":
            step_sigma = [std / math.sqrt(max(n_history, 1)) for std in self.noise_std]
            step_sigma_tensor = torch.tensor(step_sigma, device=device).view(1, 1, channels)
            offset = torch.zeros((num_cells, channels), device=device)
            for t in range(n_history):
                offset += torch.randn((num_cells, channels), device=device) * step_sigma_tensor.squeeze()
                dynamic_hist[t] += offset
        else:
            warnings.warn(f"Unknown noise_type={self.noise_type}, skipping noise.")
        return dynamic_hist

    def __getitem__(self, idx):
        run_id, target_t = self.sample_index[idx]
        sample = self.get_sample_components(run_id, target_t, apply_noise=True)
        sample["run_id"] = run_id
        sample["time_index"] = target_t
        return sample
