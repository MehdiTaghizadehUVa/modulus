# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""Neighborhood search utilities for :class:`GNOBlock`."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

# Optional open3d support
open3d_built = False
try:  # pragma: no cover - optional dependency
    from open3d.ml.torch.layers import FixedRadiusSearch

    open3d_built = True
except Exception:  # pragma: no cover - optional dependency
    pass


class NeighborSearch(nn.Module):
    """Neighborhood search between two coordinate meshes."""

    def __init__(self, use_open3d: bool = True, return_norm: bool = False) -> None:
        super().__init__()
        if use_open3d and open3d_built:
            self.search_fn = FixedRadiusSearch()
            self.use_open3d = True
        else:
            self.search_fn = native_neighbor_search
            self.use_open3d = False
        self.return_norm = return_norm

    def forward(self, data: Tensor, queries: Tensor, radius: float) -> Dict[str, Tensor]:
        """Return neighbors within ``radius`` of each query point."""
        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return {
                "neighbors_index": search_return.neighbors_index.long(),
                "neighbors_row_splits": search_return.neighbors_row_splits.long(),
            }
        return self.search_fn(data, queries, radius, self.return_norm)


def native_neighbor_search(
    data: Tensor, queries: Tensor, radius: float, return_norm: bool = False
) -> Dict[str, Tensor]:
    """Pure PyTorch radius-based neighbor search."""
    nbr_dict: Dict[str, Tensor] = {}
    dists = torch.cdist(queries, data)
    eps = 1e-7
    dists = torch.where(dists == 0.0, eps, dists)
    mask = dists <= radius
    nbr_indices = mask.nonzero(as_tuple=False)[:, 1]
    if return_norm:
        weights = dists[mask]
        nbr_dict["weights"] = weights**2
    in_nbr = mask.to(data.dtype)
    nbrhd_sizes = torch.cumsum(in_nbr.sum(dim=1), dim=0)
    splits = torch.cat((torch.tensor([0.0], device=data.device), nbrhd_sizes))

    nbr_dict["neighbors_index"] = nbr_indices.long()
    nbr_dict["neighbors_row_splits"] = splits.long()
    return nbr_dict
