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

"""Graph Neural Operator block."""

from __future__ import annotations

from typing import Callable, List, Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import LinearChannelMLP
from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch
from .embeddings import SinusoidalEmbedding


class GNOBlock(nn.Module):
    """Graph Neural Operator layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coord_dim: int,
        radius: float,
        transform_type: str = "linear",
        weighting_fn: Optional[Callable] = None,
        reduction: Literal["sum", "mean"] = "sum",
        pos_embedding_type: str = "transformer",
        pos_embedding_channels: int = 32,
        pos_embedding_max_positions: int = 10000,
        channel_mlp_layers: List[int] | None = None,
        channel_mlp_non_linearity: Callable = F.gelu,
        channel_mlp: nn.Module | None = None,
        use_torch_scatter_reduce: bool = True,
        use_open3d_neighbor_search: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim
        self.radius = radius

        self.pos_embedding_type = pos_embedding_type
        if pos_embedding_type in {"nerf", "transformer"}:
            self.pos_embedding = SinusoidalEmbedding(
                in_channels=coord_dim,
                num_frequencies=pos_embedding_channels,
                embedding_type=pos_embedding_type,
                max_positions=pos_embedding_max_positions,
            )
        else:
            self.pos_embedding = None

        if use_open3d_neighbor_search:
            assert (
                self.coord_dim == 3
            ), f"open3d only supports 3D data, got {coord_dim}"
        self.neighbor_search = NeighborSearch(
            use_open3d=use_open3d_neighbor_search, return_norm=weighting_fn is not None
        )

        if self.pos_embedding is None:
            kernel_in_dim = self.coord_dim * 2
            kernel_in_dim_str = "dim(y) + dim(x)"
        else:
            kernel_in_dim = self.pos_embedding.out_channels * 2
            kernel_in_dim_str = "dim(y_embed) + dim(x_embed)"

        if transform_type in {"nonlinear", "nonlinear_kernelonly"}:
            kernel_in_dim += self.in_channels
            kernel_in_dim_str += " + dim(f_y)"

        if channel_mlp is not None:
            if channel_mlp.in_channels != kernel_in_dim:
                raise ValueError(
                    f"ChannelMLP expects {kernel_in_dim} channels ({kernel_in_dim_str}), got {channel_mlp.in_channels}"
                )
            if channel_mlp.out_channels != out_channels:
                raise ValueError(
                    f"ChannelMLP output channels must be {out_channels}, got {channel_mlp.out_channels}"
                )
        elif channel_mlp_layers is not None:
            if channel_mlp_layers[0] != kernel_in_dim:
                channel_mlp_layers = [kernel_in_dim] + list(channel_mlp_layers)
            if channel_mlp_layers[-1] != out_channels:
                channel_mlp_layers.append(out_channels)
            channel_mlp = LinearChannelMLP(
                layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity
            )
        else:
            raise ValueError("Either channel_mlp or channel_mlp_layers must be provided")

        self.integral_transform = IntegralTransform(
            channel_mlp=channel_mlp,
            transform_type=transform_type,
            use_torch_scatter=use_torch_scatter_reduce,
            weighting_fn=weighting_fn,
            reduction=reduction,
        )

    def forward(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        f_y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the operator output."""
        neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)

        if self.pos_embedding is not None:
            y_embed = self.pos_embedding(y)
            x_embed = self.pos_embedding(x)
        else:
            y_embed = y
            x_embed = x

        out_features = self.integral_transform(
            y=y_embed,
            x=x_embed,
            neighbors=neighbors_dict,
            f_y=f_y,
        )
        return out_features
