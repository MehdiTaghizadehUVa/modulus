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

"""Kernel integral transform used in :class:`GNOBlock`."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import LinearChannelMLP
from .segment_csr import segment_csr


class IntegralTransform(nn.Module):
    """Integral kernel transform."""

    def __init__(
        self,
        channel_mlp: Optional[nn.Module] = None,
        channel_mlp_layers: Optional[Iterable[int]] = None,
        channel_mlp_non_linearity: Callable = F.gelu,
        transform_type: str = "linear",
        weighting_fn: Optional[Callable] = None,
        reduction: str = "sum",
        use_torch_scatter: bool = True,
    ) -> None:
        super().__init__()
        if channel_mlp is None and channel_mlp_layers is None:
            raise ValueError("channel_mlp or channel_mlp_layers must be provided")

        self.reduction = reduction
        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter
        if transform_type not in {
            "linear_kernelonly",
            "linear",
            "nonlinear_kernelonly",
            "nonlinear",
        }:
            raise ValueError(
                "transform_type must be one of linear_kernelonly, linear, "
                "nonlinear_kernelonly, nonlinear"
            )

        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(
                layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity
            )
        else:
            self.channel_mlp = channel_mlp

        self.weighting_fn = weighting_fn

    def forward(
        self,
        y: torch.Tensor,
        neighbors: dict,
        x: torch.Tensor | None = None,
        f_y: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the integral transform."""
        if x is None:
            x = y

        rep_features = y[neighbors["neighbors_index"]]

        batched = False
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
            elif f_y.ndim == 2:
                in_features = f_y[neighbors["neighbors_index"]]
            else:
                raise ValueError("f_y must have 2 or 3 dimensions")

        num_reps = neighbors["neighbors_row_splits"][1:] - neighbors["neighbors_row_splits"][:-1]
        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        agg_features = torch.cat([rep_features, self_features], dim=-1)
        if f_y is not None and self.transform_type in {"nonlinear_kernelonly", "nonlinear"}:
            if batched:
                agg_features = agg_features.repeat([batch_size] + [1] * agg_features.ndim)
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        rep_features = self.channel_mlp(agg_features)

        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            if rep_features.ndim == 2 and batched:
                rep_features = rep_features.unsqueeze(0).repeat([batch_size] + [1] * rep_features.ndim)
            rep_features.mul_(in_features)

        nbr_weights = neighbors.get("weights")
        if nbr_weights is None:
            nbr_weights = weights
        if nbr_weights is None and self.weighting_fn is not None:
            raise KeyError(
                "if a weighting function is provided, your neighborhoods must contain weights."
            )
        if nbr_weights is not None:
            nbr_weights = nbr_weights.unsqueeze(-1).unsqueeze(0)
            if self.weighting_fn is not None:
                nbr_weights = self.weighting_fn(nbr_weights)
            rep_features.mul_(nbr_weights)
            reduction = "sum"
        else:
            reduction = self.reduction

        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.unsqueeze(0).repeat([batch_size] + [1] * splits.ndim)

        out_features = segment_csr(rep_features, splits, reduction=reduction, use_scatter=self.use_torch_scatter)
        return out_features
