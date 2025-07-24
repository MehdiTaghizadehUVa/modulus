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

"""Sinusoidal positional embeddings."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


class SinusoidalEmbedding(nn.Module):
    """Compute sinusoidal embeddings for input coordinates.

    Parameters
    ----------
    in_channels : int
        Dimension of the input coordinate space.
    num_frequencies : int
        Number of sinusoidal frequencies per dimension.
    embedding_type : Literal["transformer", "nerf"]
        Type of embedding to compute. ``"transformer"`` follows the standard
        transformer positional encoding formulation, while ``"nerf"`` uses the
        NeRF-style frequency embedding. Default is ``"transformer"``.
    max_positions : int, optional
        Maximum sequence length for ``"transformer"`` embeddings. Default 10000.
    """

    def __init__(
        self,
        in_channels: int,
        num_frequencies: int = 32,
        embedding_type: Literal["transformer", "nerf"] = "transformer",
        max_positions: int = 10000,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        self.embedding_type = embedding_type
        self.max_positions = max_positions

        if self.embedding_type not in {"transformer", "nerf"}:
            raise ValueError(
                f"embedding_type must be 'transformer' or 'nerf', got {embedding_type}"
            )

        if self.embedding_type == "transformer":
            self.out_channels = 2 * num_frequencies * in_channels
        else:
            self.out_channels = 2 * num_frequencies * in_channels

    def forward(self, x: Tensor) -> Tensor:  # [*, in_channels]
        if self.embedding_type == "transformer":
            device = x.device
            half_dim = self.num_frequencies
            freq = torch.exp(
                -torch.arange(half_dim, dtype=x.dtype, device=device)
                * (torch.log(torch.tensor(self.max_positions, dtype=x.dtype)) / half_dim)
            )
            freq = freq[None, None, :]
            x = x.unsqueeze(-1) * freq
            emb = torch.cat([x.sin(), x.cos()], dim=-1)
            return emb.view(*x.shape[:-2], -1)
        else:  # nerf
            device = x.device
            freq = 2.0 ** torch.arange(
                self.num_frequencies, dtype=x.dtype, device=device
            )
            freq = freq.view([1] * (x.ndim - 1) + [self.num_frequencies])
            x = x.unsqueeze(-1) * freq * torch.pi
            emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            return emb.view(*x.shape[:-2], -1)
