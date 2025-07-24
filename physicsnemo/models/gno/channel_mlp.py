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

"""Channel-wise multilayer perceptron used in :class:`GNOBlock`."""

from __future__ import annotations

from typing import Callable, Iterable, List

import torch
from torch import nn


class LinearChannelMLP(nn.Module):
    """Channel MLP operating only on feature channels.

    Parameters
    ----------
    layers : Iterable[int]
        Sequence of layer widths. First element is input dimension and last
        element is output dimension.
    non_linearity : Callable, optional
        Activation function used between linear layers, by default
        :func:`torch.nn.functional.gelu`.
    """

    def __init__(self, layers: Iterable[int], non_linearity: Callable = nn.GELU()) -> None:
        super().__init__()
        layers = list(layers)
        assert len(layers) >= 2, "At least two layers required"
        self.in_channels = layers[0]
        self.out_channels = layers[-1]
        self.fcs = nn.ModuleList(
            [nn.Linear(in_c, out_c) for in_c, out_c in zip(layers[:-1], layers[1:])]
        )
        self.act = non_linearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.fcs):
            x = layer(x)
            if i < len(self.fcs) - 1:
                x = self.act(x)
        return x
