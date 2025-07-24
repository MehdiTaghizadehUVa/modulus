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

"""CSR segment reduction used by :class:`IntegralTransform`."""

from __future__ import annotations

import importlib
from typing import Literal

import torch
from torch import Tensor, einsum


def segment_csr(
    src: Tensor,
    indptr: Tensor,
    reduction: Literal["mean", "sum"],
    use_scatter: bool = True,
) -> Tensor:
    """Reduce features over CSR segments."""
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    if importlib.util.find_spec("torch_scatter") is not None and use_scatter:
        from torch_scatter import segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduction)

    if use_scatter:
        print(
            "Warning: use_scatter is True but torch_scatter is not available; falling back to a slower implementation",
        )

    batched = src.ndim == 3
    point_dim = 1 if batched else 0
    output_shape = list(src.shape)
    n_out = indptr.shape[point_dim] - 1
    output_shape[point_dim] = n_out
    out = torch.zeros(output_shape, device=src.device)

    for i in range(n_out):
        if batched:
            from_idx = (slice(None), slice(indptr[0, i], indptr[0, i + 1]))
            ein_str = "bio->bo"
            start = indptr[0, i]
            n_nbrs = indptr[0, i + 1] - start
            to_idx = (slice(None), i)
        else:
            from_idx = slice(indptr[i], indptr[i + 1])
            ein_str = "io->o"
            start = indptr[i]
            n_nbrs = indptr[i + 1] - start
            to_idx = i

        src_from = src[from_idx]
        if n_nbrs > 0:
            to_reduce = einsum(ein_str, src_from)
            if reduction == "mean":
                to_reduce /= n_nbrs
            out[to_idx] += to_reduce

    return out
