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

r"""
Data processor and loss wrappers for the FloodForecaster GINO pipeline.

Model and layer implementations live in the example-local ``models`` package.
"""

from typing import Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from jaxtyping import Float

import physicsnemo
from physicsnemo.core.module import ModelMetaData

from models.types import ModelOutputTensor, ProcessedGINOBatch, RawFloodSample


class LpLossWrapper:
    r"""
    Wrapper around LpLoss that filters out unexpected kwargs.

    The neuralop Trainer calls ``loss(out, **sample)`` where sample contains
    all keys including model inputs. This wrapper filters to only pass ``y``.

    Parameters
    ----------
    loss_fn : Any
        Wrapped loss function that consumes prediction and target tensors.

    Forward
    -------
    y_pred : torch.Tensor
        Predicted tensor of shape :math:`(B, \ldots)`.
    y : torch.Tensor, optional
        Target tensor of shape :math:`(B, \ldots)`.

    Outputs
    -------
    torch.Tensor
        Scalar loss tensor returned by ``loss_fn``.
    """

    def __init__(self, loss_fn):
        if loss_fn is None:
            raise ValueError("loss_fn cannot be None")
        self.loss_fn = loss_fn

    def __call__(
        self,
        y_pred: Float[torch.Tensor, "batch *shape"],
        y: Optional[Float[torch.Tensor, "batch *shape"]] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Compute the wrapped loss using only the prediction and target tensors.

        Parameters
        ----------
        y_pred : Float[torch.Tensor, "batch *shape"]
            Predicted tensor of shape :math:`(B, \ldots)`.
        y : Float[torch.Tensor, "batch *shape"], optional, default=None
            Target tensor of shape :math:`(B, \ldots)`.
        **kwargs : Any
            Ignored extra sample entries passed by the trainer.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor produced by the wrapped loss function.
        """
        del kwargs
        if y is None:
            raise ValueError("Expected target tensor 'y' in sample for loss computation")
        return self.loss_fn(y_pred, y)


class FloodGINODataProcessor(physicsnemo.Module):
    r"""
    Data processor for FloodForecaster's GINO training pipeline.

    Parameters
    ----------
    device : str or torch.device, optional, default="cuda"
        Device used to stage tensors before model execution.
    target_norm : nn.Module, optional, default=None
        Optional target normalizer used to invert predictions during evaluation.
    inverse_test : bool, optional, default=True
        Whether to invert normalized outputs and targets during evaluation.

    Forward
    -------
    geometry : torch.Tensor
        Shared geometry tensor of shape :math:`(N, C_{coord})` or :math:`(1, N, C_{coord})`.
    static : torch.Tensor
        Static-cell features of shape :math:`(N, C_{static})` or :math:`(B, N, C_{static})`.
    boundary : torch.Tensor
        Compact boundary history of shape :math:`(T_{h}, C_{bc})` or :math:`(B, T_{h}, C_{bc})`.
    dynamic : torch.Tensor
        Dynamic state history of shape :math:`(B, T_{h}, N, C_{dyn})` or :math:`(T_{h}, N, C_{dyn})`.
    query_points : torch.Tensor
        Latent query grid of shape :math:`(H, W, C_{coord})` or :math:`(1, H, W, C_{coord})`.
    target : torch.Tensor, optional
        Optional next-step target tensor of shape :math:`(B, N, C_{out})` or :math:`(N, C_{out})`.

    Outputs
    -------
    tuple
        Tuple of processed model outputs and processed sample dictionaries matching
        the neuralop trainer contract.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        target_norm: Optional[nn.Module] = None,
        inverse_test: bool = True,
    ):
        super().__init__(meta=ModelMetaData())
        if not isinstance(device, (str, torch.device)):
            raise TypeError(f"device must be a string or torch.device, got {type(device)}")
        self._device_str = str(device) if isinstance(device, torch.device) else device
        super().to(device)
        self.model: Optional[nn.Module] = None
        self.target_norm = target_norm
        self.inverse_test = inverse_test

    def preprocess(
        self, sample: RawFloodSample, batched: bool = True
    ) -> ProcessedGINOBatch:
        r"""
        Convert raw flood samples into the GINO input contract.

        Parameters
        ----------
        sample : RawFloodSample
            Raw sample dictionary containing geometry, static, boundary, dynamic,
            query, and optional target tensors.
        batched : bool, optional, default=True
            Compatibility flag retained for upstream trainer hooks.

        Returns
        -------
        ProcessedGINOBatch
            Processed batch dictionary containing ``input_geom``, ``latent_queries``,
            ``output_queries``, ``x``, and optional ``y``.
        """
        if not torch.compiler.is_compiling():
            required_keys = ["geometry", "static", "boundary", "dynamic", "query_points"]
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                raise KeyError(f"Missing required keys in sample: {missing_keys}")

            for key in required_keys:
                value = sample[key]
                if isinstance(value, torch.Tensor) and value.ndim < 2:
                    raise ValueError(
                        f"Expected {key} to be at least 2D tensor, got {value.ndim}D "
                        f"with shape {tuple(value.shape)}"
                    )

        del batched  # kept for API compatibility with upstream trainer hooks

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.to(self.device)

        dyn_ = sample["dynamic"]
        if dyn_.dim() == 3:
            dyn_ = dyn_.unsqueeze(0)
        # Flatten the temporal dynamic history into per-cell feature channels.
        dyn_ = dyn_.permute(0, 2, 1, 3)
        batch_size, num_cells, history_steps, dynamic_channels = dyn_.shape
        dyn_ = dyn_.reshape(batch_size, num_cells, history_steps * dynamic_channels)

        static_ = sample["static"]
        if static_.dim() == 2:
            static_ = static_.unsqueeze(0).expand(batch_size, -1, -1)
        elif static_.dim() == 3:
            if static_.shape[0] == 1 and batch_size > 1:
                static_ = static_.expand(batch_size, -1, -1)
            elif static_.shape[0] != batch_size:
                raise ValueError(
                    "Static batch dimension "
                    f"{static_.shape[0]} must match dynamic batch dimension {batch_size}"
                )
        else:
            raise ValueError(
                "Expected static to have shape (num_cells, static_dim) or "
                f"(B, num_cells, static_dim), got {tuple(static_.shape)}"
            )

        boundary_ = sample["boundary"]
        if boundary_.dim() == 2:
            boundary_ = boundary_.unsqueeze(0)
        elif boundary_.dim() != 3:
            raise ValueError(
                "Expected boundary to have shape (n_history, bc_dim) or "
                f"(B, n_history, bc_dim), got {tuple(boundary_.shape)}"
            )
        if boundary_.shape[0] == 1 and batch_size > 1:
            boundary_ = boundary_.expand(batch_size, -1, -1)
        elif boundary_.shape[0] != batch_size:
            raise ValueError(
                "Boundary batch dimension "
                f"{boundary_.shape[0]} must match dynamic batch dimension {batch_size}"
            )
        # Broadcast the compact hydrograph over cells only at the final feature build.
        boundary_ = boundary_.unsqueeze(2).expand(-1, -1, static_.shape[1], -1)
        _, boundary_history, _, boundary_channels = boundary_.shape
        boundary_ = boundary_.permute(0, 2, 1, 3).reshape(
            batch_size,
            static_.shape[1],
            boundary_history * boundary_channels,
        )

        features = torch.cat([static_, boundary_, dyn_], dim=2)

        geometry = sample["geometry"]
        if geometry.dim() == 3:
            geometry = geometry[0]

        target = sample.get("target")
        if target is not None and target.dim() == 2:
            target = target.unsqueeze(0)

        query_points = sample["query_points"]
        if query_points.dim() == 4:
            query_points = query_points[0]

        processed: ProcessedGINOBatch = {
            "input_geom": geometry,
            "latent_queries": query_points,
            "output_queries": geometry.clone(),
            "x": features,
        }
        if target is not None:
            processed["y"] = target
        return processed

    def postprocess(
        self, out: ModelOutputTensor, sample: ProcessedGINOBatch
    ) -> Tuple[ModelOutputTensor, ProcessedGINOBatch]:
        r"""
        Optionally denormalize model outputs for evaluation.

        Parameters
        ----------
        out : ModelOutputTensor
            Model output tensor of shape :math:`(B, N, C_{out})`.
        sample : ProcessedGINOBatch
            Processed batch dictionary containing the optional target tensor.

        Returns
        -------
        Tuple[ModelOutputTensor, ProcessedGINOBatch]
            Postprocessed output tensor and updated processed sample dictionary.
        """
        if not torch.compiler.is_compiling():
            if not isinstance(out, torch.Tensor):
                raise TypeError("Expected ``out`` to be a torch.Tensor")
            if out.ndim < 2:
                raise ValueError(
                    "Expected tensor of shape :math:`(B, N, C_{out})` "
                    f"but got tensor of shape {tuple(out.shape)}"
                )
            if not isinstance(sample, dict):
                raise TypeError("Expected ``sample`` to be a dict")

        if (not self.training) and self.inverse_test and (self.target_norm is not None):
            out = self.target_norm.inverse_transform(out)
            if sample.get("y") is not None:
                sample["y"] = self.target_norm.inverse_transform(sample["y"])
        return out, sample

    def to(self, device: Union[str, torch.device]) -> "FloodGINODataProcessor":
        r"""
        Move the processor and optional target normalizer to a device.

        Parameters
        ----------
        device : str or torch.device
            Target device for the processor state.

        Returns
        -------
        FloodGINODataProcessor
            The processor instance moved to ``device``.
        """
        if not isinstance(device, (str, torch.device)):
            raise TypeError(f"device must be a string or torch.device, got {type(device)}")
        self._device_str = str(device) if isinstance(device, torch.device) else device
        super().to(device)
        if self.target_norm is not None:
            self.target_norm = self.target_norm.to(device)
        return self

    def wrap(self, model: nn.Module) -> "FloodGINODataProcessor":
        r"""
        Attach a model for ``forward``-based processor execution.

        Parameters
        ----------
        model : nn.Module
            Model implementing the FloodForecaster GINO input contract.

        Returns
        -------
        FloodGINODataProcessor
            The processor instance with the attached model.
        """
        if model is None:
            raise ValueError("model cannot be None")
        self.model = model
        return self

    def _forward_impl(
        self, sample: RawFloodSample
    ) -> Tuple[ModelOutputTensor, ProcessedGINOBatch]:
        processed = self.preprocess(sample)
        if self.model is None:
            raise RuntimeError("No model attached. Call wrap(model).")

        out = self.model(
            input_geom=processed["input_geom"],
            latent_queries=processed["latent_queries"],
            output_queries=processed["output_queries"],
            x=processed["x"],
        )
        return self.postprocess(out, processed)

    def forward(self, **data_dict) -> Tuple[ModelOutputTensor, ProcessedGINOBatch]:
        r"""
        Preprocess a raw sample, run the wrapped model, and postprocess outputs.

        Parameters
        ----------
        **data_dict : Any
            Raw sample entries matching ``RawFloodSample``.

        Returns
        -------
        Tuple[ModelOutputTensor, ProcessedGINOBatch]
            Postprocessed model output tensor and processed sample dictionary.
        """
        return self._forward_impl(cast(RawFloodSample, data_dict))
