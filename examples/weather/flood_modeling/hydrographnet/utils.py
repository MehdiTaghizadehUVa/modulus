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

"""Shared HydroGraphNet helpers for model construction, rollout, and losses."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
    get_hydrograph_feature_layout,
    get_hydrograph_input_dim,
)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when the model is wrapped in DDP."""

    return model.module if isinstance(model, DistributedDataParallel) else model


def get_batch_vector(graph) -> torch.Tensor:
    """Return the PyG batch vector, falling back to a single-graph batch."""

    batch = getattr(graph, "batch", None)
    if batch is None:
        return torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
    return batch


def _expand_graph_scalars(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    values = values.reshape(-1)
    return values[batch].unsqueeze(-1)


def _sum_by_graph(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    if batch.numel() == 0:
        return torch.zeros(0, dtype=values.dtype, device=values.device)
    num_graphs = int(batch.max().item()) + 1
    out = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    out.index_add_(0, batch, values)
    return out


def build_model(cfg):
    """Build MeshGraphKAN from the shared HydroGraphNet configuration."""

    from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN

    input_dim = get_hydrograph_input_dim(cfg.n_time_steps)
    configured_input_dim = cfg.get("num_input_features")
    if configured_input_dim is not None and configured_input_dim != input_dim:
        raise ValueError(
            f"Expected num_input_features={input_dim} for n_time_steps={cfg.n_time_steps}, "
            f"but received {configured_input_dim}."
        )

    mlp_activation = cfg.get("mlp_activation_fn", "relu")
    if cfg.get("recompute_activation", False):
        mlp_activation = "silu"

    return MeshGraphKAN(
        input_dim_nodes=input_dim,
        input_dim_edges=cfg.num_edge_features,
        output_dim=cfg.num_output_features,
        processor_size=cfg.get("processor_size", 15),
        mlp_activation_fn=mlp_activation,
        num_layers_node_processor=cfg.get("num_layers_node_processor", 2),
        num_layers_edge_processor=cfg.get("num_layers_edge_processor", 2),
        hidden_dim_processor=cfg.get("hidden_dim_processor", 128),
        hidden_dim_node_encoder=cfg.get("hidden_dim_node_encoder", 128),
        num_layers_node_encoder=cfg.get("num_layers_node_encoder", 2),
        hidden_dim_edge_encoder=cfg.get("hidden_dim_edge_encoder", 128),
        num_layers_edge_encoder=cfg.get("num_layers_edge_encoder", 2),
        hidden_dim_node_decoder=cfg.get("hidden_dim_node_decoder", 128),
        num_layers_node_decoder=cfg.get("num_layers_node_decoder", 2),
        aggregation=cfg.get("aggregation", "sum"),
        do_concat_trick=cfg.do_concat_trick,
        num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
        checkpoint_offloading=cfg.get("checkpoint_offloading", False),
        recompute_activation=cfg.recompute_activation,
        num_harmonics=cfg.get("num_harmonics", 5),
    )


def roll_feature_window(
    node_features: torch.Tensor,
    pred_delta: torch.Tensor,
    next_inflow: torch.Tensor,
    next_precipitation: torch.Tensor,
    n_time_steps: int,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Roll the normalized HydroGraphNet input window forward by one step.

    The input window ends at anchor time ``t``. The returned feature tensor ends at
    anchor time ``t+1`` and uses forcing values at ``t+1``.
    """

    layout = get_hydrograph_feature_layout(n_time_steps)
    next_features = node_features.clone()

    next_features[:, layout.forcing_slice] = torch.cat(
        [
            _expand_graph_scalars(next_inflow, batch),
            _expand_graph_scalars(next_precipitation, batch),
        ],
        dim=1,
    )

    water_depth_history = node_features[:, layout.water_depth_slice]
    volume_history = node_features[:, layout.volume_slice]
    new_water_depth = water_depth_history[:, -1:] + pred_delta[:, 0:1]
    new_volume = volume_history[:, -1:] + pred_delta[:, 1:2]

    next_features[:, layout.water_depth_slice] = torch.cat(
        [water_depth_history[:, 1:], new_water_depth], dim=1
    )
    next_features[:, layout.volume_slice] = torch.cat(
        [volume_history[:, 1:], new_volume], dim=1
    )
    return next_features


def compute_prediction_loss(
    pred: torch.Tensor, target: torch.Tensor, graph
) -> Dict[str, torch.Tensor]:
    """Compute HydroGraphNet one-step prediction loss in physical units."""

    batch = get_batch_vector(graph)
    water_depth_std = _expand_graph_scalars(graph.water_depth_std, batch)
    volume_std = _expand_graph_scalars(graph.volume_std, batch)
    area_denorm = graph.area_denorm.clamp_min(1e-8)

    depth_error = (pred[:, 0:1] - target[:, 0:1]) * water_depth_std
    volume_error = ((pred[:, 1:2] - target[:, 1:2]) * volume_std) / area_denorm

    loss_depth = torch.mean(depth_error.square())
    loss_volume = torch.mean(volume_error.square())
    total_loss = loss_depth + loss_volume
    return {
        "prediction_loss": total_loss,
        "loss_depth": loss_depth,
        "loss_volume": loss_volume,
    }


def compute_depth_volume_penalty(pred: torch.Tensor, graph) -> torch.Tensor:
    """Penalize physically inconsistent nodewise depth-volume predictions."""

    batch = get_batch_vector(graph)
    water_depth_std = _expand_graph_scalars(graph.water_depth_std, batch)
    volume_std = _expand_graph_scalars(graph.volume_std, batch)
    area_denorm = graph.area_denorm.clamp_min(1e-8)

    predicted_next_depth = graph.current_water_depth_denorm + pred[:, 0:1] * water_depth_std
    predicted_next_volume = graph.current_volume_denorm + pred[:, 1:2] * volume_std
    mean_depth_from_volume = predicted_next_volume / area_denorm
    return torch.mean(F.relu(mean_depth_from_volume - predicted_next_depth).square())


def compute_physics_loss(
    pred: torch.Tensor, graph, delta_t: float = 1200.0
) -> torch.Tensor:
    """
    Compute the HydroGraphNet mass-balance loss using interval-averaged sources.

    The forward inequality constrains the predicted total volume at ``t+1`` using
    the average source term over ``[t, t+1]``. The backward inequality constrains the
    observed volume at ``t+2`` using the predicted volume at ``t+1`` and the average
    source term over ``[t+1, t+2]``.
    """

    batch = get_batch_vector(graph)
    volume_std = _expand_graph_scalars(graph.volume_std, batch).squeeze(-1)
    predicted_total_volume = (
        graph.physics_current_total_volume.reshape(-1)
        + _sum_by_graph(pred[:, 1] * volume_std, batch)
    )

    total_area = graph.total_area.reshape(-1).clamp_min(1e-8)
    forward_violation = F.relu(
        (
            predicted_total_volume
            - (
                graph.physics_current_total_volume.reshape(-1)
                + delta_t * graph.physics_avg_net_source.reshape(-1)
            )
        )
        / total_area
    )
    backward_violation = F.relu(
        (
            graph.physics_future_total_volume.reshape(-1)
            - predicted_total_volume
            - delta_t * graph.physics_next_avg_net_source.reshape(-1)
        )
        / total_area
    )
    return torch.mean(forward_violation.square() + backward_violation.square())


def compute_one_step_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    graph,
    *,
    use_physics_loss: bool,
    delta_t: float,
    physics_penalty_weight: float,
    depth_volume_penalty_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the full one-step HydroGraphNet objective."""

    prediction_loss = compute_prediction_loss(pred, target, graph)
    depth_volume_penalty = compute_depth_volume_penalty(pred, graph)

    total_loss = (
        prediction_loss["prediction_loss"]
        + depth_volume_penalty_weight * depth_volume_penalty
    )
    loss_dict = {
        "prediction_loss": prediction_loss["prediction_loss"],
        "loss_depth": prediction_loss["loss_depth"],
        "loss_volume": prediction_loss["loss_volume"],
        "depth_volume_penalty": depth_volume_penalty,
    }

    physics_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if use_physics_loss:
        physics_loss = compute_physics_loss(pred, graph, delta_t=delta_t)
        total_loss = total_loss + physics_penalty_weight * physics_loss
    loss_dict["physics_loss"] = physics_loss
    loss_dict["total_loss"] = total_loss
    return total_loss, loss_dict
