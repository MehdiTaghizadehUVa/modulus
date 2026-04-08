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

"""HydroGraphNet inference and autoregressive rollout."""

import os

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch_geometric.utils import to_networkx

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.utils import load_checkpoint
from utils import build_model, get_batch_vector, roll_feature_window


def create_animation(
    rollout_predictions,
    ground_truth,
    initial_graph,
    rmse_list,
    output_path,
    time_per_step=20 / 60,
):
    """Create a four-panel rollout animation in physical water-depth units."""

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    cax1 = fig.add_axes([0.05, 0.53, 0.02, 0.35])
    cax2 = fig.add_axes([0.95, 0.53, 0.02, 0.35])
    cax3 = fig.add_axes([0.05, 0.1, 0.02, 0.35])

    num_frames = len(rollout_predictions)
    init_node_feats = initial_graph.x
    pos = {
        i: (init_node_feats[i, 0].item(), init_node_feats[i, 1].item())
        for i in range(init_node_feats.shape[0])
    }

    all_vals = torch.cat(rollout_predictions + ground_truth)
    vmin_global = all_vals.min().item()
    vmax_global = all_vals.max().item()
    error_vmax = max(
        torch.max(torch.abs(pred - gt)).item()
        for pred, gt in zip(rollout_predictions, ground_truth)
    )

    graph_cpu = initial_graph.cpu()
    nx_graph = to_networkx(graph_cpu).to_undirected()

    def update(frame):
        for ax in axes.flat:
            ax.clear()
        current_time = (frame + 1) * time_per_step

        pred_vals = rollout_predictions[frame].cpu().numpy()
        nodes_pred = nx.draw_networkx_nodes(
            nx_graph,
            pos,
            node_color=pred_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 0],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.5, ax=axes[0, 0])
        axes[0, 0].set_title(
            f"Time {current_time:.2f} Hours - Prediction (m)", fontsize=24
        )
        fig.colorbar(nodes_pred, cax=cax1)

        gt_vals = ground_truth[frame].cpu().numpy()
        nodes_gt = nx.draw_networkx_nodes(
            nx_graph,
            pos,
            node_color=gt_vals,
            node_size=250,
            cmap=plt.cm.viridis,
            ax=axes[0, 1],
            vmin=vmin_global,
            vmax=vmax_global,
            node_shape="s",
        )
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.5, ax=axes[0, 1])
        axes[0, 1].set_title(
            f"Time {current_time:.2f} Hours - Ground Truth (m)", fontsize=24
        )
        fig.colorbar(nodes_gt, cax=cax2)

        abs_vals = torch.abs(rollout_predictions[frame] - ground_truth[frame]).cpu().numpy()
        nodes_error = nx.draw_networkx_nodes(
            nx_graph,
            pos,
            node_color=abs_vals,
            node_size=250,
            cmap=plt.cm.inferno,
            ax=axes[1, 0],
            vmin=0.0,
            vmax=error_vmax,
            node_shape="s",
        )
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.5, ax=axes[1, 0])
        axes[1, 0].set_title(
            f"Time {current_time:.2f} Hours - Absolute Error (m)", fontsize=24
        )
        fig.colorbar(nodes_error, cax=cax3)

        times = [(i + 1) * time_per_step for i in range(frame + 1)]
        axes[1, 1].plot(
            times,
            rmse_list[: frame + 1],
            label="Water Depth RMSE",
            color="b",
            linewidth=3,
        )
        axes[1, 1].set_title("RMSE Over Time", fontsize=24)
        axes[1, 1].set_xlabel("Time (Hours)", fontsize=24)
        axes[1, 1].set_ylabel("RMSE (m)", fontsize=24)
        axes[1, 1].legend(fontsize=20)
        axes[1, 1].grid(True)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(output_path, writer="pillow", fps=2)
    plt.close(fig)
    print(f"Animation saved to {output_path}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Load a trained HydroGraphNet checkpoint and run autoregressive rollout."""

    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    rollout_length = cfg.get("num_test_time_steps", 10)
    n_time_steps = cfg.get("n_time_steps", 2)
    prefix = cfg.get("prefix", "M80")
    test_dir = cfg.get("test_dir")
    stats_dir = cfg.get("stats_dir", cfg.get("data_dir"))
    test_ids_file = cfg.get("test_ids_file", "test.txt")
    ckpt_path = cfg.get("ckpt_path")
    anim_output_dir = cfg.get("animation_output_dir", "animations")
    os.makedirs(anim_output_dir, exist_ok=True)

    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    test_dataset = HydroGraphDataset(
        data_dir=test_dir,
        stats_dir=stats_dir,
        prefix=prefix,
        n_time_steps=n_time_steps,
        hydrograph_ids_file=test_ids_file,
        split="test",
        rollout_length=rollout_length,
        return_physics=False,
    )
    print(f"Loaded test dataset with {len(test_dataset)} hydrographs.")

    model = build_model(cfg).to(device)
    epoch_loaded = load_checkpoint(
        to_absolute_path(ckpt_path),
        models=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device=device,
    )
    print(f"Checkpoint loaded from epoch {epoch_loaded}")
    model.eval()

    all_rmse_all = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            g, rollout_data = test_dataset[idx]
            g = g.to(device)
            batch = get_batch_vector(g)
            edge_features = g.edge_attr
            x_iter = g.x.clone()
            current_wd = g.current_water_depth_denorm.clone()
            current_vol = g.current_volume_denorm.clone()
            wd_std = g.water_depth_std.reshape(-1)[0]
            vol_std = g.volume_std.reshape(-1)[0]

            inflow_seq = rollout_data["inflow"].to(device)
            precip_seq = rollout_data["precipitation"].to(device)
            wd_gt_seq = rollout_data["water_depth_gt"].to(device)

            rollout_preds = []
            ground_truth_list = []
            rmse_list = []

            for step in range(rollout_length):
                pred = model(x_iter, edge_features, g)
                current_wd = current_wd + pred[:, 0:1] * wd_std
                current_vol = current_vol + pred[:, 1:2] * vol_std

                pred_depth = current_wd.squeeze(1)
                gt_depth = wd_gt_seq[step]
                rollout_preds.append(pred_depth.cpu())
                ground_truth_list.append(gt_depth.cpu())
                rmse_list.append(torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2)).item())

                x_iter = roll_feature_window(
                    x_iter,
                    pred,
                    inflow_seq[step : step + 1],
                    precip_seq[step : step + 1],
                    n_time_steps,
                    batch,
                )

            all_rmse_all.append(rmse_list)
            mean_rmse_sample = sum(rmse_list) / len(rmse_list)
            sample_id = test_dataset.dynamic_data[idx].get("hydro_id", idx)
            print(f"Hydrograph {sample_id}: Mean RMSE = {mean_rmse_sample:.4f} m")

            anim_filename = os.path.join(anim_output_dir, f"animation_{sample_id}.gif")
            create_animation(rollout_preds, ground_truth_list, g.cpu(), rmse_list, anim_filename)

    all_rmse_tensor = torch.tensor(all_rmse_all)
    overall_mean_rmse = torch.mean(all_rmse_tensor, dim=0)
    overall_std_rmse = torch.std(all_rmse_tensor, dim=0, unbiased=False)
    print("Overall Mean RMSE over rollout steps (m):", overall_mean_rmse)
    print("Overall Std RMSE over rollout steps (m):", overall_std_rmse)


if __name__ == "__main__":
    main()
