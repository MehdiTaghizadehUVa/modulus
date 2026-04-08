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

import time

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb
from utils import (
    build_model,
    compute_one_step_loss,
    compute_prediction_loss,
    get_batch_vector,
    roll_feature_window,
    unwrap_model,
)

try:
    import wandb
except ImportError:
    wandb = None


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = cfg.amp
        self.n_time_steps = cfg.n_time_steps
        self.noise_type = cfg.noise_type
        self.wandb_enabled = cfg.wandb_mode != "disabled"

        self.use_physics_loss = cfg.get("use_physics_loss", False)
        self.delta_t = cfg.get("delta_t", 1200.0)
        self.physics_penalty_weight = cfg.get(
            "physics_penalty_weight", cfg.get("physics_loss_weight", 1.0)
        )
        self.depth_volume_penalty_weight = cfg.get(
            "depth_volume_penalty_weight", 1.0
        )
        self.pushforward_stability_weight = cfg.get(
            "pushforward_stability_weight", 1.0
        )

        rank_zero_logger.info("Initializing HydroGraphDataset...")
        dataset_start = time.perf_counter()
        dataset = HydroGraphDataset(
            name="hydrograph_dataset",
            data_dir=cfg.data_dir,
            stats_dir=cfg.get("stats_dir", cfg.data_dir),
            prefix=cfg.get("prefix", "M80"),
            num_samples=cfg.get("num_training_samples", 500),
            n_time_steps=cfg.n_time_steps,
            k=cfg.get("k_neighbors", 4),
            noise_type=cfg.noise_type,
            noise_std=cfg.get("noise_std", 0.01),
            hydrograph_ids_file=cfg.get("train_ids_file", "train.txt"),
            split="train",
            return_physics=self.use_physics_loss,
        )
        self.sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
        )
        self.dataloader = PyGDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=self.sampler,
            pin_memory=True,
            num_workers=cfg.num_dataloader_workers,
        )
        self.dataset_init_seconds = time.perf_counter() - dataset_start
        self.dataset_num_samples = len(dataset)
        self.dataset_num_batches = len(self.dataloader)
        rank_zero_logger.info("Dataset and dataloader initialization complete.")

        rank_zero_logger.info("Instantiating MeshGraphKAN model...")
        model_init_start = time.perf_counter()
        self.model = build_model(cfg)
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphKAN is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        self.model_init_seconds = time.perf_counter() - model_init_start
        rank_zero_logger.info("Model instantiated successfully.")

        if (
            self.wandb_enabled
            and cfg.watch_model
            and not cfg.jit
            and self.dist.rank == 0
        ):
            if wandb is None:
                raise ImportError(
                    "wandb_mode is enabled and watch_model=True, but wandb could not be imported."
                )
            wandb.watch(self.model)

        if self.dist.world_size > 1:
            rank_zero_logger.info("Wrapping model in DistributedDataParallel...")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(
                    self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
                )
            else:
                self.optimizer = None
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex is not installed; FusedAdam optimizer will not be used."
            )
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        rank_zero_logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: cfg.lr_decay_rate**step
        )
        self.scaler = GradScaler(enabled=self.amp)

        rank_zero_logger.info("Loading checkpoint if available...")
        checkpoint_load_start = time.perf_counter()
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=unwrap_model(self.model),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        self.checkpoint_load_seconds = time.perf_counter() - checkpoint_load_start
        rank_zero_logger.info(
            f"Checkpoint loaded. Starting training from epoch {self.epoch_init}."
        )

    def train_batch(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        detached = {key: value.detach() for key, value in loss_dict.items()}
        return detached

    def forward(self, graph):
        with autocast(device_type=self.dist.device.type, enabled=self.amp):
            pred = self.model(graph.x, graph.edge_attr, graph)
            total_loss, loss_dict = compute_one_step_loss(
                pred,
                graph.y,
                graph,
                use_physics_loss=self.use_physics_loss,
                delta_t=self.delta_t,
                physics_penalty_weight=self.physics_penalty_weight,
                depth_volume_penalty_weight=self.depth_volume_penalty_weight,
            )

            if self.noise_type == "pushforward":
                batch = get_batch_vector(graph)
                rolled_x = roll_feature_window(
                    graph.x,
                    pred.detach(),
                    graph.next_inflow,
                    graph.next_precipitation,
                    self.n_time_steps,
                    batch,
                )
                pred_pushforward = self.model(rolled_x, graph.edge_attr, graph)
                stability_components = compute_prediction_loss(
                    pred_pushforward, graph.y_pushforward, graph
                )
                stability_loss = stability_components["prediction_loss"]
                total_loss = total_loss + (
                    self.pushforward_stability_weight * stability_loss
                )
                loss_dict["stability_loss"] = stability_loss
                loss_dict["stability_loss_depth"] = stability_components["loss_depth"]
                loss_dict["stability_loss_volume"] = stability_components["loss_volume"]
            else:
                zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
                loss_dict["stability_loss"] = zero
                loss_dict["stability_loss_depth"] = zero
                loss_dict["stability_loss_volume"] = zero

            loss_dict["total_loss"] = total_loss
            return total_loss, loss_dict

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


def _reduce_epoch_metrics(metric_sums, num_batches, device):
    keys = sorted(metric_sums.keys())
    values = torch.tensor(
        [metric_sums[key] for key in keys] + [float(num_batches)],
        device=device,
        dtype=torch.float,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    denom = max(values[-1].item(), 1.0)
    return {key: values[idx].item() / denom for idx, key in enumerate(keys)}


def _reduce_epoch_counters(counter_sums, device):
    keys = sorted(counter_sums.keys())
    values = torch.tensor(
        [float(counter_sums[key]) for key in keys],
        device=device,
        dtype=torch.float,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return {key: values[idx].item() for idx, key in enumerate(keys)}


def _reduce_max_scalar(value, device):
    tensor = torch.tensor(float(value), device=device, dtype=torch.float)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return tensor.item()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()
    if cfg.wandb_mode != "disabled":
        if wandb is None:
            raise ImportError(
                "wandb_mode is not disabled, but wandb could not be imported. "
                "Install wandb and its dependencies or set wandb_mode=disabled."
            )
        initialize_wandb(
            project="HydroGraphNet",
            entity="Modulus",
            name="HydroGraphNet-Training",
            group="HydroGraphNet-DDP",
            mode=cfg.wandb_mode,
        )
    else:
        rank_zero_logger.info("wandb_mode=disabled, skipping Weights & Biases setup.")
    rank_zero_logger.info(f"Starting training process with configuration: {cfg}")

    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info(
        "Initialization timing | "
        f"dataset: {trainer.dataset_init_seconds:.2f}s | "
        f"model: {trainer.model_init_seconds:.2f}s | "
        f"checkpoint_load: {trainer.checkpoint_load_seconds:.2f}s | "
        f"samples: {trainer.dataset_num_samples} | "
        f"batches/epoch: {trainer.dataset_num_batches}"
    )
    rank_zero_logger.info("Beginning training loop...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        trainer.model.train()
        trainer.sampler.set_epoch(epoch)
        metric_sums = {}
        num_batches = 0
        epoch_samples = 0
        data_wait_seconds = 0.0
        train_step_seconds = 0.0
        checkpoint_seconds = 0.0
        epoch_start = time.perf_counter()
        batch_end = epoch_start

        for graph in trainer.dataloader:
            batch_start = time.perf_counter()
            data_wait_seconds += batch_start - batch_end
            train_start = time.perf_counter()
            loss_dict = trainer.train_batch(graph)
            train_step_seconds += time.perf_counter() - train_start
            for key, value in loss_dict.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value.item()
            num_batches += 1
            epoch_samples += int(getattr(graph, "num_graphs", 1))
            batch_end = time.perf_counter()

        epoch_metrics = _reduce_epoch_metrics(metric_sums, num_batches, dist.device)
        counter_totals = _reduce_epoch_counters(
            {
                "num_batches": num_batches,
                "num_samples": epoch_samples,
                "data_wait_seconds": data_wait_seconds,
                "train_step_seconds": train_step_seconds,
            },
            dist.device,
        )
        epoch_wall_seconds = _reduce_max_scalar(
            time.perf_counter() - epoch_start, dist.device
        )
        rank_zero_logger.info(
            f"Epoch {epoch} completed. Average total loss: {epoch_metrics['total_loss']:.4e}"
        )

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            checkpoint_start = time.perf_counter()
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=unwrap_model(trainer.model),
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch + 1,
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_start
            rank_zero_logger.info(f"Checkpoint saved after epoch {epoch}.")

        checkpoint_seconds = _reduce_max_scalar(checkpoint_seconds, dist.device)
        total_batches = max(counter_totals["num_batches"], 1.0)
        total_samples = max(counter_totals["num_samples"], 1.0)
        timing_metrics = {
            "epoch_wall_seconds": epoch_wall_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "avg_data_wait_seconds": counter_totals["data_wait_seconds"] / total_batches,
            "avg_train_step_seconds": counter_totals["train_step_seconds"]
            / total_batches,
            "batches_per_second": total_batches / max(epoch_wall_seconds, 1e-12),
            "samples_per_second": total_samples / max(epoch_wall_seconds, 1e-12),
        }
        rank_zero_logger.info(
            "Epoch timing | "
            f"epoch: {epoch} | "
            f"wall: {timing_metrics['epoch_wall_seconds']:.2f}s | "
            f"data_wait/batch: {timing_metrics['avg_data_wait_seconds']:.3f}s | "
            f"train_step/batch: {timing_metrics['avg_train_step_seconds']:.3f}s | "
            f"checkpoint: {timing_metrics['checkpoint_seconds']:.2f}s | "
            f"batches/s: {timing_metrics['batches_per_second']:.3f} | "
            f"samples/s: {timing_metrics['samples_per_second']:.3f}"
        )
        if dist.rank == 0 and trainer.wandb_enabled and wandb is not None:
            wandb.log({**epoch_metrics, **timing_metrics, "epoch": epoch})

    rank_zero_logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
