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

import time

import hydra
from hydra.utils import to_absolute_path
import torch
import wandb

from dgl.dataloading import GraphDataLoader

from omegaconf import DictConfig

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from modulus.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.logging.wandb import initialize_wandb
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from utils import custom_loss


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        # Ensure the distributed manager is initialized.
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()

        self.amp = cfg.amp
        self.noise_type = cfg.noise_type  # record noise type from config

        # Set the activation function based on configuration.
        mlp_act = "relu"
        if cfg.recompute_activation:
            rank_zero_logger.info("Setting MLP activation to SiLU for recompute_activation.")
            mlp_act = "silu"

        # Initialize dataset and dataloader.
        rank_zero_logger.info("Initializing HydroGraphDataset...")
        dataset = HydroGraphDataset(
            name="hydrograph_dataset",
            data_dir=cfg.data_dir,
            prefix="M80",
            num_samples=500,
            n_time_steps=cfg.n_time_steps,
            k=4,
            noise_type=cfg.noise_type,
            noise_std=0.01,
            hydrograph_ids_file="train.txt",
            split="train",
            force_reload=False,
            verbose=False,
        )
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
        )
        rank_zero_logger.info("Dataset and dataloader initialization complete.")

        # Instantiate the MeshGraphKAN model.
        rank_zero_logger.info("Instantiating MeshGraphKAN model...")
        self.model = MeshGraphKAN(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphKAN is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        rank_zero_logger.info("Model instantiated successfully.")

        if cfg.watch_model and not cfg.jit and self.dist.rank == 0:
            wandb.watch(self.model)

        # Wrap the model for distributed training if necessary.
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

        # Setup loss, optimizer, and learning rate scheduler.
        self.criterion = torch.nn.MSELoss()
        # Initialize optimizer attribute.
        self.optimizer = None
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            rank_zero_logger.warning("NVIDIA Apex is not installed; FusedAdam optimizer will not be used.")
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        rank_zero_logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # Load model checkpoint if available.
        rank_zero_logger.info("Loading checkpoint if available...")
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        rank_zero_logger.info(f"Checkpoint loaded. Starting training from epoch {self.epoch_init}.")

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss, loss_dict = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_dict

    def forward(self, graph):
        if self.noise_type == "pushforward":
            with autocast(enabled=self.amp):
                # Retrieve and parse node features.
                X = graph.ndata["x"]
                n_static = 12  # assume static features have fixed dimension
                n_time = (X.shape[1] - n_static) // 2  # for pushforward, n_time == n_time_steps+1

                static_part = X[:, :n_static]
                water_depth_full = X[:, n_static:n_static+n_time]
                volume_full = X[:, n_static+n_time:n_static+2*n_time]

                # L_one: Use last n_time_steps from the available window (indices 1:).
                water_depth_window_one = water_depth_full[:, 1:]
                volume_window_one = volume_full[:, 1:]
                X_one = torch.cat([static_part, water_depth_window_one, volume_window_one], dim=1)
                pred_one = self.model(X_one, graph.edata["x"], graph)
                one_step_loss = self.criterion(pred_one, graph.ndata["y"])

                # L_stability: Use first n_time_steps-1 (indices :n_time-1) to predict t-1.
                water_depth_window_stab = water_depth_full[:, :n_time-1]
                volume_window_stab = volume_full[:, :n_time-1]
                X_stab = torch.cat([static_part, water_depth_window_stab, volume_window_stab], dim=1)
                pred_stab = self.model(X_stab, graph.edata["x"], graph)
                pred_stab_detached = pred_stab.detach()

                # Update dynamic window: ground truth for t-2 and predicted (updated) for t-1.
                water_depth_updated = torch.cat(
                    [water_depth_full[:, 1:2], water_depth_full[:, 1:2] + pred_stab_detached[:, 0:1]],
                    dim=1
                )
                volume_updated = torch.cat(
                    [volume_full[:, 1:2], volume_full[:, 1:2] + pred_stab_detached[:, 1:2]],
                    dim=1
                )
                X_stab_updated = torch.cat([static_part, water_depth_updated, volume_updated], dim=1)
                pred_stab2 = self.model(X_stab_updated, graph.edata["x"], graph)
                stability_loss = self.criterion(pred_stab2, graph.ndata["y"])

                total_loss = one_step_loss + stability_loss
                loss_dict = {
                    "total_loss": total_loss,
                    "loss_one": one_step_loss,
                    "loss_stability": stability_loss
                }
                return total_loss, loss_dict
        else:
            with autocast(enabled=self.amp):
                pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
                loss_dict = custom_loss(pred, graph.ndata["y"])
                loss = loss_dict['total_loss']
                return loss, loss_dict

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Initialize distributed training manager.
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize WandB logging.
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="Vortex_Shedding-Training",
        group="Vortex_Shedding-DDP-Group",
        mode=cfg.wandb_mode,
    )

    # Setup primary Python logger and wrap for rank zero.
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()
    rank_zero_logger.info(f"Starting training process with configuration: {cfg}")

    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Beginning training loop...")
    start_time = time.time()

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        num_batches = 0
        for graph in trainer.dataloader:
            loss, loss_dict = trainer.train(graph)
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        rank_zero_logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4e}")

        # Log loss details to WandB.
        if "loss_one" in loss_dict:
            wandb.log({
                "total_loss": loss_dict["total_loss"].detach().cpu(),
                "loss_one": loss_dict["loss_one"].detach().cpu(),
                "loss_stability": loss_dict["loss_stability"].detach().cpu(),
                "epoch": epoch
            })
        else:
            wandb.log({
                "total_loss": loss_dict["total_loss"].detach().cpu(),
                "loss_depth": loss_dict["loss_depth"].detach().cpu(),
                "loss_volume": loss_dict["loss_volume"].detach().cpu(),
                "epoch": epoch
            })

        # Save checkpoint (rank 0 only).
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Checkpoint saved at epoch {epoch}.")

        elapsed = time.time() - start_time
        rank_zero_logger.info(f"Epoch {epoch} duration: {elapsed:.2f} seconds.")
        start_time = time.time()

    rank_zero_logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
