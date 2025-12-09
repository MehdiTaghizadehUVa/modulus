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
Training script for FloodForecaster using GINO with domain adaptation.

This script implements a three-stage training pipeline:
1. Pretraining on source domain
2. Domain adaptation on source + target domains
3. Rollout evaluation and metric computation
"""

import os
import sys
from typing import Optional

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from neuralop import get_model
from neuralop.utils import get_wandb_api_key

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.logging.wandb import initialize_wandb

from datasets import (
    FloodDatasetWithQueryPoints,
    FloodRolloutTestDatasetNew,
    NormalizedDataset,
    NormalizedRolloutTestDataset,
)
from data_processing import FloodGINODataProcessor
from training.pretraining import pretrain_model
from training.domain_adaptation import adapt_model
from inference.rollout import rollout_prediction
from utils.normalization import (
    collect_all_fields,
    stack_and_fit_transform,
    transform_with_existing_normalizers,
)


def log_section(logger: RankZeroLoggingWrapper, title: str, char: str = "=", width: int = 60):
    r"""Log a section header for visual separation."""
    separator = char * width
    logger.info("")
    logger.info(separator)
    logger.info(title)
    logger.info(separator)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def train_flood_forecaster(cfg: DictConfig) -> None:
    r"""
    Main training and evaluation pipeline for FloodForecaster.

    This function orchestrates the complete training and evaluation workflow:
    1. Configuration loading and device setup
    2. Pretraining on source domain
    3. Domain adaptation on source + target domains
    4. Rollout evaluation and metric computation

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.

    Raises
    ------
    SystemExit
        If critical errors occur during execution.
    """
    # Initialize distributed manager (must be called first)
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logging
    log = PythonLogger(name="flood_forecaster")
    log_rank_zero = RankZeroLoggingWrapper(log, dist)

    log_section(log_rank_zero, "FLOOD FORECASTER - Training and Evaluation Pipeline")

    config = None
    try:
        # Get device from distributed manager or config
        device = dist.device if dist.device is not None else cfg.distributed.device
        is_logger = dist.rank == 0

        # Log device information prominently
        log_rank_zero.info("=" * 50)
        log_rank_zero.info(f"PyTorch version: {torch.__version__}")
        log_rank_zero.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log_rank_zero.info(f"CUDA version: {torch.version.cuda}")
            log_rank_zero.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            log_rank_zero.info(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        log_rank_zero.info(f"Using device: {device}")
        log_rank_zero.info(f"Distributed: rank={dist.rank}, world_size={dist.world_size}")
        log_rank_zero.info("=" * 50)

        if not torch.cuda.is_available():
            log_rank_zero.warning("CUDA is not available! Training will be very slow on CPU.")
            log_rank_zero.warning("Please check your PyTorch installation with CUDA support.")

        # Adjust FNO modes if needed (access via OmegaConf)
        if (
            hasattr(cfg, "source_data")
            and hasattr(cfg.source_data, "resolution")
            and hasattr(cfg.model, "fno_n_modes")
            and cfg.source_data.resolution < cfg.model.fno_n_modes[0]
        ):
            cfg.model.fno_n_modes = [cfg.source_data.resolution] * len(cfg.model.fno_n_modes)
            log_rank_zero.debug(f"Adjusted FNO modes to: {cfg.model.fno_n_modes}")

        # Initialize wandb if logging is enabled
        if cfg.wandb.log and is_logger:
            log_rank_zero.info("Initializing Weights & Biases logging...")
            wandb.login(key=get_wandb_api_key())
            wandb_name = (
                cfg.wandb.name
                if cfg.wandb.name
                else f"flood-run_{getattr(cfg.source_data, 'resolution', 64)}"
            )
            wandb_init_args = dict(
                config=OmegaConf.to_container(cfg, resolve=True),
                name=wandb_name,
                group=cfg.wandb.group,
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
            )
            if cfg.wandb.sweep:
                for key in wandb.config.keys():
                    if hasattr(cfg, "params"):
                        cfg.params[key] = wandb.config[key]
            wandb.init(**wandb_init_args)
            log_rank_zero.info(f"W&B initialized: project={cfg.wandb.project}, name={wandb_name}")

        # Stage 1: Pretraining on source domain
        log_section(log_rank_zero, "Stage 1: Pretraining on Source Domain")
        model, normalizers, trainer_src = pretrain_model(
            config=cfg,
            device=device,
            is_logger=is_logger,
            source_data_config=cfg.source_data,
            logger=log_rank_zero,
        )

        # Recreate source loaders for domain adaptation
        log_rank_zero.info("Recreating source loaders for domain adaptation...")
        source_full_dataset = FloodDatasetWithQueryPoints(
            data_root=cfg.source_data.root,
            n_history=cfg.source_data.n_history,
            xy_file=cfg.source_data.get("xy_file", None),
            query_res=cfg.source_data.get("query_res", [64, 64]),
            static_files=cfg.source_data.get("static_files", []),
            dynamic_patterns=cfg.source_data.get("dynamic_patterns", {}),
            boundary_patterns=cfg.source_data.get("boundary_patterns", {}),
            raise_on_smaller=True,
            skip_before_timestep=cfg.source_data.get("skip_before_timestep", 0),
            noise_type=cfg.source_data.get("noise_type", "none"),
            noise_std=cfg.source_data.get("noise_std", None),
        )
        train_sz_source = int(0.9 * len(source_full_dataset))
        source_train_raw, source_val_raw = random_split(
            source_full_dataset,
            [train_sz_source, len(source_full_dataset) - train_sz_source],
        )

        # Move normalizers to CPU for data transformation
        for norm in normalizers.values():
            norm.to("cpu")

        geom_s_tr, static_s_tr, boundary_s_tr, dyn_s_tr, tgt_s_tr = collect_all_fields(
            source_train_raw, True
        )
        _, big_source_train = stack_and_fit_transform(
            geom_s_tr,
            static_s_tr,
            boundary_s_tr,
            dyn_s_tr,
            tgt_s_tr,
            normalizers=normalizers,
            fit_normalizers=False,
        )
        source_train_ds = NormalizedDataset(
            geometry=big_source_train["geometry"],
            static=big_source_train["static"],
            boundary=big_source_train["boundary"],
            dynamic=big_source_train["dynamic"],
            target=big_source_train["target"],
            query_res=cfg.source_data.query_res,
        )
        source_train_loader = DataLoader(
            source_train_ds, batch_size=cfg.source_data.batch_size, shuffle=True
        )

        geom_s_val, static_s_val, boundary_s_val, dyn_s_val, tgt_s_val = collect_all_fields(
            source_val_raw, True
        )
        _, big_source_val = stack_and_fit_transform(
            geom_s_val,
            static_s_val,
            boundary_s_val,
            dyn_s_val,
            tgt_s_val,
            normalizers=normalizers,
            fit_normalizers=False,
        )
        source_val_ds = NormalizedDataset(
            geometry=big_source_val["geometry"],
            static=big_source_val["static"],
            boundary=big_source_val["boundary"],
            dynamic=big_source_val["dynamic"],
            target=big_source_val["target"],
            query_res=cfg.source_data.query_res,
        )
        source_val_loader = DataLoader(
            source_val_ds, batch_size=cfg.source_data.batch_size, shuffle=False
        )

        # Stage 2: Domain adaptation
        log_section(log_rank_zero, "Stage 2: Domain Adaptation")
        data_processor = trainer_src.data_processor
        model, domain_classifier, trainer_adapt = adapt_model(
            model=model,
            normalizers=normalizers,
            data_processor=data_processor,
            config=cfg,
            device=device,
            is_logger=is_logger,
            source_train_loader=source_train_loader,
            source_val_loader=source_val_loader,
            target_data_config=cfg.target_data,
            logger=log_rank_zero,
        )

        # Stage 3: Rollout evaluation
        log_section(log_rank_zero, "Stage 3: Rollout Evaluation")
        log_rank_zero.info("Loading rollout test dataset...")
        rollout_test_dataset = FloodRolloutTestDatasetNew(
            rollout_data_root=cfg.rollout_data.root,
            n_history=cfg.source_data.n_history,  # Use source_data for n_history
            rollout_length=cfg.source_data.rollout_length,
            xy_file=cfg.rollout_data.get("xy_file", None),
            query_res=cfg.source_data.get("query_res", [32, 32]),
            static_files=cfg.rollout_data.get("static_files", []),
            dynamic_patterns=cfg.rollout_data.get("dynamic_patterns", {}),
            boundary_patterns=cfg.rollout_data.get("boundary_patterns", {}),
            raise_on_smaller=True,
            skip_before_timestep=cfg.source_data.get("skip_before_timestep", 0),
        )
        log_rank_zero.info(f"Loaded {len(rollout_test_dataset)} rollout test samples")

        # Pass the raw cell area data along with other fields
        (
            rollout_geom,
            rollout_static,
            rollout_boundary,
            rollout_dyn,
            _,
            rollout_cell_area,
        ) = collect_all_fields(rollout_test_dataset, expect_target=False)

        # Move normalizers to CPU for data transformation
        for norm in normalizers.values():
            norm.to("cpu")

        transformed_rollout = transform_with_existing_normalizers(
            rollout_geom, rollout_static, rollout_boundary, rollout_dyn, normalizers
        )

        normalized_rollout_samples = [
            {
                "run_id": rollout_test_dataset.valid_run_ids[i],
                "geometry": transformed_rollout["geometry"][i],
                "static": transformed_rollout["static"][i],
                "boundary": transformed_rollout["boundary"][i],
                "dynamic": transformed_rollout["dynamic"][i],
                "cell_area": rollout_cell_area[i],
            }
            for i in range(len(rollout_test_dataset))
        ]

        log_rank_zero.info("Starting rollout prediction...")
        rollout_prediction(
            model=trainer_adapt.model,
            rollout_dataset=NormalizedRolloutTestDataset(
                normalized_rollout_samples, cfg.source_data.query_res
            ),
            rollout_length=cfg.source_data.rollout_length,
            history_steps=cfg.source_data.n_history,
            dynamic_norm=normalizers["dynamic"],
            target_norm=normalizers["target"],
            boundary_norm=normalizers["boundary"],
            device=device,
            skip_before_timestep=cfg.source_data.get("skip_before_timestep", 0),
            dt=cfg.source_data.dt,
            out_dir=cfg.rollout.out_dir,
            logger=log_rank_zero,
        )

        if cfg.wandb.log and is_logger:
            wandb.finish()
            log_rank_zero.info("W&B logging finished")

        log_section(log_rank_zero, "Training and Evaluation Complete!")

    except KeyboardInterrupt:
        log_rank_zero.warning("Training interrupted by user")
        if config is not None and hasattr(config, "wandb") and config.wandb.log:
            wandb.finish()
        sys.exit(1)
    except Exception as e:
        log_rank_zero.error(f"Fatal error in main pipeline: {e}", exc_info=True)
        if config is not None and hasattr(config, "wandb") and config.wandb.log:
            wandb.finish()
        raise


if __name__ == "__main__":
    train_flood_forecaster()

