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
Domain adaptation module for fine-tuning on target domain.

This module implements adversarial domain adaptation using gradient reversal layers
to enable transfer learning from source to target domains. Compatible with neuralop 2.0.0 API
and physicsnemo framework.

Key components:
- GradientReversal: Implements gradient reversal layer for adversarial training
- CNNDomainClassifier: CNN-based domain classifier for domain discrimination
- DomainAdaptationTrainer: Custom trainer for domain adaptation training loop
- adapt_model: High-level function to perform domain adaptation
"""

import os
import sys
import math
import random
from contextlib import nullcontext
from timeit import default_timer
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple, Any
from itertools import cycle

import torch
import torch.nn as nn
import torch.distributed as torch_dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralop.training import AdamW
from neuralop.losses import LpLoss

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint


def _supports_tqdm_output(stream: Any) -> bool:
    r"""Return True when tqdm can safely render to the given stream."""
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except OSError:
            return False
    return False


def _safe_tqdm_postfix(progress_bar: Any, values: Dict[str, str]) -> None:
    r"""Best-effort tqdm postfix update for non-interactive or captured consoles."""
    if not hasattr(progress_bar, "set_postfix"):
        return
    try:
        progress_bar.set_postfix(values)
    except OSError:
        disable = getattr(progress_bar, "disable", None)
        if disable is not None:
            progress_bar.disable = True
    except ValueError:
        disable = getattr(progress_bar, "disable", None)
        if disable is not None:
            progress_bar.disable = True


from data_processing import LpLossWrapper
from models import CNNDomainClassifier

# Try to import comm for distributed training, fallback if not available
try:
    import neuralop.mpu.comm as comm
    _has_comm = True
except ImportError:
    _has_comm = False
    # Fallback: create a dummy comm module
    class _DummyComm:
        @staticmethod
        def get_local_rank():
            return 0
    comm = _DummyComm()

from datasets import FloodDatasetWithQueryPoints, LazyNormalizedDataset
from utils.checkpointing import (
    resolve_checkpoint_epoch,
    resolve_legacy_neuralop_checkpoint_name,
    validate_checkpoint_files,
    write_best_checkpoint_metadata,
)
from utils.runtime import (
    create_loader_from_config,
    resolve_amp_autocast_enabled,
    resolve_eval_interval,
    set_loader_epoch,
    split_dataset,
)
from training.pretraining import create_scheduler


class DomainAdaptationTrainer:
    r"""
    Custom trainer for domain adaptation compatible with neuralop 2.0.0.
    
    Implements adversarial domain adaptation using gradient reversal layers (GRL).
    The training process alternates between:
    1. Task loss: Regression loss on both source and target domains
    2. Adversarial loss: Domain classification loss with reversed gradients
    
    The GRL lambda is scheduled during training to gradually increase the strength
    of adversarial training.
    
    Parameters
    ----------
    model : nn.Module
        The main model to train (should support return_features=True).
    data_processor : nn.Module, optional
        Data processor for preprocessing/postprocessing.
    domain_classifier : nn.Module
        Domain classifier module.
    device : str or torch.device, optional, default="cuda"
        Device to train on ('cuda', 'cpu', or torch.device).
    verbose : bool, optional, default=True
        Whether to print training progress.
    logger : Any, optional
        Optional logger instance (if None, uses print when verbose=True).
    
    Attributes
    ----------
    model : nn.Module
        The main GINO model (wrapped with GINOWrapper).
    data_processor : nn.Module, optional
        Data processor for preprocessing/postprocessing.
    domain_classifier : nn.Module
        CNN-based domain classifier.
    device : str or torch.device
        Device to train on.
    verbose : bool
        Whether to print training progress.
    _eval_interval : int
        Interval for evaluation (default: 1).
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_processor: Optional[nn.Module],
        domain_classifier: nn.Module,
        device: Union[str, torch.device] = "cuda",
        mixed_precision: bool = False,
        eval_interval: int = 1,
        verbose: bool = True,
        logger: Optional[Any] = None,
        wandb_step_offset: int = 0,
    ):
        r"""
        Initialize domain adaptation trainer.

        Parameters
        ----------
        model : nn.Module
            The main model to train (should support return_features=True).
        data_processor : nn.Module, optional
            Optional data processor for preprocessing.
        domain_classifier : nn.Module
            Domain classifier module.
        device : str or torch.device, optional, default="cuda"
            Device to train on ('cuda', 'cpu', or torch.device).
        verbose : bool, optional, default=True
            Whether to print training progress.
        logger : Any, optional
            Optional logger instance (if None, uses print when verbose=True).
        wandb_step_offset : int, optional, default=0
            Step offset for wandb logging to continue from pretraining step count.
        """
        self.model = model
        self.data_processor = data_processor
        self.domain_classifier = domain_classifier
        self.device = device
        self.mixed_precision = mixed_precision
        self.verbose = verbose
        self.logger = logger
        self.wandb_step_offset = wandb_step_offset
        self.eval_interval = eval_interval
        self.checkpoint_stage = "adapt"
        self.best_metric_name: Optional[str] = None
        self.best_metric_value = float("inf")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=self.mixed_precision and torch.device(device).type == "cuda",
            )
        else:
            self.scaler = GradScaler(enabled=self.mixed_precision and torch.device(device).type == "cuda")

    def _distributed_active(self) -> bool:
        return (
            DistributedManager.is_initialized()
            and DistributedManager().distributed
            and torch_dist.is_available()
            and torch_dist.is_initialized()
        )

    def _all_reduce_scalar(self, value: Union[int, float]) -> float:
        if not self._distributed_active():
            return float(value)
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        torch_dist.all_reduce(tensor, op=torch_dist.ReduceOp.SUM)
        return float(tensor.item())

    def _autocast_context(self):
        if not self.mixed_precision:
            return nullcontext()
        if hasattr(torch.amp, "autocast") and torch.device(self.device).type == "cuda":
            return torch.amp.autocast(
                device_type=torch.device(self.device).type,
                enabled=self.mixed_precision,
            )
        return nullcontext()

    @staticmethod
    def _model_forward_kwargs(sample: Dict[str, Any]) -> Dict[str, Any]:
        r"""Drop loss-only and metadata keys before forwarding samples to the model."""
        loss_only_keys = {"y", "target", "run_id", "time_index", "cell_area"}
        return {key: value for key, value in sample.items() if key not in loss_only_keys}

    def _wrap_for_ddp(self, module: nn.Module) -> nn.Module:
        if not DistributedManager.is_initialized():
            return module
        dist_manager = DistributedManager()
        if not dist_manager.distributed or isinstance(module, torch.nn.parallel.DistributedDataParallel):
            return module

        ddp_kwargs = {
            "broadcast_buffers": dist_manager.broadcast_buffers,
            "find_unused_parameters": dist_manager.find_unused_parameters,
        }
        if torch.device(self.device).type == "cuda":
            ddp_kwargs["device_ids"] = [dist_manager.local_rank]
            ddp_kwargs["output_device"] = dist_manager.local_rank
        return torch.nn.parallel.DistributedDataParallel(module, **ddp_kwargs)
        
    def train_domain_adaptation(
        self,
        src_loader: DataLoader,
        tgt_loader: Union[DataLoader, List[DataLoader]],
        optimizer,
        scheduler,
        training_loss,
        class_loss_weight: float = 0.1,
        adaptation_epochs: int = 100,
        save_every: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
        resume_classifier_from_dir: Union[str, Path] = None,
        val_loaders: Optional[Dict[str, DataLoader]] = None,
    ):
        r"""
        Domain-adaptation training loop with adversarial classifier.
        
        Handles both a single target DataLoader and a list of target DataLoaders.
        This implementation exactly matches the original neuralop trainer's
        train_domain_adaptation method.

        Parameters
        ----------
        src_loader : DataLoader
            Source domain dataloader.
        tgt_loader : DataLoader or List[DataLoader]
            Target domain dataloader (single or list).
        optimizer : torch.optim.Optimizer
            Optimizer for both model and classifier.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler.
        training_loss : callable
            Loss function for main task.
        class_loss_weight : float, optional, default=0.1
            Weight for domain classification loss.
        adaptation_epochs : int, optional, default=100
            Number of epochs to train.
        save_every : int, optional
            Interval at which to save checkpoints.
        save_dir : str or Path, optional, default="./ckpt"
            Directory to save checkpoints.
        resume_from_dir : str or Path, optional
            Directory to resume training from.
        resume_classifier_from_dir : str or Path, optional
            Directory to resume classifier from.
        val_loaders : Dict[str, DataLoader], optional
            Dict of validation dataloaders.

        Returns
        -------
        nn.Module
            Trained model.
        """
        self.model = self.model.to(self.device)
        self.domain_classifier = self.domain_classifier.to(self.device)
        self.model = self._wrap_for_ddp(self.model)
        self.domain_classifier = self._wrap_for_ddp(self.domain_classifier)
        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)

        adv_criterion = nn.BCEWithLogitsLoss()
        lambda_max = float(getattr(self.domain_classifier.module if isinstance(self.domain_classifier, torch.nn.parallel.DistributedDataParallel) else self.domain_classifier, "grl").lambda_)

        start_epoch = -1
        if resume_from_dir is not None:
            start_epoch = self._resume_from_checkpoint(resume_from_dir, optimizer, scheduler)
        
        # Optionally resume classifier from separate directory (fallback mechanism)
        if resume_classifier_from_dir is not None:
            classifier_loaded = False
            resume_classifier_dir = Path(resume_classifier_from_dir)
            
            # Try PhysicsNeMo format first (classifier saved as second model)
            try:
                resolved_epoch = resolve_checkpoint_epoch(resume_classifier_dir, "latest")
                validate_checkpoint_files(
                    resume_classifier_dir,
                    [self.domain_classifier],
                    resolved_epoch,
                    require_training_state=False,
                )
                metadata_dict = {}
                load_checkpoint(
                    path=str(resume_classifier_dir),
                    models=[self.domain_classifier],
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    epoch=resolved_epoch,
                    metadata_dict=metadata_dict,
                    device=self.device,
                )
                msg = f"Loaded classifier from PhysicsNeMo checkpoint: {resume_classifier_dir}"
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
                classifier_loaded = True
            except (FileNotFoundError, KeyError, ValueError):
                # Fall back to old format (separate classifier_state_dict.pt file)
                ckpt = resume_classifier_dir / "classifier_state_dict.pt"
                if ckpt.exists():
                    self.domain_classifier.load_state_dict(
                        torch.load(str(ckpt), map_location=self.device)
                    )
                    msg = f"Loaded classifier from neuralop checkpoint: {ckpt}"
                    if self.logger:
                        self.logger.info(msg)
                    elif self.verbose:
                        print(msg)
                    classifier_loaded = True
            
            if not classifier_loaded:
                msg = f"Warning: Could not load classifier from {resume_classifier_from_dir}"
                if self.logger:
                    self.logger.warning(msg)
                elif self.verbose:
                    print(msg)
        
        val_loaders = val_loaders or {}
        if "target_val" in val_loaders:
            self.best_metric_name = "target_val"
        elif val_loaders:
            self.best_metric_name = next(iter(val_loaders))
        
        # Handle both single loader and list of loaders
        if not isinstance(tgt_loader, list):
            tgt_loaders = [tgt_loader]
        else:
            tgt_loaders = tgt_loader
        
        # Determine iteration strategy based on source loader
        base_batches = len(src_loader)
        total_iters = adaptation_epochs * base_batches
        
        # Create cycling iterators for all loaders
        src_iter = cycle(src_loader)
        tgt_iters = [cycle(loader) for loader in tgt_loaders]
        
        msg1 = f"Starting domain adaptation training for {adaptation_epochs} epochs"
        msg2 = f"Source samples: {len(src_loader.dataset)}, Target loaders: {len(tgt_loaders)}"
        if self.logger:
            self.logger.info(msg1)
            self.logger.info(msg2)
        elif self.verbose:
            print(msg1)
            print(msg2)
        
        for epoch in range(start_epoch + 1, adaptation_epochs):
            set_loader_epoch(src_loader, epoch)
            for loader in tgt_loaders:
                set_loader_epoch(loader, epoch)
            for loader in val_loaders.values():
                set_loader_epoch(loader, epoch)

            self.on_epoch_start(epoch)
            self.model.train()
            self.domain_classifier.train()
            if self.data_processor is not None:
                self.data_processor.train()
            
            total_reg, total_adv = 0.0, 0.0
            n_batches = 0
             
            # Progress bar
            pbar = tqdm(
                range(base_batches),
                desc=f"DA Epoch {epoch}/{adaptation_epochs}",
                disable=not (self.verbose and _supports_tqdm_output(sys.stdout)),
                file=sys.stdout
            )
             
            for batch_idx in pbar:
                grl_owner = (
                    self.domain_classifier.module
                    if isinstance(self.domain_classifier, torch.nn.parallel.DistributedDataParallel)
                    else self.domain_classifier
                )
                if class_loss_weight > 0.0:
                    progress = (epoch * base_batches + batch_idx) / max(total_iters - 1, 1)
                    lambda_val = lambda_max * (2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)
                else:
                    lambda_val = 0.0
                grl_owner.grl.set_lambda(lambda_val)
                 
                # Randomly select one target domain from the list for this training step
                chosen_tgt_iter = random.choice(tgt_iters)
                 
                src_batch = next(src_iter)
                tgt_batch = next(chosen_tgt_iter)
                
                # Preprocess batches
                if self.data_processor is not None:
                    s = self.data_processor.preprocess(src_batch)
                    t = self.data_processor.preprocess(tgt_batch)
                else:
                    s = {k: v.to(self.device) for k, v in src_batch.items() if torch.is_tensor(v)}
                    t = {k: v.to(self.device) for k, v in tgt_batch.items() if torch.is_tensor(v)}
                s_model_kwargs = self._model_forward_kwargs(s)
                t_model_kwargs = self._model_forward_kwargs(t)
                
                optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    if class_loss_weight > 0.0:
                        try:
                            out_s, f_s = self.model(**s_model_kwargs, return_features=True)
                            out_t, f_t = self.model(**t_model_kwargs, return_features=True)
                        except TypeError as e:
                            raise RuntimeError(
                                "Model must support return_features=True for domain adaptation. "
                                "Ensure model is wrapped with GINOWrapper."
                            ) from e
                    else:
                        out_s = self.model(**s_model_kwargs)
                        out_t = self.model(**t_model_kwargs)
                        f_s = None
                        f_t = None

                    if self.data_processor is not None:
                        out_s, s = self.data_processor.postprocess(out_s, s)
                        out_t, t = self.data_processor.postprocess(out_t, t)

                    reg_loss = training_loss(out_s, **s) + training_loss(out_t, **t)

                    if class_loss_weight > 0.0:
                        if f_s.dim() != 4 or f_t.dim() != 4:
                            raise ValueError(
                                f"Expected 4D features (B, C, H, W), got f_s.shape={f_s.shape}, f_t.shape={f_t.shape}. "
                                "Ensure GINOWrapper returns features in correct format."
                            )
                        feats = torch.cat([f_s, f_t], dim=0)
                        logits = self.domain_classifier(feats).squeeze(1)
                        labels = torch.cat(
                            [
                                torch.ones(f_s.size(0), device=self.device),
                                torch.zeros(f_t.size(0), device=self.device),
                            ],
                            dim=0,
                        ).float()
                        adv_loss = adv_criterion(logits, labels)
                        loss = reg_loss + class_loss_weight * adv_loss
                    else:
                        adv_loss = torch.zeros((), device=self.device)
                        loss = reg_loss

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_reg += reg_loss.item()
                total_adv += adv_loss.item()
                n_batches += 1

                # Update progress bar
                if self.verbose:
                    _safe_tqdm_postfix(pbar, {
                        'loss': f'{loss.item():.4f}',
                        'reg': f'{reg_loss.item():.4f}',
                        'adv': f'{adv_loss.item():.4f}',
                        'lambda': f'{lambda_val:.3f}'
                    })
            
            total_reg = self._all_reduce_scalar(total_reg)
            total_adv = self._all_reduce_scalar(total_adv)
            n_batches = self._all_reduce_scalar(n_batches)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_reg + total_adv)
            else:
                scheduler.step()
            avg_reg = total_reg / n_batches if n_batches > 0 else 0.0
            avg_adv = total_adv / n_batches if n_batches > 0 else 0.0
            msg = f"[DA Epoch {epoch}] reg={avg_reg:.4f}, adv={avg_adv:.4f}, lambda={lambda_val:.3f}"
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)

            # Validation (if val_loaders provided)
            val_metrics = {}
            if val_loaders and (epoch % self.eval_interval == 0 or epoch == adaptation_epochs - 1):
                val_metrics = self._evaluate(val_loaders, training_loss, epoch)

            if self.best_metric_name and self.best_metric_name in val_metrics:
                best_value = val_metrics[self.best_metric_name]
                if best_value < self.best_metric_value:
                    self.best_metric_value = best_value
                    self._save_checkpoint(
                        save_dir,
                        optimizer,
                        scheduler,
                        epoch,
                        save_classifier=True,
                        is_best=True,
                        metric_name=self.best_metric_name,
                    )

            should_save_latest = save_every is None or epoch % save_every == 0
            if should_save_latest:
                self._save_checkpoint(
                    save_dir,
                    optimizer,
                    scheduler,
                    epoch,
                    save_classifier=True,
                    is_best=False,
                    metric_name=self.best_metric_name,
                )

        msg = "Domain adaptation training completed!"
        if self.logger:
            self.logger.info(msg)
        elif self.verbose:
            print(msg)
        return self.model
    
    def on_epoch_start(self, epoch):
        r"""Stub called at the start of each epoch."""
        self.epoch = epoch
        return None
    
    @property
    def eval_interval(self):
        r"""Evaluation interval (default 1)."""
        return getattr(self, '_eval_interval', 1)
    
    @eval_interval.setter
    def eval_interval(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("eval_interval must be a positive integer.")
        self._eval_interval = value
    
    def _evaluate(
        self, 
        val_loaders: Dict[str, DataLoader], 
        loss_fn: Any, 
        epoch: int
    ) -> Dict[str, float]:
        r"""
        Evaluate model on validation loaders.

        Parameters
        ----------
        val_loaders : Dict[str, DataLoader]
            Dictionary of validation dataloaders.
        loss_fn : callable
            Loss function to use for evaluation.
        epoch : int
            Current epoch number (for logging).
        """
        self.model.eval()
        if self.data_processor is not None:
            self.data_processor.eval()
        
        metrics: Dict[str, float] = {}
        with torch.no_grad():
            for name, loader in val_loaders.items():
                total_loss = 0.0
                n_samples = 0

                for sample in loader:
                    try:
                        if self.data_processor is not None:
                            sample = self.data_processor.preprocess(sample)
                        else:
                            sample = {
                                k: v.to(self.device)
                                for k, v in sample.items()
                                if torch.is_tensor(v)
                            }
                        model_kwargs = self._model_forward_kwargs(sample)

                        with self._autocast_context():
                            out = self.model(**model_kwargs)

                            if self.data_processor is not None:
                                out, sample = self.data_processor.postprocess(out, sample)

                            loss = loss_fn(out, **sample)
                        total_loss += loss.item()
                        if isinstance(sample.get("y"), torch.Tensor):
                            n_samples += sample["y"].shape[0]
                        else:
                            n_samples += out.shape[0]
                    except Exception as e:
                        msg = f"Error evaluating on {name}: {e}"
                        if self.logger:
                            self.logger.error(msg)
                        elif self.verbose:
                            print(msg)
                        raise

                total_loss = self._all_reduce_scalar(total_loss)
                n_samples = self._all_reduce_scalar(n_samples)
                avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
                metrics[name] = avg_loss
                msg = f"  Eval {name}: loss={avg_loss:.6f}"
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
        return metrics
    
    def _save_checkpoint(
        self, 
        save_dir: Union[str, Path], 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any, 
        epoch: int, 
        save_classifier: bool = False,
        is_best: bool = False,
        metric_name: Optional[str] = None,
    ) -> None:
        r"""
        Save training checkpoint using PhysicsNeMo checkpoint system.
        
        This method saves both the main model and domain classifier (if requested)
        using PhysicsNeMo's checkpoint format.

        Parameters
        ----------
        save_dir : str or Path
            Directory to save checkpoint.
        optimizer : torch.optim.Optimizer
            Optimizer instance.
        scheduler : Any
            Scheduler instance.
        epoch : int
            Current epoch number.
        save_classifier : bool, optional, default=False
            Whether to save classifier state dict.
        """
        if save_dir is None:
            return
        save_dir = Path(save_dir)
        
        # Only save on rank 0 in distributed training
        should_save = True
        if DistributedManager.is_initialized():
            dist_manager = DistributedManager()
            should_save = (dist_manager.rank == 0)
        elif _has_comm:
            should_save = (comm.get_local_rank() == 0)
        
        if not should_save:
            return
        
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            model_to_save = self.model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_to_save = self.model.module

            models_to_save = [model_to_save]
            if save_classifier:
                models_to_save.append(self.domain_classifier)
            
            # Save checkpoint with models and training state
            save_checkpoint(
                path=str(save_dir),
                models=models_to_save if models_to_save else None,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=self.scaler if self.scaler.is_enabled() else None,
                epoch=epoch,
                metadata={
                    "stage": self.checkpoint_stage,
                    "epoch": epoch,
                    "is_best": is_best,
                    "best_metric_value": self.best_metric_value if is_best else None,
                },
            )

            if is_best and metric_name is not None:
                write_best_checkpoint_metadata(
                    save_dir,
                    stage=self.checkpoint_stage,
                    epoch=epoch,
                    metric_name=metric_name,
                    metric_value=self.best_metric_value,
                    models=models_to_save,
                )
            
            msg = f"Saved checkpoint to {save_dir}"
            if self.logger:
                self.logger.info(msg)
            elif self.verbose:
                print(msg)
        except Exception as e:
            msg = f"Error saving checkpoint to {save_dir}: {e}"
            if self.logger:
                self.logger.error(msg)
            elif self.verbose:
                print(msg)
            raise
    
    def _resume_from_checkpoint(
        self, 
        resume_dir: Union[str, Path], 
        optimizer: torch.optim.Optimizer, 
        scheduler: Any
    ) -> int:
        r"""
        Resume training from checkpoint using PhysicsNeMo checkpoint system.
        
        Supports both PhysicsNeMo format (new) and neuralop format (old) for backward compatibility.

        Parameters
        ----------
        resume_dir : str or Path
            Directory containing checkpoint.
        optimizer : torch.optim.Optimizer
            Optimizer instance (will be updated with checkpoint state).
        scheduler : Any
            Scheduler instance (will be updated with checkpoint state).

        Returns
        -------
        int
            Epoch number to resume from (-1 if no checkpoint found).
        """
        resume_dir = Path(resume_dir)
        
        if not resume_dir.exists():
            msg = f"Resume directory does not exist: {resume_dir}"
            if self.logger:
                self.logger.warning(msg)
            elif self.verbose:
                print(msg)
            return -1

        try:
            metadata_dict = {}

            try:
                models_to_load = [self.model]
                if hasattr(self, 'domain_classifier') and self.domain_classifier is not None:
                    models_to_load.append(self.domain_classifier)

                resolved_epoch = resolve_checkpoint_epoch(resume_dir, "latest")
                validate_checkpoint_files(resume_dir, models_to_load, resolved_epoch)
                load_checkpoint(
                    path=str(resume_dir),
                    models=models_to_load,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=self.scaler if self.scaler.is_enabled() else None,
                    epoch=resolved_epoch,
                    metadata_dict=metadata_dict,
                    device=self.device,
                )
                if self.logger:
                    self.logger.info(f"Loaded checkpoint using PhysicsNeMo format (epoch={resolved_epoch})")
                resume_epoch = resolved_epoch
            except (FileNotFoundError, KeyError, ValueError) as e:
                if self.logger:
                    self.logger.info(f"PhysicsNeMo checkpoint not found, trying neuralop format: {e}")

                save_name = resolve_legacy_neuralop_checkpoint_name(resume_dir, "latest")
                if save_name is None:
                    msg = f"No checkpoint found in {resume_dir} (tried both formats)"
                    if self.logger:
                        self.logger.warning(msg)
                    elif self.verbose:
                        print(msg)
                    return -1

                from neuralop.training.training_state import load_training_state

                model_for_load = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
                if hasattr(model_for_load, "gino"):
                    model_for_load = model_for_load.gino
                    if hasattr(model_for_load, "inner_model"):
                        model_for_load = model_for_load.inner_model

                _, optimizer, scheduler, _, resume_epoch = load_training_state(
                    save_dir=resume_dir,
                    save_name=save_name,
                    model=model_for_load,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                # Try to load domain classifier if it exists
                classifier_path = resume_dir / "classifier_state_dict.pt"
                if classifier_path.exists() and hasattr(self, 'domain_classifier') and self.domain_classifier is not None:
                    classifier_target = (
                        self.domain_classifier.module
                        if isinstance(self.domain_classifier, torch.nn.parallel.DistributedDataParallel)
                        else self.domain_classifier
                    )
                    classifier_target.load_state_dict(
                        torch.load(str(classifier_path), map_location=self.device)
                    )
                    if self.logger:
                        self.logger.info("Loaded domain classifier from neuralop checkpoint")

                if self.logger:
                    self.logger.info(f"Loaded checkpoint using neuralop format: {save_name}")

            if resume_epoch is not None and resume_epoch >= 0:
                msg = f"Resumed from epoch {resume_epoch}"
                if self.logger:
                    self.logger.info(msg)
                elif self.verbose:
                    print(msg)
                return resume_epoch
            return -1
                
        except Exception as e:
            msg = f"Error loading checkpoint from {resume_dir}: {e}"
            if self.logger:
                self.logger.error(msg)
            elif self.verbose:
                print(msg)
            raise


def adapt_model(
    model: nn.Module,
    normalizers: Dict[str, Any],
    data_processor: Optional[nn.Module],
    config: Any,
    device: Union[str, torch.device],
    is_logger: bool,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_data_config: Any,
    logger: Optional[Any] = None,
    wandb_step_offset: int = 0,
) -> Tuple[nn.Module, nn.Module, "DomainAdaptationTrainer"]:
    r"""
    Perform domain adaptation on target domain data.

    This function orchestrates the domain adaptation training process:
    1. Loads and normalizes target domain data
    2. Creates domain classifier
    3. Sets up optimizer and scheduler
    4. Trains model with adversarial domain adaptation

    Parameters
    ----------
    model : nn.Module
        Pretrained model (should be wrapped with GINOWrapper).
    normalizers : Dict[str, Any]
        Dictionary of normalizers from pretraining.
    data_processor : nn.Module, optional
        Data processor instance.
    config : Any
        Configuration object (OmegaConf DictConfig).
    device : str or torch.device
        Device to train on ('cuda', 'cpu', or torch.device).
    is_logger : bool
        Whether this process is the logger (for distributed training).
    source_train_loader : DataLoader
        Source domain training dataloader.
    source_val_loader : DataLoader
        Source domain validation dataloader.
    target_data_config : Any
        Target data configuration (OmegaConf DictConfig).
    logger : Any, optional
        Optional logger instance (physicsnemo PythonLogger or compatible).
    wandb_step_offset : int, optional, default=0
        Step offset for wandb logging to continue from pretraining step count.
        Currently not used but reserved for future wandb integration.

    Returns
    -------
    Tuple[nn.Module, nn.Module, DomainAdaptationTrainer]
        Tuple of (adapted_model, domain_classifier, trainer).

    Raises
    ------
    ValueError
        If required config keys are missing.
    RuntimeError
        If model doesn't support return_features.
    """
    if logger is None:
        # Fallback to print if no logger provided (for backward compatibility)
        def log_info(msg: str) -> None:
            print(msg)

        logger = type("Logger", (), {"info": lambda self, msg: log_info(msg)})()

    logger.info("Starting domain adaptation on source + target...")
    data_io_cfg = getattr(config, "data_io", {})
    
    # Validate inputs
    if not hasattr(model, 'fno_hidden_channels'):
        raise AttributeError(
            "Model must have 'fno_hidden_channels' attribute. "
            "Ensure model is a GINO model or wrapped with GINOWrapper."
        )
    
    # Create target dataset
    logger.info(f"Loading target dataset from: {target_data_config.root}")
    try:
        target_full_dataset = FloodDatasetWithQueryPoints(
        data_root=target_data_config.root,
        n_history=target_data_config.n_history,
        xy_file=getattr(target_data_config, "xy_file", None),
        query_res=getattr(target_data_config, "query_res", [64, 64]),
        static_files=getattr(target_data_config, "static_files", []),
        dynamic_patterns=getattr(target_data_config, "dynamic_patterns", {}),
        boundary_patterns=getattr(target_data_config, "boundary_patterns", {}),
        raise_on_smaller=True,
        skip_before_timestep=getattr(target_data_config, "skip_before_timestep", 0),
        noise_type=getattr(target_data_config, "noise_type", "none"),
        noise_std=getattr(target_data_config, "noise_std", None),
        backend=getattr(data_io_cfg, "backend", "auto"),
        cache_dir_name=getattr(data_io_cfg, "cache_dir_name", ".flood_cache"),
        rebuild_cache=bool(getattr(data_io_cfg, "rebuild_cache", False)),
        run_cache_size=int(getattr(data_io_cfg, "run_cache_size", 4)),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load target dataset from {target_data_config.root}: {e}") from e
    
    # Split into train/val
    train_sz_target = int(0.9 * len(target_full_dataset))
    target_train_raw, target_val_raw = split_dataset(
        target_full_dataset,
        [train_sz_target, len(target_full_dataset) - train_sz_target],
        seed=config.distributed.seed,
        offset=23,
    )
    logger.info(f"Target domain: total={len(target_full_dataset)}, train={train_sz_target}, val={len(target_val_raw)}")
    
    # Move normalizers to CPU temporarily
    for nm in normalizers.values():
        nm.to('cpu')
    
    logger.info("Preparing lazily normalized target training data...")
    target_train_ds = LazyNormalizedDataset(
        base_dataset=target_train_raw,
        normalizers=normalizers,
        query_res=target_data_config.query_res,
        apply_noise=True,
    )
    
    logger.info("Preparing lazily normalized target validation data...")
    target_val_ds = LazyNormalizedDataset(
        base_dataset=target_val_raw,
        normalizers=normalizers,
        query_res=target_data_config.query_res,
        apply_noise=False,
    )
    target_val_loader = create_loader_from_config(
        target_val_ds,
        target_data_config,
        shuffle=False,
    )
    
    # Create domain classifier
    logger.info("Creating domain classifier...")
    try:
        da_cfg = config.training.get("da_classifier", {})
        if not da_cfg:
            raise ValueError("config.training.da_classifier is required for domain adaptation")
        domain_classifier = CNNDomainClassifier(
            model.fno_hidden_channels,
            config.training.get("da_lambda_max", 1.0),
            da_cfg,
        ).to(device)
    except (AttributeError, KeyError) as e:
        raise ValueError(
            f"Invalid domain adaptation configuration: {e}. "
            "Ensure config.training.da_classifier contains 'conv_layers' and 'fc_dim'."
        ) from e
    
    # Create optimizer and scheduler
    adapt_lr = config.training.get("adapt_learning_rate", config.training.get("learning_rate", 1e-4))
    weight_decay = config.training.get("weight_decay", 1e-4)
    optimizer_adapt = AdamW(
        list(model.parameters()) + list(domain_classifier.parameters()),
        lr=adapt_lr,
        weight_decay=weight_decay,
    )
    logger.info(f"Optimizer: AdamW (lr={adapt_lr}, weight_decay={weight_decay})")
    scheduler_adapt = create_scheduler(optimizer_adapt, config, logger)

    # Create loss - wrap with LpLossWrapper to filter out unexpected kwargs
    # Get loss type from config, default to 'l2'
    def create_loss(loss_type_str, default="l2"):
        """Helper function to create loss function from string."""
        loss_type_str = loss_type_str.lower()
        if loss_type_str == "l1":
            return LpLossWrapper(LpLoss(d=2, p=1)), "l1"
        elif loss_type_str == "l2":
            return LpLossWrapper(LpLoss(d=2, p=2)), "l2"
        else:
            if logger:
                logger.warning(f"Unknown loss type '{loss_type_str}', defaulting to '{default}'")
            return LpLossWrapper(LpLoss(d=2, p=2)), default
    
    training_loss_type = config.training.get("training_loss", "l2")
    training_loss_fn, training_loss_name = create_loss(training_loss_type)
    if logger:
        logger.info(f"Using {training_loss_name.upper()} loss for domain adaptation training")
    
    # Use testing_loss for evaluation if specified, otherwise use training_loss
    # Note: Currently domain adaptation uses the same loss for both training and evaluation
    # but we create it here for consistency and future extensibility
    testing_loss_type = config.training.get("testing_loss", training_loss_type)
    eval_loss_fn, eval_loss_name = create_loss(testing_loss_type, default=training_loss_name)
    if testing_loss_type.lower() != training_loss_type.lower() and logger:
        logger.info(f"Note: testing_loss specified but domain adaptation currently uses training_loss for evaluation")

    # Create custom domain adaptation trainer
    spatial_shape = getattr(target_data_config, "query_res", None)
    if spatial_shape is None:
        resolution = getattr(target_data_config, "resolution", None)
        spatial_shape = [resolution, resolution] if resolution is not None else None
    mixed_precision_enabled = resolve_amp_autocast_enabled(
        config.training.get("amp_autocast", False),
        device=device,
        spatial_shape=spatial_shape,
        logger=logger if hasattr(logger, "warning") else None,
        context="FloodForecaster domain adaptation",
    )

    trainer_adapt = DomainAdaptationTrainer(
        model=model,
        data_processor=data_processor,
        domain_classifier=domain_classifier,
        device=device,
        mixed_precision=mixed_precision_enabled,
        eval_interval=resolve_eval_interval(config),
        verbose=is_logger,
        logger=logger,
        wandb_step_offset=wandb_step_offset,
    )

    # Train with domain adaptation
    save_dir = os.path.join(config.checkpoint.get("save_dir", "./checkpoints"), "adapt")
    logger.info(f"Starting training... Checkpoints will be saved to: {save_dir}")
    logger.info(f"Starting domain adaptation training for {config.training.get('n_epochs_adapt', 50)} epochs")
    trainer_adapt.train_domain_adaptation(
        src_loader=source_train_loader,
        tgt_loader=create_loader_from_config(
            target_train_ds,
            target_data_config,
            shuffle=True,
        ),
        optimizer=optimizer_adapt,
        scheduler=scheduler_adapt,
        training_loss=training_loss_fn,
        # da_class_loss_weight controls adversarial training strength
        # Default 0.0 disables adversarial training (standard fine-tuning)
        # Set to positive value (e.g., 0.1) to enable domain adaptation
        class_loss_weight=config.training.get("da_class_loss_weight", 0.0),
        adaptation_epochs=config.training.get("n_epochs_adapt", 50),
        save_every=None,  # Save at end only, or set to save interval
        save_dir=save_dir,
        resume_from_dir=config.checkpoint.get("resume_from_adapt", None),
        resume_classifier_from_dir=config.checkpoint.get("resume_from_adapt", None),
        val_loaders={"source_val": source_val_loader, "target_val": target_val_loader},
    )

    dist_manager = DistributedManager()
    if dist_manager.rank == 0:
        normalizers_path = os.path.join(save_dir, "normalizers.pt")
        torch.save(normalizers, normalizers_path)
        logger.info(f"Saved normalizers to {normalizers_path}")
    
    return model, domain_classifier, trainer_adapt
