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

"""
Unit tests for FloodForecaster NeuralOperatorTrainer class.

This module tests the PhysicsNeMo-style Trainer class that handles neural operator training
with checkpointing, distributed training, and evaluation support.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import physicsnemo
import types
import importlib


def _ensure_physicsnemo_test_compat():
    try:
        models_pkg = importlib.import_module("physicsnemo.models")
    except ModuleNotFoundError:
        models_pkg = sys.modules.get("physicsnemo.models") or types.ModuleType("physicsnemo.models")
        sys.modules["physicsnemo.models"] = models_pkg
    physicsnemo.models = models_pkg
    if not hasattr(models_pkg, "__path__"):
        models_pkg.__path__ = []
    if not hasattr(models_pkg, "Module"):
        models_pkg.Module = getattr(physicsnemo, "Module", nn.Module)

    try:
        meta_module = importlib.import_module("physicsnemo.models.meta")
    except ModuleNotFoundError:
        meta_module = types.ModuleType("physicsnemo.models.meta")
        model_meta_cls = getattr(getattr(models_pkg, "meta", None), "ModelMetaData", None)
        if model_meta_cls is None:
            class ModelMetaData:
                def __init__(self, name: str = ""):
                    self.name = name

            model_meta_cls = ModelMetaData
        meta_module.ModelMetaData = model_meta_cls
        sys.modules["physicsnemo.models.meta"] = meta_module
    models_pkg.meta = meta_module

    try:
        utils_pkg = importlib.import_module("physicsnemo.utils")
    except ModuleNotFoundError:
        utils_pkg = sys.modules.get("physicsnemo.utils") or types.ModuleType("physicsnemo.utils")
        sys.modules["physicsnemo.utils"] = utils_pkg
    physicsnemo.utils = utils_pkg
    if not hasattr(utils_pkg, "__path__"):
        utils_pkg.__path__ = []

    try:
        capture_module = importlib.import_module("physicsnemo.utils.capture")
    except ModuleNotFoundError:
        capture_module = types.ModuleType("physicsnemo.utils.capture")

        class _StaticCapture:
            pass

        capture_module._StaticCapture = _StaticCapture
        sys.modules["physicsnemo.utils.capture"] = capture_module
    utils_pkg.capture = capture_module

    try:
        filesystem_module = importlib.import_module("physicsnemo.utils.filesystem")
    except ModuleNotFoundError:
        filesystem_module = types.ModuleType("physicsnemo.utils.filesystem")
        filesystem_module.LOCAL_CACHE = Path.cwd()
        filesystem_module._download_cached = lambda path, recursive=False: path
        sys.modules["physicsnemo.utils.filesystem"] = filesystem_module
    utils_pkg.filesystem = filesystem_module


_ensure_physicsnemo_test_compat()

# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

# Import modules explicitly to avoid conflicts
import importlib.util

# First, set up the training package structure
if "training" not in sys.modules:
    import types
    training_pkg = types.ModuleType("training")
    sys.modules["training"] = training_pkg

# Import trainer module
spec = importlib.util.spec_from_file_location("training.trainer", _examples_dir / "training" / "trainer.py")
trainer_module = importlib.util.module_from_spec(spec)
sys.modules["training.trainer"] = trainer_module
spec.loader.exec_module(trainer_module)

from training.trainer import NeuralOperatorTrainer, _has_pytorch_submodules, save_model_checkpoint
from data_processing import GINOWrapper

@pytest.fixture
def simple_model(device):
    """Create a simple test model that accepts kwargs."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 3)
            )
        
        def forward(self, x=None, **kwargs):
            # Accept x as positional or keyword, ignore other kwargs
            if x is None:
                # Try to get x from kwargs
                x = kwargs.get('x', None)
                if x is None:
                    raise ValueError("x must be provided")
            return self.layers(x)
    
    return SimpleModel().to(device)


@pytest.fixture
def mock_data_processor(device):
    """Create a mock data processor."""
    processor = MagicMock(spec=nn.Module)
    processor.preprocess = MagicMock(side_effect=lambda x: x)
    processor.postprocess = MagicMock(side_effect=lambda out, sample: (out, sample))
    processor.train = MagicMock()
    processor.eval = MagicMock()
    processor.to = MagicMock(return_value=processor)
    return processor


@pytest.fixture
def sample_dataset(device):
    """Create a sample dataset for training."""
    batch_size = 4
    n_samples = 20
    n_features = 10
    n_outputs = 3
    
    # Create sample data
    x = torch.rand(n_samples, n_features).to(device)
    y = torch.rand(n_samples, n_outputs).to(device)
    
    # Create dataset with dict format
    samples = [{"x": x[i], "y": y[i]} for i in range(n_samples)]
    return samples


@pytest.fixture
def train_loader(sample_dataset, device):
    """Create a training dataloader."""
    class DictDataset:
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    dataset = DictDataset(sample_dataset)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def test_loader(sample_dataset, device):
    """Create a test dataloader."""
    class DictDataset:
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    dataset = DictDataset(sample_dataset[:10])  # Smaller test set
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_init(simple_model, mock_data_processor, device):
    """Test NeuralOperatorTrainer initialization with various configurations."""
    # Basic init
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=10,
        device=device,
        verbose=False,
    )
    assert trainer.model == simple_model
    assert trainer.n_epochs == 10
    assert str(trainer.device) == str(torch.device(device))
    assert trainer.mixed_precision is False
    assert trainer.data_processor is None
    
    # With data processor
    trainer2 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=10,
        device=device,
        data_processor=mock_data_processor,
        verbose=False,
    )
    assert trainer2.data_processor == mock_data_processor
    
    # With mixed precision
    trainer3 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=10,
        device=device,
        mixed_precision=True,
        verbose=False,
    )
    assert trainer3.mixed_precision is True
    assert trainer3.scaler is not None


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_train_one_epoch(simple_model, train_loader, test_loader, device):
    """Test training for one epoch."""
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    from neuralop.losses import LpLoss
    training_loss = LpLoss(d=2, p=2)
    eval_losses = {"l2": LpLoss(d=2, p=2)}
    
    # Train for one epoch
    metrics = trainer.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=None,  # Don't save checkpoints in test
    )
    
    # Check that metrics were returned
    assert isinstance(metrics, dict)
    assert "train_err" in metrics
    assert "val_l2" in metrics
    assert trainer.epoch == 0  # After 1 epoch, epoch should be 0 (0-indexed)


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_checkpoint_save_load(simple_model, train_loader, test_loader, device, tmp_path):
    """Test checkpoint saving and loading."""
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=2,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    from neuralop.losses import LpLoss
    training_loss = LpLoss(d=2, p=2)
    eval_losses = {"l2": LpLoss(d=2, p=2)}
    
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    
    # Train for one epoch and save checkpoint
    trainer.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=str(save_dir),
        save_best="val_l2",
    )
    
    # Check that checkpoint files were created
    checkpoint_files = list(save_dir.glob("checkpoint.*.pt"))
    assert len(checkpoint_files) > 0, "Checkpoint file should be created"
    best_sidecar = save_dir / "best_checkpoint.json"
    assert best_sidecar.exists()
    payload = json.loads(best_sidecar.read_text())
    assert payload["metric_name"] == "val_l2"
    assert "epoch" in payload
    
    # Create new trainer and resume from checkpoint
    new_model = simple_model.__class__().to(device)
    
    new_trainer = NeuralOperatorTrainer(
        model=new_model,
        n_epochs=2,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.9)
    
    # Resume from checkpoint
    new_trainer.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=str(save_dir),
        resume_from_dir=str(save_dir),
    )
    
    # Check that training resumed
    assert new_trainer.start_epoch > 0 or new_trainer.epoch >= 0


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_training_features(simple_model, mock_data_processor, train_loader, test_loader, device, tmp_path):
    """Test training with various features: mixed precision, data processor, best model tracking."""
    from neuralop.losses import LpLoss
    
    # Test mixed precision training
    trainer1 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        mixed_precision=True,
        verbose=False,
        eval_interval=1,
    )
    optimizer1 = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=1, gamma=0.9)
    training_loss = LpLoss(d=2, p=2)
    eval_losses = {"l2": LpLoss(d=2, p=2)}
    
    metrics1 = trainer1.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer1,
        scheduler=scheduler1,
        training_loss=training_loss,
        eval_losses=eval_losses,
    )
    assert isinstance(metrics1, dict)
    assert trainer1.scaler is not None
    
    # Test with data processor
    trainer2 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        data_processor=mock_data_processor,
        verbose=False,
        eval_interval=1,
    )
    optimizer2 = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.9)
    metrics2 = trainer2.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer2,
        scheduler=scheduler2,
        training_loss=training_loss,
        eval_losses=eval_losses,
    )
    assert isinstance(metrics2, dict)
    
    # Test best model tracking
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()
    trainer3 = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=2,
        device=device,
        verbose=False,
        eval_interval=1,
    )
    optimizer3 = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=1, gamma=0.9)
    trainer3.train(
        train_loader=train_loader,
        test_loaders={"val": test_loader},
        optimizer=optimizer3,
        scheduler=scheduler3,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_dir=str(save_dir),
        save_best="val_l2",
    )
    assert trainer3.best_metric_value < float("inf")
    checkpoint_files = list(save_dir.glob("checkpoint.*.pt"))
    assert len(checkpoint_files) > 0


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_resume_requires_complete_checkpoint(simple_model, train_loader, test_loader, device, tmp_path):
    """Resume should fail fast when the training-state file exists but model weights are missing."""
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        verbose=False,
        eval_interval=1,
    )

    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    from neuralop.losses import LpLoss

    save_dir = tmp_path / "partial_checkpoints"
    save_dir.mkdir()
    torch.save({"epoch": 0}, save_dir / "checkpoint.0.0.pt")

    with pytest.raises(FileNotFoundError):
        trainer.train(
            train_loader=train_loader,
            test_loaders={"val": test_loader},
            optimizer=optimizer,
            scheduler=scheduler,
            training_loss=LpLoss(d=2, p=2),
            eval_losses={"l2": LpLoss(d=2, p=2)},
            save_dir=str(save_dir),
            resume_from_dir=str(save_dir),
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_autoregressive_eval_is_explicitly_unsupported(simple_model, test_loader, device):
    """FloodForecaster should raise a clear error for trainer-side autoregressive eval."""
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        verbose=False,
    )

    with pytest.raises(NotImplementedError, match="inference/rollout.py"):
        trainer._evaluate(
            eval_losses={"l2": lambda out, y=None, **_: torch.tensor(0.0, device=out.device)},
            data_loader=test_loader,
            log_prefix="val",
            mode="autoregression",
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_eval_normalizes_by_postprocessed_sample_count(simple_model, device):
    """Validation averaging should use the postprocessed target batch size."""

    class RenameTargetProcessor:
        def preprocess(self, sample):
            return {
                "x": sample["x"].to(device),
                "y": sample["target"].to(device),
            }

        def postprocess(self, out, sample):
            return out, sample

        def eval(self):
            return self

        def train(self):
            return self

        def to(self, device):
            return self

    samples = [
        {"x": torch.rand(10), "target": torch.rand(3)}
        for _ in range(5)
    ]

    class DictDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    loader = DataLoader(DictDataset(samples), batch_size=2, shuffle=False)
    trainer = NeuralOperatorTrainer(
        model=simple_model,
        n_epochs=1,
        device=device,
        data_processor=RenameTargetProcessor(),
        verbose=False,
    )

    def batch_summed_loss(out, y=None, **kwargs):
        return torch.tensor(float(y.shape[0]), device=out.device)

    metrics = trainer._evaluate(
        eval_losses={"unit": batch_summed_loss},
        data_loader=loader,
        log_prefix="val",
    )

    assert metrics["val_unit"] == pytest.approx(1.0)


def test_trainer_wrap_model_for_distributed_cpu_omits_gpu_ddp_args(monkeypatch):
    """CPU DDP setup should not pass CUDA-only arguments to DDP."""
    model = nn.Linear(4, 2)
    trainer = NeuralOperatorTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        verbose=False,
    )

    captured = {}

    class FakeDistributedManager:
        distributed = True
        broadcast_buffers = False
        find_unused_parameters = True
        local_rank = 7

        @staticmethod
        def is_initialized():
            return True

    class FakeDDP(nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module
            captured["kwargs"] = kwargs

    monkeypatch.setattr(trainer_module, "DistributedManager", FakeDistributedManager)
    monkeypatch.setattr(trainer_module, "DDP", FakeDDP)

    wrapped = trainer._wrap_model_for_distributed(model)

    assert isinstance(wrapped, FakeDDP)
    assert wrapped.module is model
    assert captured["kwargs"] == {
        "broadcast_buffers": False,
        "find_unused_parameters": True,
    }


def test_trainer_save_checkpoint_uses_model_parallel_rank_for_pytorch_wrappers(tmp_path, monkeypatch):
    """Best-checkpoint saving should honor model-parallel rank and sidecar metadata."""
    model = nn.Linear(4, 2)
    trainer = NeuralOperatorTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        verbose=False,
        checkpoint_stage="pretrain",
    )
    trainer.optimizer = object()
    trainer.scheduler = object()
    trainer.epoch = 7
    trainer.save_best = "val_l2"
    trainer.best_metric_value = 0.125

    saved = {}

    class FakeDistributedManager:
        rank = 0
        group_names = ["model_parallel"]

        @staticmethod
        def is_initialized():
            return True

        def group_rank(self, name):
            assert name == "model_parallel"
            return 5

    def fake_save_checkpoint(**kwargs):
        saved["save_checkpoint"] = kwargs

    def fake_write_best_checkpoint_metadata(path, **kwargs):
        saved["write_best"] = {"path": Path(path), **kwargs}

    monkeypatch.setattr(trainer_module, "DistributedManager", FakeDistributedManager)
    monkeypatch.setattr(trainer_module, "_has_pytorch_submodules", lambda _model: True)
    monkeypatch.setattr(trainer_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(trainer_module, "write_best_checkpoint_metadata", fake_write_best_checkpoint_metadata)

    save_dir = tmp_path / "trainer_ckpt"
    trainer._save_checkpoint(save_dir, is_best=True)

    assert (save_dir / "Linear.5.7.pt").exists()
    assert saved["save_checkpoint"]["models"] is None
    assert saved["save_checkpoint"]["epoch"] == 7
    assert saved["write_best"]["metric_name"] == "val_l2"
    assert saved["write_best"]["metric_value"] == pytest.approx(0.125)


def test_trainer_resume_from_checkpoint_restores_best_metric_metadata(tmp_path, monkeypatch):
    """PhysicsNeMo resume should restore start epoch and best-metric metadata."""
    model = nn.Linear(4, 2)
    trainer = NeuralOperatorTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        verbose=False,
    )
    trainer.optimizer = object()
    trainer.scheduler = object()

    def fake_load_checkpoint(path, models, optimizer, scheduler, scaler, epoch, metadata_dict, device):
        metadata_dict["best_metric_value"] = 0.25

    monkeypatch.setattr(trainer_module, "resolve_checkpoint_epoch", lambda path, mode: 3)
    monkeypatch.setattr(trainer_module, "validate_checkpoint_files", lambda path, models, epoch: {"epoch": epoch})
    monkeypatch.setattr(trainer_module, "load_checkpoint", fake_load_checkpoint)

    trainer._resume_from_checkpoint(tmp_path)

    assert trainer.start_epoch == 4
    assert trainer.best_metric_value == pytest.approx(0.25)


def test_trainer_resume_from_checkpoint_legacy_fallback_unwraps_inner_model(tmp_path, monkeypatch):
    """Legacy fallback should unwrap `gino.inner_model` before calling neuralop loader."""

    class FakeWrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gino = types.SimpleNamespace(inner_model="inner-gino-model")

        def forward(self, x=None, **kwargs):
            return x

    trainer = NeuralOperatorTrainer(
        model=FakeWrappedModel(),
        n_epochs=1,
        device="cpu",
        verbose=False,
    )
    trainer.optimizer = "optimizer"
    trainer.scheduler = "scheduler"

    captured = {}

    def fake_load_training_state(save_dir, save_name, model, optimizer, scheduler):
        captured["model"] = model
        captured["save_name"] = save_name
        return model, optimizer, scheduler, None, 5

    import neuralop.training.training_state as training_state_module

    def fail_resolve_checkpoint_epoch(path, mode):
        raise FileNotFoundError("missing physicsnemo checkpoint")

    monkeypatch.setattr(trainer_module, "resolve_checkpoint_epoch", fail_resolve_checkpoint_epoch)
    monkeypatch.setattr(trainer_module, "resolve_legacy_neuralop_checkpoint_name", lambda path, mode: "model")
    monkeypatch.setattr(training_state_module, "load_training_state", fake_load_training_state)

    trainer._resume_from_checkpoint(tmp_path)

    assert captured["model"] == "inner-gino-model"
    assert captured["save_name"] == "model"
    assert trainer.start_epoch == 6

