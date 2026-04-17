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
Unit tests for FloodForecaster training modules.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

# Import modules explicitly to avoid conflicts with other utils modules
import importlib.util

# First, set up the training package structure
if "training" not in sys.modules:
    import types
    training_pkg = types.ModuleType("training")
    sys.modules["training"] = training_pkg

# Import trainer module FIRST (pretraining depends on it)
spec = importlib.util.spec_from_file_location("training.trainer", _examples_dir / "training" / "trainer.py")
trainer_module = importlib.util.module_from_spec(spec)
sys.modules["training.trainer"] = trainer_module
spec.loader.exec_module(trainer_module)

# Import pretraining (depends on trainer)
spec = importlib.util.spec_from_file_location("training.pretraining", _examples_dir / "training" / "pretraining.py")
pretraining_module = importlib.util.module_from_spec(spec)
sys.modules["training.pretraining"] = pretraining_module
spec.loader.exec_module(pretraining_module)

# Import domain_adaptation (depends on pretraining, but not on __init__)
# Use full module path to ensure __module__ attribute is set correctly for checkpoint loading
spec = importlib.util.spec_from_file_location("training.domain_adaptation", _examples_dir / "training" / "domain_adaptation.py")
domain_adaptation_module = importlib.util.module_from_spec(spec)
sys.modules["training.domain_adaptation"] = domain_adaptation_module
spec.loader.exec_module(domain_adaptation_module)

# Fix __module__ attributes for classes to ensure checkpoint loading works
for name in dir(domain_adaptation_module):
    obj = getattr(domain_adaptation_module, name)
    if isinstance(obj, type) and issubclass(obj, (torch.nn.Module, physicsnemo.Module)):
        obj.__module__ = "training.domain_adaptation"

# Now import __init__ (it can safely import from domain_adaptation and pretraining)
spec = importlib.util.spec_from_file_location("training", _examples_dir / "training" / "__init__.py")
training_init = importlib.util.module_from_spec(spec)
sys.modules["training"].__dict__.update(training_init.__dict__)
spec.loader.exec_module(training_init)

from training.domain_adaptation import (
    CNNDomainClassifier,
    DomainAdaptationTrainer,
    GradientReversal,
    GradientReversalFunction,
)
from training.pretraining import create_scheduler


class _SimpleDARegressionModel(nn.Module):
    """Small model that supports optional feature return for DA tests."""

    def __init__(self, feature_channels: int = 4):
        super().__init__()
        self.linear = nn.Linear(10, 3)
        self.feature_channels = feature_channels

    def forward(self, x=None, return_features: bool = False, **kwargs):
        if x is None:
            x = kwargs["x"]
        out = self.linear(x)
        if return_features:
            features = torch.ones(x.shape[0], self.feature_channels, 2, 2, device=x.device)
            return out, features
        return out


class _ClassifierShouldNotRun(nn.Module):
    """Classifier sentinel for verifying the zero-weight DA bypass."""

    def __init__(self):
        super().__init__()
        self.grl = GradientReversal(lambda_max=1.0)
        self.dummy = nn.Linear(1, 1)

    def forward(self, x):
        raise AssertionError("Domain classifier should not be called when class_loss_weight <= 0")


class _DictDataset(Dataset):
    """Tiny dataset wrapper for DA trainer tests."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("scheduler_type", ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"])
def test_create_scheduler(device, scheduler_type):
    """Test scheduler creation for different types."""
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    config = MagicMock()
    config.training = MagicMock()
    if scheduler_type == "StepLR":
        config.training.get = lambda key, default=None: {
            "scheduler": "StepLR",
            "step_size": 10,
            "gamma": 0.5,
        }.get(key, default)
        expected_type = torch.optim.lr_scheduler.StepLR
    elif scheduler_type == "CosineAnnealingLR":
        config.training.get = lambda key, default=None: {
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": 100,
        }.get(key, default)
        expected_type = torch.optim.lr_scheduler.CosineAnnealingLR
    else:  # ReduceLROnPlateau
        config.training.get = lambda key, default=None: {
            "scheduler": "ReduceLROnPlateau",
            "gamma": 0.5,
            "scheduler_patience": 5,
        }.get(key, default)
        expected_type = torch.optim.lr_scheduler.ReduceLROnPlateau

    scheduler = create_scheduler(optimizer, config)
    assert isinstance(scheduler, expected_type)


@pytest.mark.parametrize("device", _DEVICES)
def test_gradient_reversal(device):
    """Test gradient reversal forward and backward passes."""
    grl = GradientReversal(lambda_max=1.0)
    x = torch.rand(4, 10, requires_grad=True).to(device).detach().requires_grad_(True)

    # Forward: identity
    y = grl(x)
    assert torch.allclose(x, y)

    # Backward: negates gradient
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, -torch.ones_like(x.grad))


@pytest.fixture
def da_config():
    """Create mock DA config that supports .get() method, 'in' operator, and [] access."""
    # Use a dict-like object that supports both attribute access and dict-like access
    class DictLike:
        def __init__(self):
            self.conv_layers = [
                {"out_channels": 16, "kernel_size": 3, "pool_size": 2},
                {"out_channels": 32, "kernel_size": 3, "pool_size": 2},
            ]
            self.fc_dim = 1
        
        def get(self, key, default=None):
            """Support .get() method for dict-like access."""
            return getattr(self, key, default)
        
        def __contains__(self, key):
            """Support 'in' operator for dict-like access."""
            return hasattr(self, key)
        
        def __getitem__(self, key):
            """Support [] subscript access for dict-like access."""
            return getattr(self, key)
    
    return DictLike()


@pytest.mark.parametrize("device", _DEVICES)
def test_cnn_domain_classifier_init(da_config, device):
    """Test CNN domain classifier initialization."""
    classifier = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)

    assert isinstance(classifier, nn.Module)
    assert hasattr(classifier, "grl")
    assert isinstance(classifier.grl, GradientReversal)


@pytest.mark.parametrize("device", _DEVICES)
def test_cnn_domain_classifier_forward(da_config, device):
    """Test CNN domain classifier forward pass."""
    classifier = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)

    x = torch.rand(4, 64, 16, 16).to(device)  # (B, C, H, W)
    y = classifier(x)

    assert y.shape == (4, 1)  # (B, fc_dim)


@pytest.mark.parametrize("device", _DEVICES)
def test_domain_adaptation_trainer_init(device):
    """Test DomainAdaptationTrainer initialization."""
    mock_model = MagicMock(spec=nn.Module)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_classifier = MagicMock(spec=nn.Module)
    mock_classifier.to = MagicMock(return_value=mock_classifier)
    mock_processor = MagicMock()
    mock_processor.to = MagicMock(return_value=mock_processor)

    trainer = DomainAdaptationTrainer(
        model=mock_model,
        data_processor=mock_processor,
        domain_classifier=mock_classifier,
        device=device,
    )

    assert trainer.model == mock_model
    assert trainer.domain_classifier == mock_classifier
    assert trainer.data_processor == mock_processor
    assert trainer.device == device


@pytest.mark.parametrize("device", _DEVICES)
def test_gradient_reversal_lambda(device):
    """Test GradientReversal lambda setting and scaling."""
    grl = GradientReversal(lambda_max=1.0)
    grl.set_lambda(0.5)
    assert grl.lambda_ == 0.5

    # Test lambda scales gradient
    grl2 = GradientReversal(lambda_max=0.5)
    x2 = torch.ones(4, 10, requires_grad=True).to(device).detach().requires_grad_(True)
    y2 = grl2(x2)
    loss2 = y2.sum()
    loss2.backward()
    assert x2.grad is not None
    assert torch.allclose(x2.grad, -0.5 * torch.ones_like(x2.grad))


@pytest.mark.parametrize("device", _DEVICES)
def test_create_scheduler_unknown_raises(device):
    """Test that unknown scheduler raises ValueError."""
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    config = MagicMock()
    config.training = MagicMock()
    config.training.get = lambda key, default=None: {
        "scheduler": "UnknownScheduler",
    }.get(key, default)

    with pytest.raises(ValueError, match="Unknown scheduler"):
        create_scheduler(optimizer, config)




def _instantiate_model(cls, seed: int = 0, **kwargs):
    """Helper to create model with reproducible parameters."""
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model


@pytest.mark.parametrize("device", _DEVICES)
def test_checkpoint_save_load(device):
    """Test checkpoint save/load for training components."""
    import physicsnemo
    from pathlib import Path
    
    # Test GradientReversal checkpoint
    grl_orig = GradientReversal(lambda_max=1.0).to(device)
    grl_path = Path("checkpoint_grl.mdlus")
    grl_orig.save(str(grl_path))
    grl_loaded = physicsnemo.Module.from_checkpoint(str(grl_path)).to(device)
    assert grl_loaded.lambda_ == 1.0
    assert isinstance(grl_loaded, GradientReversal)
    grl_path.unlink(missing_ok=True)
    
    # Test CNNDomainClassifier checkpoint
    da_config = {
        "conv_layers": [
            {"out_channels": 16, "kernel_size": 3, "pool_size": 2},
            {"out_channels": 32, "kernel_size": 3, "pool_size": 2},
        ],
        "fc_dim": 1
    }
    classifier_orig = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)
    classifier_path = Path("checkpoint_classifier.mdlus")
    classifier_orig.save(str(classifier_path))
    classifier_loaded = physicsnemo.Module.from_checkpoint(str(classifier_path)).to(device)
    assert classifier_loaded.grl.lambda_ == 1.0
    assert isinstance(classifier_loaded, CNNDomainClassifier)
    classifier_path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", _DEVICES)
def test_domain_adaptation_skips_classifier_when_weight_is_zero(device, tmp_path):
    """Zero classifier weight should bypass the adversarial classifier path entirely."""
    model = _SimpleDARegressionModel().to(device)
    classifier = _ClassifierShouldNotRun().to(device)
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=None,
        domain_classifier=classifier,
        device=device,
        verbose=False,
    )

    samples = [{"x": torch.rand(10).to(device), "y": torch.rand(3).to(device)} for _ in range(8)]
    loader = DataLoader(_DictDataset(samples), batch_size=4, shuffle=False)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer.train_domain_adaptation(
        src_loader=loader,
        tgt_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y),
        class_loss_weight=0.0,
        adaptation_epochs=1,
        save_dir=tmp_path / "adapt_skip_classifier",
        val_loaders={},
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_domain_adaptation_respects_da_lambda_max(device, tmp_path):
    """GRL scheduling should be scaled by the configured lambda max."""
    lambda_max = 0.5
    model = _SimpleDARegressionModel(feature_channels=4).to(device)
    classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=lambda_max,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    ).to(device)
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=None,
        domain_classifier=classifier,
        device=device,
        verbose=False,
    )

    samples = [{"x": torch.rand(10).to(device), "y": torch.rand(3).to(device)} for _ in range(4)]
    loader = DataLoader(_DictDataset(samples), batch_size=4, shuffle=False)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer.train_domain_adaptation(
        src_loader=loader,
        tgt_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y),
        class_loss_weight=1.0,
        adaptation_epochs=2,
        save_dir=tmp_path / "adapt_lambda_max",
        val_loaders={},
    )

    expected = lambda_max * (2.0 / (1.0 + torch.exp(torch.tensor(-10.0))) - 1.0)
    assert abs(classifier.grl.lambda_ - float(expected)) < 1e-6


@pytest.mark.parametrize("device", _DEVICES)
def test_domain_adaptation_eval_normalizes_by_postprocessed_sample_count(device):
    """DA validation should divide summed loss by the postprocessed sample count."""

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

    model = _SimpleDARegressionModel().to(device)
    classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=1.0,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    ).to(device)
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=RenameTargetProcessor(),
        domain_classifier=classifier,
        device=device,
        verbose=False,
    )

    samples = [{"x": torch.rand(10), "target": torch.rand(3)} for _ in range(5)]
    loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)

    def batch_summed_loss(out, y=None, **kwargs):
        return torch.tensor(float(y.shape[0]), device=out.device)

    metrics = trainer._evaluate({"target_val": loader}, batch_summed_loss, epoch=0)
    assert metrics["target_val"] == pytest.approx(1.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_domain_adaptation_writes_best_checkpoint_sidecar(device, tmp_path):
    """The first epoch should save a best-checkpoint sidecar keyed to target validation."""
    model = _SimpleDARegressionModel().to(device)
    classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=1.0,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    ).to(device)
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=None,
        domain_classifier=classifier,
        device=device,
        verbose=False,
    )

    samples = [{"x": torch.rand(10).to(device), "y": torch.rand(3).to(device)} for _ in range(4)]
    loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    save_dir = tmp_path / "adapt_best"

    trainer.train_domain_adaptation(
        src_loader=loader,
        tgt_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y),
        class_loss_weight=0.0,
        adaptation_epochs=1,
        save_dir=save_dir,
        val_loaders={"source_val": loader, "target_val": loader},
    )

    sidecar = save_dir / "best_checkpoint.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["metric_name"] == "target_val"
    assert payload["epoch"] == 0


@pytest.mark.parametrize("device", _DEVICES)
def test_pretrain_model_wires_datasets_loaders_and_normalizer_save(device, tmp_path, monkeypatch):
    """Pretraining should split data, build loaders, invoke the trainer, and save normalizers on rank 0."""
    from omegaconf import OmegaConf

    fake_samples = [{"index": idx} for idx in range(10)]

    class FakeNormalizedDataset:
        def __init__(self, geometry, static, boundary, dynamic, target, query_res):
            self.geometry = geometry
            self.static = static
            self.boundary = boundary
            self.dynamic = dynamic
            self.target = target
            self.query_res = query_res

        def __len__(self):
            return int(self.geometry.shape[0])

        def __getitem__(self, idx):
            return {
                "geometry": self.geometry[idx],
                "static": self.static[idx],
                "boundary": self.boundary[idx],
                "dynamic": self.dynamic[idx],
                "target": self.target[idx],
            }

    class FakeWrappedModel(nn.Module):
        def __init__(self, inner, autoregressive=False):
            super().__init__()
            self.inner = inner
            self.autoregressive = autoregressive

        def forward(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

    class FakeDataProcessor:
        def __init__(self, device, target_norm=None, inverse_test=True):
            self.device = device
            self.target_norm = target_norm
            self.inverse_test = inverse_test
            self.wrapped_model = None

        def wrap(self, model):
            self.wrapped_model = model

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.train_kwargs = None

        def train(self, **kwargs):
            self.train_kwargs = kwargs
            return {"source_val_l2": 0.0}

    loader_calls = []

    def fake_collect_all_fields(dataset, expect_target):
        n = len(dataset)
        geom = [torch.zeros(2, 2) for _ in range(n)]
        static = [torch.zeros(2, 1) for _ in range(n)]
        boundary = [torch.zeros(3, 2, 1) for _ in range(n)]
        dynamic = [torch.zeros(3, 2, 3) for _ in range(n)]
        target = [torch.zeros(2, 3) for _ in range(n)]
        cell_area = [None for _ in range(n)]
        return geom, static, boundary, dynamic, target, cell_area

    def fake_stack_and_fit_transform(geom, static, boundary, dyn, target, normalizers=None, fit_normalizers=True):
        n = len(geom)
        fitted = normalizers or {
            "dynamic": "dynamic_norm",
            "target": "target_norm",
            "boundary": "boundary_norm",
            "static": "static_norm",
        }
        big = {
            "geometry": torch.zeros(n, 2, 2),
            "static": torch.zeros(n, 2, 1),
            "boundary": torch.zeros(n, 3, 2, 1),
            "dynamic": torch.zeros(n, 3, 2, 3),
            "target": torch.zeros(n, 2, 3),
        }
        return fitted, big

    def fake_create_loader_from_config(dataset, data_config, shuffle):
        loader = {
            "dataset": dataset,
            "shuffle": shuffle,
            "batch_size": data_config.batch_size,
        }
        loader_calls.append(loader)
        return loader

    monkeypatch.setattr(pretraining_module, "FloodDatasetWithQueryPoints", lambda **kwargs: fake_samples)
    monkeypatch.setattr(pretraining_module, "NormalizedDataset", FakeNormalizedDataset)
    monkeypatch.setattr(pretraining_module, "collect_all_fields", fake_collect_all_fields)
    monkeypatch.setattr(pretraining_module, "stack_and_fit_transform", fake_stack_and_fit_transform)
    monkeypatch.setattr(pretraining_module, "create_loader_from_config", fake_create_loader_from_config)
    monkeypatch.setattr(pretraining_module, "FloodGINODataProcessor", FakeDataProcessor)
    monkeypatch.setattr(pretraining_module, "NeuralOperatorTrainer", FakeTrainer)
    monkeypatch.setattr(pretraining_module, "get_model", lambda cfg: nn.Linear(1, 1))
    monkeypatch.setattr(pretraining_module, "GINOWrapper", FakeWrappedModel)
    monkeypatch.setattr(pretraining_module, "DistributedManager", lambda: type("Dist", (), {"rank": 0})())

    config = OmegaConf.create(
        {
            "model": {"name": "stub_model", "autoregressive": True},
            "distributed": {"seed": 123},
            "training": {
                "batch_size": 2,
                "scheduler": "StepLR",
                "step_size": 2,
                "gamma": 0.5,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "training_loss": "l2",
                "testing_loss": "l2",
                "amp_autocast": True,
                "n_epochs_source": 3,
            },
            "wandb": {"log": False},
            "checkpoint": {
                "save_dir": str(tmp_path),
                "save_best": "source_val_l2",
                "save_every": None,
                "resume_from_source": None,
            },
        }
    )
    source_data_config = OmegaConf.create(
        {
            "root": str(tmp_path / "source"),
            "n_history": 3,
            "query_res": [8, 8],
            "batch_size": 2,
            "static_files": [],
            "dynamic_patterns": {},
            "boundary_patterns": {},
        }
    )

    model, normalizers, trainer = pretraining_module.pretrain_model(
        config=config,
        device=device,
        is_logger=False,
        source_data_config=source_data_config,
        logger=MagicMock(),
    )

    assert isinstance(model, FakeWrappedModel)
    assert model.autoregressive is True
    assert normalizers["target"] == "target_norm"
    assert trainer.init_kwargs["mixed_precision"] is True
    assert trainer.train_kwargs["save_best"] == "source_val_l2"
    assert trainer.train_kwargs["test_loaders"]["source_val"]["shuffle"] is False
    assert trainer.source_train_loader["shuffle"] is True
    assert trainer.source_val_loader["shuffle"] is False
    assert len(trainer.source_train_dataset) == 9
    assert len(trainer.source_val_dataset) == 1
    assert (tmp_path / "pretrain" / "normalizers.pt").exists()
    assert len(loader_calls) == 2


def test_domain_adaptation_wrap_for_ddp_cpu_omits_gpu_args(monkeypatch):
    """CPU DDP wrapping in DA should avoid `device_ids` and `output_device`."""
    trainer = DomainAdaptationTrainer(
        model=nn.Linear(4, 2),
        data_processor=None,
        domain_classifier=CNNDomainClassifier(
            in_channels=4,
            lambda_max=1.0,
            da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
        ),
        device="cpu",
        verbose=False,
    )
    module = nn.Linear(4, 2)
    captured = {}

    class FakeDistributedManager:
        distributed = True
        broadcast_buffers = True
        find_unused_parameters = False
        local_rank = 4

        @staticmethod
        def is_initialized():
            return True

    class FakeDDP(nn.Module):
        def __init__(self, wrapped_module, **kwargs):
            super().__init__()
            self.module = wrapped_module
            captured["kwargs"] = kwargs

    monkeypatch.setattr(domain_adaptation_module, "DistributedManager", FakeDistributedManager)
    monkeypatch.setattr(domain_adaptation_module.torch.nn.parallel, "DistributedDataParallel", FakeDDP)

    wrapped = trainer._wrap_for_ddp(module)

    assert isinstance(wrapped, FakeDDP)
    assert wrapped.module is module
    assert captured["kwargs"] == {
        "broadcast_buffers": True,
        "find_unused_parameters": False,
    }


def test_domain_adaptation_save_checkpoint_includes_classifier_and_best_sidecar(tmp_path, monkeypatch):
    """DA checkpoint saving should include classifier state and emit best metadata."""
    model = _SimpleDARegressionModel()
    classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=1.0,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    )
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=None,
        domain_classifier=classifier,
        device="cpu",
        verbose=False,
    )
    trainer.best_metric_value = 0.2

    saved = {}

    class FakeDistributedManager:
        rank = 0
        group_names = ["model_parallel"]

        @staticmethod
        def is_initialized():
            return True

        def group_rank(self, name):
            assert name == "model_parallel"
            return 3

    monkeypatch.setattr(domain_adaptation_module, "DistributedManager", FakeDistributedManager)
    monkeypatch.setattr(domain_adaptation_module, "save_model_checkpoint", lambda **kwargs: False)
    monkeypatch.setattr(domain_adaptation_module, "save_checkpoint", lambda **kwargs: saved.setdefault("save", kwargs))
    monkeypatch.setattr(
        domain_adaptation_module,
        "write_best_checkpoint_metadata",
        lambda path, **kwargs: saved.setdefault("best", {"path": Path(path), **kwargs}),
    )

    trainer._save_checkpoint(
        save_dir=tmp_path / "adapt_ckpt",
        optimizer="optimizer",
        scheduler="scheduler",
        epoch=2,
        save_classifier=True,
        is_best=True,
        metric_name="target_val",
    )

    assert len(saved["save"]["models"]) == 2
    assert saved["save"]["models"][0] is model
    assert saved["save"]["models"][1] is classifier
    assert saved["best"]["metric_name"] == "target_val"
    assert saved["best"]["metric_value"] == pytest.approx(0.2)


def test_domain_adaptation_resume_from_checkpoint_legacy_loads_classifier(tmp_path, monkeypatch):
    """Legacy DA resume should restore both trainer epoch and classifier weights."""
    model = _SimpleDARegressionModel()
    classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=1.0,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    )
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=None,
        domain_classifier=classifier,
        device="cpu",
        verbose=False,
    )

    reference_classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=1.0,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    )
    with torch.no_grad():
        for param in reference_classifier.parameters():
            param.fill_(7.0)
    torch.save(reference_classifier.state_dict(), tmp_path / "classifier_state_dict.pt")

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    import neuralop.training.training_state as training_state_module

    def fail_resolve_checkpoint_epoch(path, mode):
        raise FileNotFoundError("missing physicsnemo checkpoint")

    monkeypatch.setattr(domain_adaptation_module, "resolve_checkpoint_epoch", fail_resolve_checkpoint_epoch)
    monkeypatch.setattr(domain_adaptation_module, "resolve_legacy_neuralop_checkpoint_name", lambda path, mode: "model")
    monkeypatch.setattr(
        training_state_module,
        "load_training_state",
        lambda save_dir, save_name, model, optimizer, scheduler: (model, optimizer, scheduler, None, 4),
    )

    epoch = trainer._resume_from_checkpoint(tmp_path, optimizer, scheduler)

    assert epoch == 4
    for key, value in reference_classifier.state_dict().items():
        assert torch.allclose(trainer.domain_classifier.state_dict()[key], value)


def test_train_domain_adaptation_loads_separate_classifier_checkpoint_fallback(tmp_path, monkeypatch):
    """`resume_classifier_from_dir` should fall back to legacy classifier_state_dict loading."""
    model = _SimpleDARegressionModel()
    classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=1.0,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    )
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=None,
        domain_classifier=classifier,
        device="cpu",
        verbose=False,
    )

    reference_classifier = CNNDomainClassifier(
        in_channels=4,
        lambda_max=1.0,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    )
    with torch.no_grad():
        for param in reference_classifier.parameters():
            param.fill_(3.0)
    torch.save(reference_classifier.state_dict(), tmp_path / "classifier_state_dict.pt")

    monkeypatch.setattr(trainer, "_resume_from_checkpoint", lambda *args, **kwargs: -1)

    def fail_resolve_checkpoint_epoch(path, mode):
        raise FileNotFoundError("missing classifier checkpoint")

    monkeypatch.setattr(domain_adaptation_module, "resolve_checkpoint_epoch", fail_resolve_checkpoint_epoch)

    samples = [{"x": torch.rand(10), "y": torch.rand(3)} for _ in range(2)]
    loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer.train_domain_adaptation(
        src_loader=loader,
        tgt_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y),
        class_loss_weight=0.0,
        adaptation_epochs=0,
        save_dir=tmp_path / "unused",
        resume_classifier_from_dir=tmp_path,
        val_loaders={},
    )

    for key, value in reference_classifier.state_dict().items():
        assert torch.allclose(trainer.domain_classifier.state_dict()[key], value)
