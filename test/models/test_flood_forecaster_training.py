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

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "FloodForecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from training.domain_adaptation import (
    CNNDomainClassifier,
    DomainAdaptationTrainer,
    GradientReversal,
    GradientReversalFunction,
)
from training.pretraining import create_scheduler


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_create_scheduler_step_lr(device):
    """Test StepLR scheduler creation."""
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    config = MagicMock()
    config.training = MagicMock()
    config.training.get = lambda key, default=None: {
        "scheduler": "StepLR",
        "step_size": 10,
        "gamma": 0.5,
    }.get(key, default)

    scheduler = create_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_create_scheduler_cosine_annealing(device):
    """Test CosineAnnealingLR scheduler creation."""
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    config = MagicMock()
    config.training = MagicMock()
    config.training.get = lambda key, default=None: {
        "scheduler": "CosineAnnealingLR",
        "scheduler_T_max": 100,
    }.get(key, default)

    scheduler = create_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_create_scheduler_reduce_lr_on_plateau(device):
    """Test ReduceLROnPlateau scheduler creation."""
    model = nn.Linear(10, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    config = MagicMock()
    config.training = MagicMock()
    config.training.get = lambda key, default=None: {
        "scheduler": "ReduceLROnPlateau",
        "gamma": 0.5,
        "scheduler_patience": 5,
    }.get(key, default)

    scheduler = create_scheduler(optimizer, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gradient_reversal_forward(device):
    """Test gradient reversal forward pass is identity."""
    grl = GradientReversal(lambda_max=1.0)
    x = torch.rand(4, 10, requires_grad=True).to(device)

    y = grl(x)

    assert torch.allclose(x, y)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gradient_reversal_backward(device):
    """Test gradient reversal backward pass negates gradient."""
    grl = GradientReversal(lambda_max=1.0)
    x = torch.rand(4, 10, requires_grad=True).to(device)

    y = grl(x)
    loss = y.sum()
    loss.backward()

    # Gradient should be negated (-1 * lambda)
    assert x.grad is not None
    assert torch.allclose(x.grad, -torch.ones_like(x.grad))


@pytest.fixture
def da_config():
    """Create mock DA config."""
    config = MagicMock()
    config.conv_layers = [
        {"out_channels": 16, "kernel_size": 3, "pool_size": 2},
        {"out_channels": 32, "kernel_size": 3, "pool_size": 2},
    ]
    config.fc_dim = 1
    return config


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_cnn_domain_classifier_init(da_config, device):
    """Test CNN domain classifier initialization."""
    classifier = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)

    assert isinstance(classifier, nn.Module)
    assert hasattr(classifier, "grl")
    assert isinstance(classifier.grl, GradientReversal)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_cnn_domain_classifier_forward(da_config, device):
    """Test CNN domain classifier forward pass."""
    classifier = CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=da_config).to(device)

    x = torch.rand(4, 64, 16, 16).to(device)  # (B, C, H, W)
    y = classifier(x)

    assert y.shape == (4, 1)  # (B, fc_dim)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gradient_reversal_set_lambda(device):
    """Test setting lambda value in GradientReversal."""
    grl = GradientReversal(lambda_max=1.0)
    grl.set_lambda(0.5)

    assert grl.lambda_ == 0.5


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_gradient_reversal_lambda_scales_gradient(device):
    """Test that lambda scales the reversed gradient."""
    grl = GradientReversal(lambda_max=0.5)
    x = torch.ones(4, 10, requires_grad=True).to(device)

    y = grl(x)
    loss = y.sum()
    loss.backward()

    # Gradient should be -0.5 * ones
    assert torch.allclose(x.grad, -0.5 * torch.ones_like(x.grad))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_domain_adaptation_trainer_eval_interval(device):
    """Test DomainAdaptationTrainer eval_interval property."""
    mock_model = MagicMock(spec=nn.Module)
    mock_classifier = MagicMock(spec=nn.Module)
    mock_processor = MagicMock()

    trainer = DomainAdaptationTrainer(
        model=mock_model,
        data_processor=mock_processor,
        domain_classifier=mock_classifier,
        device=device,
    )

    # Default should be 1
    assert trainer.eval_interval == 1

    # Should be settable
    trainer.eval_interval = 5
    assert trainer.eval_interval == 5


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_domain_adaptation_trainer_on_epoch_start(device):
    """Test DomainAdaptationTrainer on_epoch_start method."""
    mock_model = MagicMock(spec=nn.Module)
    mock_classifier = MagicMock(spec=nn.Module)
    mock_processor = MagicMock()

    trainer = DomainAdaptationTrainer(
        model=mock_model,
        data_processor=mock_processor,
        domain_classifier=mock_classifier,
        device=device,
    )

    result = trainer.on_epoch_start(epoch=10)
    assert trainer.epoch == 10
    assert result is None


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_cnn_domain_classifier_missing_config(device):
    """Test CNNDomainClassifier raises error with missing config."""
    with pytest.raises(ValueError, match="conv_layers"):
        CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg={})

    config = MagicMock()
    config.conv_layers = [{"out_channels": 16}]
    with pytest.raises(ValueError, match="fc_dim"):
        CNNDomainClassifier(in_channels=64, lambda_max=1.0, da_cfg=config)
