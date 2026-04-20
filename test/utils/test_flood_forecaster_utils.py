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
Unit tests for FloodForecaster utility modules.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from utils.normalization import (
    collect_all_fields,
    stack_and_fit_transform,
    transform_with_existing_normalizers,
)
from utils.checkpointing import (
    resolve_checkpoint_epoch,
    resolve_legacy_neuralop_checkpoint_name,
    validate_checkpoint_files,
    write_best_checkpoint_metadata,
)
from utils.runtime import (
    create_loader_from_config,
    seed_everything,
    set_loader_epoch,
    split_dataset,
)


# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")


@pytest.mark.parametrize("device", _DEVICES)
def test_collect_all_fields_with_target(device):
    """Test collecting fields with target."""
    # Create mock dataset
    mock_dataset = [
        {
            "geometry": torch.rand(100, 2).to(device),
            "static": torch.rand(100, 7).to(device),
            "boundary": torch.rand(3, 100, 1).to(device),
            "dynamic": torch.rand(3, 100, 3).to(device),
            "target": torch.rand(100, 3).to(device),
        }
        for _ in range(5)
    ]

    geom, static, boundary, dyn, target, cell_area = collect_all_fields(mock_dataset, expect_target=True)

    assert len(geom) == 5
    assert len(static) == 5
    assert len(boundary) == 5
    assert len(dyn) == 5
    assert len(target) == 5
    assert len(cell_area) == 5
    assert all(area is None for area in cell_area)


@pytest.mark.parametrize("device", _DEVICES)
def test_collect_all_fields_without_target(device):
    """Test collecting fields without target."""
    mock_dataset = [
        {
            "geometry": torch.rand(100, 2).to(device),
            "static": torch.rand(100, 7).to(device),
            "boundary": torch.rand(3, 100, 1).to(device),
            "dynamic": torch.rand(3, 100, 3).to(device),
        }
        for _ in range(5)
    ]

    result = collect_all_fields(mock_dataset, expect_target=False)

    assert len(result) == 6
    assert len(result[4]) == 5
    assert all(target is None for target in result[4])
    assert len(result[5]) == 5
    assert all(area is None for area in result[5])


@pytest.mark.parametrize("device", _DEVICES)
def test_collect_all_fields_with_cell_area(device):
    """Test collecting fields with cell_area."""
    mock_dataset = [
        {
            "geometry": torch.rand(100, 2).to(device),
            "static": torch.rand(100, 7).to(device),
            "boundary": torch.rand(3, 100, 1).to(device),
            "dynamic": torch.rand(3, 100, 3).to(device),
            "target": torch.rand(100, 3).to(device),
            "cell_area": torch.rand(100).to(device),
        }
        for _ in range(5)
    ]

    result = collect_all_fields(mock_dataset, expect_target=True)

    assert len(result) == 6  # Includes cell_area
    assert len(result[5]) == 5  # 5 cell_area tensors


@pytest.mark.parametrize("device", _DEVICES)
def test_stack_and_fit_transform_creates_normalizers(device):
    """Test that stack_and_fit_transform creates normalizers."""
    geom = [torch.rand(100, 2).to(device) for _ in range(5)]
    static = [torch.rand(100, 7).to(device) for _ in range(5)]
    boundary = [torch.rand(3, 100, 1).to(device) for _ in range(5)]
    dyn = [torch.rand(3, 100, 3).to(device) for _ in range(5)]
    target = [torch.rand(100, 3).to(device) for _ in range(5)]

    normalizers, big_tensors = stack_and_fit_transform(geom, static, boundary, dyn, target)

    assert "static" in normalizers
    assert "boundary" in normalizers
    assert "target" in normalizers
    assert "dynamic" in normalizers
    assert normalizers["dynamic"] is normalizers["target"]

    assert big_tensors["geometry"].shape[0] == 5
    assert big_tensors["static"].shape[0] == 5
    assert torch.allclose(big_tensors["geometry"], torch.stack(geom, dim=0))


@pytest.mark.parametrize("device", _DEVICES)
def test_stack_and_fit_transform_uses_existing(device):
    """Test using existing normalizers."""
    geom = [torch.rand(100, 2).to(device) for _ in range(5)]
    static = [torch.rand(100, 7).to(device) for _ in range(5)]
    boundary = [torch.rand(3, 100, 1).to(device) for _ in range(5)]
    dyn = [torch.rand(3, 100, 3).to(device) for _ in range(5)]
    target = [torch.rand(100, 3).to(device) for _ in range(5)]

    # First pass - fit normalizers
    normalizers, _ = stack_and_fit_transform(
        geom, static, boundary, dyn, target, fit_normalizers=True
    )

    # Second pass - use existing
    new_geom = [torch.rand(100, 2).to(device) for _ in range(3)]
    new_static = [torch.rand(100, 7).to(device) for _ in range(3)]
    new_boundary = [torch.rand(3, 100, 1).to(device) for _ in range(3)]
    new_dyn = [torch.rand(3, 100, 3).to(device) for _ in range(3)]
    new_target = [torch.rand(100, 3).to(device) for _ in range(3)]

    _, new_tensors = stack_and_fit_transform(
        new_geom,
        new_static,
        new_boundary,
        new_dyn,
        new_target,
        normalizers=normalizers,
        fit_normalizers=False,
    )

    assert new_tensors["geometry"].shape[0] == 3


@pytest.mark.parametrize("device", _DEVICES)
def test_transform_with_existing_normalizers(device):
    """Test transform_with_existing_normalizers."""
    # Create and fit normalizers
    geom = [torch.rand(100, 2).to(device) for _ in range(5)]
    static = [torch.rand(100, 7).to(device) for _ in range(5)]
    boundary = [torch.rand(3, 100, 1).to(device) for _ in range(5)]
    dyn = [torch.rand(3, 100, 3).to(device) for _ in range(5)]
    target = [torch.rand(100, 3).to(device) for _ in range(5)]

    normalizers, _ = stack_and_fit_transform(geom, static, boundary, dyn, target)

    # Transform new data
    new_geom = [torch.rand(100, 2).to(device) for _ in range(3)]
    new_static = [torch.rand(100, 7).to(device) for _ in range(3)]
    new_boundary = [torch.rand(3, 100, 1).to(device) for _ in range(3)]
    new_dyn = [torch.rand(3, 100, 3).to(device) for _ in range(3)]

    transformed = transform_with_existing_normalizers(
        new_geom, new_static, new_boundary, new_dyn, normalizers
    )

    assert "geometry" in transformed
    assert "static" in transformed
    assert "boundary" in transformed
    assert "dynamic" in transformed

    assert transformed["geometry"].shape[0] == 3
    assert torch.allclose(transformed["geometry"], torch.stack(new_geom, dim=0))


@pytest.mark.parametrize("device", _DEVICES)
def test_state_normalizer_is_per_channel_and_shared(device):
    """Dynamic history and targets should share one per-channel normalizer."""
    geom = [torch.rand(4, 2).to(device) for _ in range(2)]
    static = [torch.rand(4, 1).to(device) for _ in range(2)]
    boundary = [torch.rand(3, 4, 1).to(device) for _ in range(2)]
    dyn = [
        torch.tensor(
            [
                [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0], [4.0, 40.0, 400.0]],
                [[5.0, 50.0, 500.0], [6.0, 60.0, 600.0], [7.0, 70.0, 700.0], [8.0, 80.0, 800.0]],
                [[9.0, 90.0, 900.0], [10.0, 100.0, 1000.0], [11.0, 110.0, 1100.0], [12.0, 120.0, 1200.0]],
            ],
            device=device,
        )
        for _ in range(2)
    ]
    target = [
        torch.tensor(
            [[13.0, 130.0, 1300.0], [14.0, 140.0, 1400.0], [15.0, 150.0, 1500.0], [16.0, 160.0, 1600.0]],
            device=device,
        )
        for _ in range(2)
    ]

    normalizers, transformed = stack_and_fit_transform(geom, static, boundary, dyn, target)

    assert normalizers["dynamic"] is normalizers["target"]
    assert tuple(normalizers["target"].mean.shape) == (1, 3)
    restored_dynamic = normalizers["dynamic"].inverse_transform(transformed["dynamic"])
    restored_target = normalizers["target"].inverse_transform(transformed["target"])
    assert torch.allclose(restored_dynamic, torch.stack(dyn, dim=0))
    assert torch.allclose(restored_target, torch.stack(target, dim=0))


def test_checkpoint_epoch_resolution(tmp_path):
    """Best/latest checkpoint resolution should use checkpoint files and the sidecar."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    torch.save({"epoch": 2}, checkpoint_dir / "checkpoint.0.2.pt")
    torch.save({"epoch": 5}, checkpoint_dir / "checkpoint.0.5.pt")

    write_best_checkpoint_metadata(
        checkpoint_dir,
        stage="pretrain",
        epoch=2,
        metric_name="val_l2",
        metric_value=0.1,
        models=None,
    )

    assert resolve_checkpoint_epoch(checkpoint_dir, "latest") == 5
    assert resolve_checkpoint_epoch(checkpoint_dir, "best") == 2
    assert resolve_checkpoint_epoch(checkpoint_dir, 7) == 7


def test_seed_everything_is_reproducible():
    """Repeated seeding should reproduce Python, NumPy, and Torch draws."""
    seed_everything(1234, rank=0)
    first_random = random.random()
    first_numpy = float(np.random.rand())
    first_torch = torch.rand(4)

    seed_everything(1234, rank=0)
    second_random = random.random()
    second_numpy = float(np.random.rand())
    second_torch = torch.rand(4)

    seed_everything(1234, rank=1)
    rank_shifted_random = random.random()

    assert first_random == second_random
    assert first_numpy == second_numpy
    assert torch.allclose(first_torch, second_torch)
    assert rank_shifted_random != first_random


def test_split_dataset_is_deterministic():
    """random_split wrappers should be deterministic for a fixed seed and offset."""
    dataset = TensorDataset(torch.arange(12))

    first_a, second_a = split_dataset(dataset, [8, 4], seed=99, offset=7)
    first_b, second_b = split_dataset(dataset, [8, 4], seed=99, offset=7)
    first_c, second_c = split_dataset(dataset, [8, 4], seed=99, offset=8)

    assert first_a.indices == first_b.indices
    assert second_a.indices == second_b.indices
    assert first_a.indices != first_c.indices
    assert second_a.indices != second_c.indices


def test_create_loader_from_config_and_set_loader_epoch():
    """Loader helpers should respect config values and forward sampler epochs."""

    class LoaderConfig:
        batch_size = 2
        num_workers = 0
        pin_memory = False
        persistent_workers = True

    dataset = TensorDataset(torch.arange(5))
    loader = create_loader_from_config(dataset, LoaderConfig(), shuffle=True)

    assert loader.batch_size == 2
    assert loader.drop_last is False
    assert loader.num_workers == 0
    assert loader.pin_memory is False

    class _Sampler:
        def __init__(self):
            self.epochs = []

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    fake_loader = type("FakeLoader", (), {"sampler": _Sampler()})()
    set_loader_epoch(fake_loader, 3)

    assert fake_loader.sampler.epochs == [3]


def test_checkpoint_validation_and_legacy_resolution(tmp_path, monkeypatch):
    """Checkpoint helpers should validate required files and legacy fallback names."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    model = nn.Linear(2, 2)

    import types
    from utils import checkpointing as checkpointing_module

    physicsnemo_models = getattr(checkpointing_module.physicsnemo, "models", None)
    if physicsnemo_models is None or not hasattr(physicsnemo_models, "Module"):
        shim = types.SimpleNamespace(
            Module=getattr(checkpointing_module.physicsnemo, "Module", nn.Module)
        )
        monkeypatch.setattr(checkpointing_module.physicsnemo, "models", shim, raising=False)

    torch.save(model.state_dict(), checkpoint_dir / "Linear.0.3.pt")
    torch.save({"epoch": 3}, checkpoint_dir / "checkpoint.0.3.pt")

    validation = validate_checkpoint_files(checkpoint_dir, model, 3)
    assert validation["epoch"] == 3
    assert validation["model_files"] == ["Linear.0.3.pt"]
    assert validation["training_state_file"] == "checkpoint.0.3.pt"

    with pytest.raises(FileNotFoundError):
        validate_checkpoint_files(checkpoint_dir, model, 4)

    torch.save({}, checkpoint_dir / "model_state_dict.pt")
    torch.save({}, checkpoint_dir / "best_model_state_dict.pt")

    assert resolve_legacy_neuralop_checkpoint_name(checkpoint_dir, "latest") == "model"
    assert resolve_legacy_neuralop_checkpoint_name(checkpoint_dir, "best") == "best_model"

