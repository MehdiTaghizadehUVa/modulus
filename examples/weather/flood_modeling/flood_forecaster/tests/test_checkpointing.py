# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest
import torch
from neuralop import get_model
from omegaconf import OmegaConf


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
REPO_ROOT = EXAMPLE_ROOT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import CNNDomainClassifier, GINOWrapper, ImportableTorchModuleAdapter  # noqa: E402
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from tests.model_fixtures import FakeGINOBackbone  # noqa: E402
from utils.checkpointing import (  # noqa: E402
    expected_model_files,
    resolve_checkpoint_epoch,
    validate_checkpoint_files,
    write_best_checkpoint_metadata,
)


def _make_classifier(seed: int = 0) -> CNNDomainClassifier:
    torch.manual_seed(seed)
    return CNNDomainClassifier(
        in_channels=4,
        lambda_max=0.7,
        da_cfg={
            "conv_layers": [{"out_channels": 5, "kernel_size": 3, "pool_size": 2}],
            "fc_dim": 2,
        },
    )


def _build_real_gino_wrapper(seed: int = 0) -> GINOWrapper:
    cfg = OmegaConf.load(EXAMPLE_ROOT / "conf" / "config.yaml")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    autoregressive = bool(model_cfg.pop("autoregressive", False))
    torch.manual_seed(seed)
    gino_model = get_model(OmegaConf.create({"model": model_cfg}))
    return GINOWrapper(gino_model, autoregressive=autoregressive)


def test_expected_model_files_match_single_model_checkpoint(tmp_path: Path):
    model = GINOWrapper(FakeGINOBackbone())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_checkpoint(
        path=str(tmp_path),
        models=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=7,
        metadata={"stage": "pretrain"},
    )

    expected_files = expected_model_files(tmp_path, model, 7)
    assert expected_files == ["GINOWrapper.0.7.mdlus"]
    assert (tmp_path / expected_files[0]).exists()
    assert (tmp_path / "checkpoint.0.7.pt").exists()
    assert validate_checkpoint_files(tmp_path, model, 7)["model_files"] == expected_files


def test_expected_model_files_match_duplicate_model_checkpoint(tmp_path: Path):
    classifier_a = _make_classifier(seed=1)
    classifier_b = _make_classifier(seed=2)
    optimizer = torch.optim.Adam(
        list(classifier_a.parameters()) + list(classifier_b.parameters()),
        lr=1e-3,
    )

    save_checkpoint(
        path=str(tmp_path),
        models=[classifier_a, classifier_b],
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=7,
        metadata={"stage": "adapt"},
    )

    expected_files = expected_model_files(tmp_path, [classifier_a, classifier_b], 7)
    assert expected_files == [
        "CNNDomainClassifier0.0.7.mdlus",
        "CNNDomainClassifier1.0.7.mdlus",
    ]
    for file_name in expected_files:
        assert (tmp_path / file_name).exists()
    assert validate_checkpoint_files(tmp_path, [classifier_a, classifier_b], 7)["model_files"] == expected_files


def test_best_checkpoint_sidecar_uses_class_name_files(tmp_path: Path):
    model = GINOWrapper(FakeGINOBackbone())
    classifier = _make_classifier()

    sidecar_path = write_best_checkpoint_metadata(
        tmp_path,
        stage="adapt",
        epoch=7,
        metric_name="target_val",
        metric_value=0.25,
        models=[model, classifier],
    )

    payload = OmegaConf.to_container(OmegaConf.load(sidecar_path), resolve=True)
    assert payload["model_files"] == [
        "GINOWrapper.0.7.mdlus",
        "CNNDomainClassifier.0.7.mdlus",
    ]
    assert payload["training_state_file"] == "checkpoint.0.7.pt"


def test_validate_checkpoint_files_fails_fast_on_missing_native_model(tmp_path: Path):
    model = GINOWrapper(FakeGINOBackbone())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_checkpoint(
        path=str(tmp_path),
        models=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=5,
        metadata={"stage": "pretrain"},
    )
    missing_file = tmp_path / expected_model_files(tmp_path, model, 5)[0]
    missing_file.unlink()

    with pytest.raises(FileNotFoundError, match="missing required files"):
        validate_checkpoint_files(tmp_path, model, 5)


def test_real_gino_native_checkpoint_roundtrip(tmp_path: Path):
    model = _build_real_gino_wrapper(seed=0)
    assert isinstance(model.model, ImportableTorchModuleAdapter)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(
        path=str(tmp_path),
        models=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=3,
        metadata={"stage": "pretrain"},
    )

    expected_files = expected_model_files(tmp_path, model, 3)
    assert expected_files == ["GINOWrapper.0.3.mdlus"]
    assert resolve_checkpoint_epoch(tmp_path, "latest") == 3
    validate_checkpoint_files(tmp_path, model, 3)

    reloaded_model = _build_real_gino_wrapper(seed=1)
    load_checkpoint(
        path=str(tmp_path),
        models=reloaded_model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=3,
        metadata_dict={},
        device="cpu",
    )

    original_state = model.state_dict()
    reloaded_state = reloaded_model.state_dict()
    assert original_state.keys() == reloaded_state.keys()
    for key in original_state:
        if torch.is_tensor(original_state[key]):
            torch.testing.assert_close(reloaded_state[key], original_state[key])
        else:
            assert reloaded_state[key] == original_state[key]
