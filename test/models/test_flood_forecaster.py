# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Unified FloodForecaster tests.

This module intentionally keeps all collected FloodForecaster unit and smoke tests
in one place. Reference assets and toy model fixtures remain under the example's
``tests/`` package so binary fixtures are not embedded in the test module.

Following rule MOD-008a, this module keeps constructor and public-attribute tests
for FloodForecaster's custom PhysicsNeMo modules. Following rule MOD-008b, it
keeps non-regression tests with reference ``.pth`` data. Following rule
MOD-008c, it keeps checkpoint-loading tests for ``.mdlus`` serialization.
"""

from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import types
from unittest.mock import MagicMock

from neuralop import get_model
from neuralop.losses import LpLoss
from omegaconf import OmegaConf
import physicsnemo
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _load_example_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if "training" not in sys.modules:
    training_pkg = types.ModuleType("training")
    training_pkg.__path__ = [str(EXAMPLE_ROOT / "training")]
    sys.modules["training"] = training_pkg

trainer_module = _load_example_module("training.trainer", EXAMPLE_ROOT / "training" / "trainer.py")
pretraining_module = _load_example_module("training.pretraining", EXAMPLE_ROOT / "training" / "pretraining.py")
domain_adaptation_module = _load_example_module(
    "training.domain_adaptation", EXAMPLE_ROOT / "training" / "domain_adaptation.py"
)
inference_script_module = _load_example_module(
    "flood_forecaster_inference_script", EXAMPLE_ROOT / "inference.py"
)
training_init = _load_example_module("training", EXAMPLE_ROOT / "training" / "__init__.py")
sys.modules["training"].__dict__.update(training_init.__dict__)

for name in dir(domain_adaptation_module):
    obj = getattr(domain_adaptation_module, name)
    if isinstance(obj, type) and issubclass(obj, (torch.nn.Module, physicsnemo.Module)):
        obj.__module__ = "training.domain_adaptation"

from datasets import (  # noqa: E402
    FloodDatasetWithQueryPoints,
    FloodRolloutTestDatasetNew,
    LazyNormalizedDataset,
    LazyNormalizedRolloutDataset,
    prepare_flood_cache,
)
from data_processing import FloodGINODataProcessor, LpLossWrapper  # noqa: E402
from models import CNNDomainClassifier, GINOWrapper, ImportableTorchModuleAdapter  # noqa: E402
from models.domain_classifier import GradientReversal, GradientReversalFunction  # noqa: E402
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from tests.model_fixtures import FakeGINOBackbone  # noqa: E402
from training.domain_adaptation import DomainAdaptationTrainer  # noqa: E402
from training.pretraining import create_scheduler  # noqa: E402
from training.trainer import NeuralOperatorTrainer  # noqa: E402
from utils.checkpointing import (  # noqa: E402
    expected_model_files,
    resolve_checkpoint_epoch,
    validate_checkpoint_files,
    write_best_checkpoint_metadata,
)
from utils.normalization import fit_normalizers_from_sample_index  # noqa: E402
from utils.runtime import create_loader_from_config, resolve_eval_interval  # noqa: E402


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")

FIXTURE_DIR = EXAMPLE_ROOT / "tests" / "data"
STATIC_FILES = [
    "M40_XY.txt",
    "M40_CA.txt",
    "M40_CE.txt",
    "M40_CS.txt",
    "M40_FA.txt",
    "M40_A.txt",
    "M40_CU.txt",
]
DYNAMIC_PATTERNS = {
    "WD": "M40_WD_{}.txt",
    "VX": "M40_VX_{}.txt",
    "VY": "M40_VY_{}.txt",
}
BOUNDARY_PATTERNS = {"inflow": "M40_US_InF_{}.txt"}



# ---- Checkpointing Coverage ----

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

def test_cnn_domain_classifier_checkpoint_accepts_omegaconf_config(tmp_path: Path):
    """Hydra DictConfig inputs should not leak into PhysicsNeMo .mdlus args."""
    da_cfg = OmegaConf.create(
        {
            "conv_layers": [{"out_channels": 5, "kernel_size": 3, "pool_size": 1}],
            "fc_dim": 2,
        }
    )
    classifier = CNNDomainClassifier(
        in_channels=4,
        da_cfg=da_cfg,
        lambda_max=0.5,
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    save_checkpoint(
        path=str(tmp_path),
        models=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=2,
    )

    expected_files = expected_model_files(tmp_path, classifier, 2)
    assert expected_files == ["CNNDomainClassifier.0.2.mdlus"]
    assert (tmp_path / expected_files[0]).exists()
    validate_checkpoint_files(tmp_path, classifier, 2)

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


# ---- Data Loading Coverage ----

def _copy_tree(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst

def _make_source_root(tmp_path: Path) -> Path:
    source_root = EXAMPLE_ROOT / "smoke_data" / "source"
    return _copy_tree(source_root, tmp_path / "source")

def _make_rollout_root(tmp_path: Path) -> Path:
    rollout_root = _copy_tree(EXAMPLE_ROOT / "smoke_data" / "source", tmp_path / "rollout")
    shutil.copy2(rollout_root / "train.txt", rollout_root / "test.txt")
    return rollout_root

def _build_train_dataset(
    root: Path,
    *,
    backend: str,
    noise_type: str = "none",
    noise_std=None,
) -> FloodDatasetWithQueryPoints:
    return FloodDatasetWithQueryPoints(
        data_root=root,
        n_history=3,
        xy_file="M40_XY.txt",
        query_res=[48, 48],
        static_files=STATIC_FILES,
        dynamic_patterns=DYNAMIC_PATTERNS,
        boundary_patterns=BOUNDARY_PATTERNS,
        raise_on_smaller=True,
        skip_before_timestep=0,
        noise_type=noise_type,
        noise_std=noise_std,
        backend=backend,
        cache_dir_name=".flood_cache",
        rebuild_cache=False,
        run_cache_size=4,
    )

def _build_rollout_dataset(root: Path, *, backend: str) -> FloodRolloutTestDatasetNew:
    return FloodRolloutTestDatasetNew(
        rollout_data_root=root,
        n_history=3,
        rollout_length=5,
        xy_file="M40_XY.txt",
        query_res=[48, 48],
        static_files=STATIC_FILES,
        dynamic_patterns=DYNAMIC_PATTERNS,
        boundary_patterns=BOUNDARY_PATTERNS,
        raise_on_smaller=True,
        skip_before_timestep=0,
        backend=backend,
        cache_dir_name=".flood_cache",
        rebuild_cache=False,
        run_cache_size=4,
    )

def _manual_fit_normalizers(dataset):
    static_samples = []
    boundary_samples = []
    state_samples = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        static_samples.append(sample["static"])
        boundary_samples.append(sample["boundary"])
        state_samples.append(sample["dynamic"])
        state_samples.append(sample["target"])

    static_big = torch.stack(static_samples, dim=0)
    boundary_big = torch.stack(boundary_samples, dim=0)
    state_big = torch.cat(
        [tensor.reshape(-1, tensor.shape[-1]) for tensor in state_samples],
        dim=0,
    )

    def stats(tensor: torch.Tensor):
        flat = tensor.reshape(-1, tensor.shape[-1]).to(torch.float64)
        mean = flat.mean(dim=0).to(torch.float32)
        std = flat.var(dim=0, unbiased=False).sqrt().to(torch.float32)
        return mean, std

    static_mean, static_std = stats(static_big)
    boundary_mean, boundary_std = stats(boundary_big)
    state_mean, state_std = stats(state_big)
    return {
        "static": (static_mean, static_std),
        "boundary": (boundary_mean, boundary_std),
        "dynamic": (state_mean, state_std),
        "target": (state_mean, state_std),
    }

def _manual_fit_normalizers_from_sample_index(dataset_or_subset, *, apply_noise: bool):
    base_dataset = dataset_or_subset.dataset if hasattr(dataset_or_subset, "dataset") else dataset_or_subset
    selected_indices = dataset_or_subset.indices if hasattr(dataset_or_subset, "indices") else range(len(dataset_or_subset))

    static_samples = [base_dataset.static_data]
    boundary_samples = []
    state_samples = []

    for sample_idx in selected_indices:
        run_id, target_t = base_dataset.sample_index[sample_idx]
        sample = base_dataset.get_sample_components(run_id, target_t, apply_noise=apply_noise)
        boundary_samples.append(sample["boundary"])
        state_samples.append(sample["dynamic"])
        state_samples.append(sample["target"])

    static_big = torch.stack(static_samples, dim=0)
    boundary_big = torch.stack(boundary_samples, dim=0)
    state_big = torch.cat(
        [tensor.reshape(-1, tensor.shape[-1]) for tensor in state_samples],
        dim=0,
    )

    def stats(tensor: torch.Tensor):
        flat = tensor.reshape(-1, tensor.shape[-1]).to(torch.float64)
        mean = flat.mean(dim=0).to(torch.float32)
        std = flat.var(dim=0, unbiased=False).sqrt().to(torch.float32)
        return mean, std

    static_mean, static_std = stats(static_big)
    boundary_mean, boundary_std = stats(boundary_big)
    state_mean, state_std = stats(state_big)
    return {
        "static": (static_mean, static_std),
        "boundary": (boundary_mean, boundary_std),
        "dynamic": (state_mean, state_std),
        "target": (state_mean, state_std),
    }

def test_cache_build_and_reuse(tmp_path):
    root = _make_source_root(tmp_path)
    manifest = prepare_flood_cache(
        root,
        list_file_name="train.txt",
        xy_file="M40_XY.txt",
        static_files=STATIC_FILES,
        dynamic_patterns=DYNAMIC_PATTERNS,
        boundary_patterns=BOUNDARY_PATTERNS,
        rebuild=False,
    )
    cache_dir = root / ".flood_cache"
    manifest_path = cache_dir / "manifest.json"
    cache_path = cache_dir / "flood_forecaster_v1.h5"

    assert cache_path.exists()
    assert manifest_path.exists()
    original_mtime = cache_path.stat().st_mtime_ns
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    time.sleep(0.02)

    reused_manifest = prepare_flood_cache(
        root,
        list_file_name="train.txt",
        xy_file="M40_XY.txt",
        static_files=STATIC_FILES,
        dynamic_patterns=DYNAMIC_PATTERNS,
        boundary_patterns=BOUNDARY_PATTERNS,
        rebuild=False,
    )

    assert manifest == reused_manifest
    assert cache_path.stat().st_mtime_ns == original_mtime
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original_manifest

def test_manifest_invalidation_when_tracked_files_change(tmp_path):
    root = _make_source_root(tmp_path)
    prepare_flood_cache(
        root,
        list_file_name="train.txt",
        xy_file="M40_XY.txt",
        static_files=STATIC_FILES,
        dynamic_patterns=DYNAMIC_PATTERNS,
        boundary_patterns=BOUNDARY_PATTERNS,
        rebuild=False,
    )
    cache_path = root / ".flood_cache" / "flood_forecaster_v1.h5"
    baseline_mtime = cache_path.stat().st_mtime_ns

    time.sleep(0.02)
    train_txt = root / "train.txt"
    train_txt.write_text(train_txt.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    prepare_flood_cache(
        root,
        list_file_name="train.txt",
        xy_file="M40_XY.txt",
        static_files=STATIC_FILES,
        dynamic_patterns=DYNAMIC_PATTERNS,
        boundary_patterns=BOUNDARY_PATTERNS,
        rebuild=False,
    )
    updated_mtime = cache_path.stat().st_mtime_ns
    assert updated_mtime > baseline_mtime

    time.sleep(0.02)
    dynamic_file = root / "M40_WD_H1.txt"
    dynamic_file.write_text(dynamic_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    prepare_flood_cache(
        root,
        list_file_name="train.txt",
        xy_file="M40_XY.txt",
        static_files=STATIC_FILES,
        dynamic_patterns=DYNAMIC_PATTERNS,
        boundary_patterns=BOUNDARY_PATTERNS,
        rebuild=False,
    )
    assert cache_path.stat().st_mtime_ns > updated_mtime

def test_training_dataset_backend_parity(tmp_path):
    root = _make_source_root(tmp_path)
    raw_dataset = _build_train_dataset(root, backend="raw_txt")
    cached_dataset = _build_train_dataset(root, backend="auto")

    assert len(raw_dataset) == len(cached_dataset)
    raw_sample = raw_dataset[0]
    cached_sample = cached_dataset[0]

    assert raw_sample["run_id"] == cached_sample["run_id"]
    assert raw_sample["time_index"] == cached_sample["time_index"]
    for key in ["geometry", "static", "boundary", "dynamic", "target"]:
        assert torch.allclose(raw_sample[key], cached_sample[key])

def test_training_dataset_returns_compact_boundary_history(tmp_path):
    root = _make_source_root(tmp_path)
    cached_dataset = _build_train_dataset(root, backend="auto")
    run_id, target_t = cached_dataset.sample_index[0]
    sample = cached_dataset[0]
    run_data = cached_dataset.run_store.load_run(run_id)
    t0 = target_t - cached_dataset.n_history
    expected_boundary = run_data["boundary"][t0:target_t]
    assert torch.allclose(sample["boundary"], expected_boundary)
    assert sample["boundary"].shape == (
        cached_dataset.n_history,
        len(cached_dataset.required_boundary_keys),
    )

def test_rollout_dataset_backend_parity(tmp_path):
    root = _make_rollout_root(tmp_path)
    raw_dataset = _build_rollout_dataset(root, backend="raw_txt")
    cached_dataset = _build_rollout_dataset(root, backend="auto")

    assert len(raw_dataset) == len(cached_dataset)
    raw_sample = raw_dataset[0]
    cached_sample = cached_dataset[0]

    assert raw_sample["run_id"] == cached_sample["run_id"]
    for key in ["geometry", "static", "boundary", "dynamic"]:
        assert torch.allclose(raw_sample[key], cached_sample[key])
    assert torch.allclose(raw_sample["cell_area"], cached_sample["cell_area"])

def test_data_processor_expands_compact_boundary_when_building_x(tmp_path):
    root = _make_source_root(tmp_path)
    dataset = _build_train_dataset(root, backend="auto")
    subset = torch.utils.data.Subset(dataset, list(range(min(6, len(dataset)))))
    normalizers = fit_normalizers_from_sample_index(subset)
    normalized_dataset = LazyNormalizedDataset(subset, normalizers=normalizers, query_res=[48, 48])
    sample = normalized_dataset[0]

    processor = FloodGINODataProcessor(device="cpu")
    processed = processor.preprocess(sample)

    static_dim = sample["static"].shape[-1]
    boundary_dim = sample["boundary"].shape[0] * sample["boundary"].shape[1]
    expected_boundary = sample["boundary"].unsqueeze(1).expand(
        -1,
        sample["static"].shape[0],
        -1,
    )
    expected_boundary = expected_boundary.permute(1, 0, 2).reshape(
        1,
        sample["static"].shape[0],
        boundary_dim,
    )

    assert processed["x"].shape[0] == 1
    assert torch.allclose(
        processed["x"][:, :, static_dim : static_dim + boundary_dim],
        expected_boundary,
    )

def test_lazy_normalized_loader_collates_shared_fields_once(tmp_path):
    root = _make_source_root(tmp_path)
    dataset = _build_train_dataset(root, backend="auto")
    subset = torch.utils.data.Subset(dataset, list(range(min(6, len(dataset)))))
    normalizers = fit_normalizers_from_sample_index(subset)
    normalized_dataset = LazyNormalizedDataset(subset, normalizers=normalizers, query_res=[48, 48])

    loader = create_loader_from_config(
        normalized_dataset,
        type(
            "Cfg",
            (),
            {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
            },
        )(),
        shuffle=False,
    )
    batch = next(iter(loader))

    assert batch["geometry"].shape == normalized_dataset.shared_geometry.shape
    assert batch["static"].shape == normalized_dataset.shared_static.shape
    assert batch["query_points"].shape == normalized_dataset.shared_query_points.shape
    assert batch["dynamic"].shape[0] == 2
    assert batch["boundary"].shape[0] == 2
    assert batch["target"].shape[0] == 2

    processor = FloodGINODataProcessor(device="cpu")
    processed = processor.preprocess(batch)
    assert processed["input_geom"].shape == normalized_dataset.shared_geometry.shape
    assert processed["latent_queries"].shape == normalized_dataset.shared_query_points.shape
    assert processed["x"].shape[0] == 2

def test_grouped_normalizer_fit_matches_manual_sample_loop(tmp_path):
    root = _make_source_root(tmp_path)
    dataset = _build_train_dataset(root, backend="auto")
    subset_indices = list(range(min(6, len(dataset))))
    subset = torch.utils.data.Subset(dataset, subset_indices)

    grouped = fit_normalizers_from_sample_index(subset)
    manual = _manual_fit_normalizers(subset)

    for key in ["static", "boundary", "dynamic", "target"]:
        expected_mean, expected_std = manual[key]
        assert torch.allclose(grouped[key].mean.cpu(), expected_mean, atol=1e-6, rtol=1e-5)
        assert torch.allclose(grouped[key].std.cpu(), expected_std, atol=1e-6, rtol=1e-5)

def test_normalizer_fit_ignores_noise_augmentation(tmp_path):
    root = _make_source_root(tmp_path)
    dataset = _build_train_dataset(
        root,
        backend="auto",
        noise_type="only_last",
        noise_std=[0.25, 0.1, 0.1],
    )
    subset = torch.utils.data.Subset(dataset, list(range(min(6, len(dataset)))))

    grouped = fit_normalizers_from_sample_index(subset)
    manual_clean = _manual_fit_normalizers_from_sample_index(subset, apply_noise=False)
    manual_noisy = _manual_fit_normalizers_from_sample_index(subset, apply_noise=True)

    for key in ["static", "boundary", "dynamic", "target"]:
        clean_mean, clean_std = manual_clean[key]
        _, noisy_std = manual_noisy[key]
        assert torch.allclose(grouped[key].mean.cpu(), clean_mean, atol=1e-6, rtol=1e-5)
        assert torch.allclose(grouped[key].std.cpu(), clean_std, atol=1e-6, rtol=1e-5)
        if key in {"dynamic", "target"}:
            assert not torch.allclose(clean_std, noisy_std)

def test_lazy_normalized_dataset_noise_flag_controls_reload_path(tmp_path):
    root = _make_source_root(tmp_path)
    dataset = _build_train_dataset(
        root,
        backend="auto",
        noise_type="only_last",
        noise_std=[0.5, 0.25, 0.25],
    )
    subset = torch.utils.data.Subset(dataset, [0])
    normalizers = fit_normalizers_from_sample_index(subset)
    run_id, target_t = dataset.sample_index[0]

    clean_wrapper = LazyNormalizedDataset(
        subset,
        normalizers=normalizers,
        query_res=[48, 48],
        apply_noise=False,
    )
    noisy_wrapper = LazyNormalizedDataset(
        subset,
        normalizers=normalizers,
        query_res=[48, 48],
        apply_noise=True,
    )

    clean_raw = dataset.get_sample_components(run_id, target_t, apply_noise=False)
    clean_expected = normalizers["dynamic"].transform(clean_raw["dynamic"])

    torch.manual_seed(7)
    noisy_sample = noisy_wrapper[0]["dynamic"]
    torch.manual_seed(7)
    noisy_raw = dataset.get_sample_components(run_id, target_t, apply_noise=True)
    noisy_expected = normalizers["dynamic"].transform(noisy_raw["dynamic"])

    clean_sample = clean_wrapper[0]["dynamic"]
    assert torch.allclose(clean_sample, clean_expected)
    assert torch.allclose(noisy_sample, noisy_expected)
    assert not torch.allclose(clean_sample, noisy_sample)

def test_lazy_rollout_normalization_matches_manual_transform(tmp_path):
    source_root = _make_source_root(tmp_path / "train")
    rollout_root = _make_rollout_root(tmp_path / "rollout")

    train_dataset = _build_train_dataset(source_root, backend="auto")
    subset = torch.utils.data.Subset(train_dataset, list(range(min(6, len(train_dataset)))))
    normalizers = fit_normalizers_from_sample_index(subset)

    rollout_dataset = _build_rollout_dataset(rollout_root, backend="auto")
    lazy_dataset = LazyNormalizedRolloutDataset(rollout_dataset, normalizers=normalizers, query_res=[48, 48])

    raw_sample = rollout_dataset[0]
    normalized_sample = lazy_dataset[0]

    assert normalized_sample["run_id"] == raw_sample["run_id"]
    assert torch.allclose(normalized_sample["geometry"], raw_sample["geometry"])
    assert torch.allclose(normalized_sample["static"], normalizers["static"].transform(raw_sample["static"]))
    assert torch.allclose(normalized_sample["boundary"], normalizers["boundary"].transform(raw_sample["boundary"]))
    assert torch.allclose(normalized_sample["dynamic"], normalizers["dynamic"].transform(raw_sample["dynamic"]))
    assert torch.allclose(normalized_sample["cell_area"], raw_sample["cell_area"])

def test_resolve_eval_interval_prefers_training_config():
    cfg = type(
        "Cfg",
        (),
        {
            "training": type("Training", (), {"get": lambda self, key, default=None: {"eval_interval": 5}.get(key, default)})(),
            "wandb": type("Wandb", (), {"get": lambda self, key, default=None: {"eval_interval": 2}.get(key, default)})(),
        },
    )()
    assert resolve_eval_interval(cfg) == 5

def test_resolve_eval_interval_falls_back_to_wandb_alias():
    cfg = type(
        "Cfg",
        (),
        {
            "training": type("Training", (), {"get": lambda self, key, default=None: {}.get(key, default)})(),
            "wandb": type("Wandb", (), {"get": lambda self, key, default=None: {"eval_interval": 3}.get(key, default)})(),
        },
    )()
    assert resolve_eval_interval(cfg) == 3

def test_resolve_eval_interval_rejects_non_positive_values():
    cfg = type(
        "Cfg",
        (),
        {
            "training": type("Training", (), {"get": lambda self, key, default=None: {"eval_interval": 0}.get(key, default)})(),
        },
    )()
    try:
        resolve_eval_interval(cfg)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("resolve_eval_interval should reject non-positive values")

def test_domain_adaptation_weight_config_defaults():
    """Full/default configs should enable adversarial DA, while short/smoke stay cheap."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(EXAMPLE_ROOT / "conf"), version_base="1.3"):
            base_cfg = compose(config_name="config")
            short_cfg = compose(config_name="config_short")
            smoke_cfg = compose(config_name="config_smoke")
    finally:
        GlobalHydra.instance().clear()

    assert base_cfg.training.da_class_loss_weight == pytest.approx(0.1)
    assert short_cfg.training.da_class_loss_weight == pytest.approx(0.0)
    assert smoke_cfg.training.da_class_loss_weight == pytest.approx(0.0)
    assert smoke_cfg.source_data.rollout_length == 4
    smoke_cfg_unresolved = OmegaConf.to_container(smoke_cfg, resolve=False)
    assert smoke_cfg_unresolved["rollout_data"]["root"].endswith("smoke_data/target")
    assert smoke_cfg_unresolved["rollout_data"]["list_file_name"] == "train.txt"

def test_inference_normalizer_fallback_uses_full_source_train_split(monkeypatch):
    """Inference fallback should refit on the deterministic pretraining split, not a 100-sample slice."""
    captured = {}

    class FakeSourceDataset:
        def __init__(self, **kwargs):
            captured["dataset_kwargs"] = kwargs

        def __len__(self):
            return 10

    def fake_split_dataset(dataset, lengths, seed, offset):
        captured["split"] = {
            "dataset": dataset,
            "lengths": lengths,
            "seed": seed,
            "offset": offset,
        }
        return "train-subset", "val-subset"

    def fake_fit_normalizers(subset):
        captured["fit_subset"] = subset
        return {"target": "normalizer"}

    monkeypatch.setattr(inference_script_module, "FloodDatasetWithQueryPoints", FakeSourceDataset)
    monkeypatch.setattr(inference_script_module, "split_dataset", fake_split_dataset)
    monkeypatch.setattr(inference_script_module, "fit_normalizers_from_sample_index", fake_fit_normalizers)

    cfg = OmegaConf.create(
        {
            "source_data": {
                "n_history": 3,
                "xy_file": "M40_XY.txt",
                "query_res": [32, 32],
                "static_files": [],
                "dynamic_patterns": {},
                "boundary_patterns": {},
                "skip_before_timestep": 0,
            },
            "distributed": {"seed": 123},
        }
    )
    data_io_cfg = OmegaConf.create(
        {
            "backend": "auto",
            "cache_dir_name": ".flood_cache",
            "rebuild_cache": False,
            "run_cache_size": 4,
        }
    )

    normalizers = inference_script_module.recreate_normalizers_from_source_split(
        cfg,
        "source-root",
        data_io_cfg,
    )

    assert normalizers == {"target": "normalizer"}
    assert captured["dataset_kwargs"]["data_root"] == "source-root"
    assert captured["split"]["lengths"] == [9, 1]
    assert captured["split"]["seed"] == 123
    assert captured["split"]["offset"] == 11
    assert captured["fit_subset"] == "train-subset"


# ---- Model Module Coverage ----

class ShapeAwareGNOIn(torch.nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

    def forward(self, y: torch.Tensor, x: torch.Tensor, f_y: torch.Tensor) -> torch.Tensor:
        del y
        batch_size = f_y.shape[0]
        num_latent_points = x.shape[0]
        base = torch.arange(
            num_latent_points,
            dtype=f_y.dtype,
            device=f_y.device,
        ).view(1, num_latent_points, 1)
        channel_offsets = torch.arange(
            self.hidden_channels,
            dtype=f_y.dtype,
            device=f_y.device,
        ).view(1, 1, self.hidden_channels)
        return base.expand(batch_size, -1, self.hidden_channels) + channel_offsets

class ShapeAwareLatentEmbedding(torch.nn.Module):
    def forward(self, in_p: torch.Tensor, ada_in=None) -> torch.Tensor:
        del ada_in
        return in_p.permute(0, 3, 1, 2).contiguous()

class ShapeAwareGNOOut(torch.nn.Module):
    def forward(self, y: torch.Tensor, x: torch.Tensor, f_y: torch.Tensor = None) -> torch.Tensor:
        del y
        query_count = x.shape[0]
        base = f_y.mean(dim=1, keepdim=True).expand(-1, query_count, -1)
        query_term = x.sum(dim=-1, keepdim=True).unsqueeze(0)
        return base + query_term

class ShapeAwareProjection(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        torch.nn.init.constant_(self.linear.weight, 0.25)
        torch.nn.init.constant_(self.linear.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

class ShapeAwareBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fno_hidden_channels = 4
        self.out_channels = 3
        self.in_coord_dim_reverse_order = (2, 3)
        self.gno_in = ShapeAwareGNOIn(self.fno_hidden_channels)
        self.latent_embedding = ShapeAwareLatentEmbedding()
        self.gno_out = ShapeAwareGNOOut()
        self.projection = ShapeAwareProjection(self.fno_hidden_channels, self.out_channels)

def _load_fixture(file_name: str):
    return torch.load(FIXTURE_DIR / file_name, map_location="cpu", weights_only=False)

def _clone_structure(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_structure(item) for key, item in value.items()}
    return deepcopy(value)

@pytest.mark.parametrize(
    ("backbone_kwargs", "wrapper_kwargs", "expected"),
    [
        (
            {},
            {},
            {
                "autoregressive": False,
                "residual_output": False,
                "max_autoregressive_steps": 1,
            },
        ),
        (
            {"fno_hidden_channels": 4, "out_channels": 3, "scale": 0.25},
            {
                "autoregressive": True,
                "residual_output": True,
                "max_autoregressive_steps": 4,
            },
            {
                "autoregressive": True,
                "residual_output": True,
                "max_autoregressive_steps": 4,
            },
        ),
    ],
    ids=["defaults", "custom"],
)
def test_gino_wrapper_constructor_attributes(backbone_kwargs, wrapper_kwargs, expected):
    wrapper = GINOWrapper(FakeGINOBackbone(**backbone_kwargs), **wrapper_kwargs)

    assert isinstance(wrapper.model, ImportableTorchModuleAdapter)
    assert wrapper.autoregressive is expected["autoregressive"]
    assert wrapper.residual_output is expected["residual_output"]
    assert wrapper.max_autoregressive_steps == expected["max_autoregressive_steps"]

@pytest.mark.parametrize(
    ("classifier_kwargs", "expected_fc_dim", "expected_lambda"),
    [
        (
            {
                "in_channels": 4,
                "conv_layers": [],
                "fc_dim": 1,
            },
            1,
            1.0,
        ),
        (
            {
                "in_channels": 4,
                "lambda_max": 0.7,
                "da_cfg": {
                    "conv_layers": [{"out_channels": 5, "kernel_size": 3, "pool_size": 2}],
                    "fc_dim": 2,
                },
            },
            2,
            0.7,
        ),
    ],
    ids=["defaults", "custom"],
)
def test_cnn_domain_classifier_constructor_attributes(
    classifier_kwargs,
    expected_fc_dim,
    expected_lambda,
):
    classifier = CNNDomainClassifier(**classifier_kwargs)

    assert classifier.in_channels == 4
    assert classifier.lambda_max == expected_lambda
    assert classifier.fc.out_features == expected_fc_dim
    assert classifier.grl.lambda_ == expected_lambda

@pytest.mark.parametrize(
    ("processor_kwargs", "expected_inverse_test"),
    [
        ({"device": "cpu"}, True),
        ({"device": "cpu", "inverse_test": False}, False),
    ],
    ids=["defaults", "custom"],
)
def test_flood_gino_data_processor_constructor_attributes(
    processor_kwargs,
    expected_inverse_test,
):
    processor = FloodGINODataProcessor(**processor_kwargs)

    assert processor.model is None
    assert processor.inverse_test is expected_inverse_test
    assert processor._device_str == "cpu"

def test_gino_wrapper_reference_tensor_output():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    wrapper = GINOWrapper(FakeGINOBackbone())

    output = wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )

    torch.testing.assert_close(output, fixture["tensor_output"], rtol=1e-6, atol=1e-6)

def test_gino_wrapper_reference_dict_output():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    wrapper = GINOWrapper(FakeGINOBackbone())

    output = wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=_clone_structure(fixture["dict_output_queries"]),
        x=fixture["x"],
    )

    assert output.keys() == fixture["dict_output"].keys()
    for key in output:
        torch.testing.assert_close(output[key], fixture["dict_output"][key], rtol=1e-6, atol=1e-6)

def test_gino_wrapper_autoregressive_residual_output():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    base_wrapper = GINOWrapper(FakeGINOBackbone())
    autoregressive_wrapper = GINOWrapper(
        FakeGINOBackbone(),
        autoregressive=True,
        residual_output=True,
    )

    base_output = base_wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )
    autoregressive_output = autoregressive_wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )

    expected = base_output + fixture["x"][:, :, -3:]
    torch.testing.assert_close(autoregressive_output, expected, rtol=1e-6, atol=1e-6)

def test_gino_wrapper_maps_mesh_inputs_to_latent_grid_points():
    wrapper = GINOWrapper(ShapeAwareBackbone())
    input_geom = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    latent_queries = torch.tensor(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    output_queries = input_geom.clone()
    x = torch.randn(2, input_geom.shape[0], 6)

    output = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
    )

    assert output.shape == (2, output_queries.shape[0], 3)

def test_cnn_domain_classifier_reference_output():
    fixture = _load_fixture("domain_classifier_reference.pth")
    classifier = CNNDomainClassifier(
        in_channels=fixture["config"]["in_channels"],
        lambda_max=fixture["config"]["lambda_max"],
        da_cfg=fixture["config"]["da_cfg"],
    )
    classifier.load_state_dict(fixture["state_dict"])

    output = classifier(fixture["input"])

    torch.testing.assert_close(output, fixture["output"], rtol=1e-6, atol=1e-6)

def test_flood_gino_data_processor_reference_behaviour():
    fixture = _load_fixture("processor_reference.pth")
    processor = FloodGINODataProcessor(device="cpu", inverse_test=False)
    processor.eval()

    processed = processor.preprocess(_clone_structure(fixture["sample"]))
    for key, expected in fixture["processed"].items():
        actual = processed.get(key)
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    post_out, post_sample = processor.postprocess(
        fixture["postprocess_input"].clone(),
        _clone_structure(processed),
    )
    torch.testing.assert_close(post_out, fixture["postprocess_output"], rtol=1e-6, atol=1e-6)
    for key in fixture["processed"]:
        if fixture["processed"][key] is not None:
            torch.testing.assert_close(post_sample[key], fixture["processed"][key], rtol=1e-6, atol=1e-6)

def test_gino_wrapper_from_checkpoint_smoke():
    fixture = _load_fixture("gino_wrapper_reference.pth")
    wrapper = GINOWrapper.from_checkpoint(str(FIXTURE_DIR / "gino_wrapper.mdlus"))

    output = wrapper(
        input_geom=fixture["input_geom"],
        latent_queries=fixture["latent_queries"],
        output_queries=fixture["output_queries"],
        x=fixture["x"],
    )

    torch.testing.assert_close(output, fixture["tensor_output"], rtol=1e-6, atol=1e-6)

def test_cnn_domain_classifier_from_checkpoint_roundtrip():
    fixture = _load_fixture("domain_classifier_reference.pth")
    classifier = CNNDomainClassifier.from_checkpoint(str(FIXTURE_DIR / "domain_classifier.mdlus"))

    output = classifier(fixture["input"])

    torch.testing.assert_close(output, fixture["output"], rtol=1e-6, atol=1e-6)

def test_flood_gino_data_processor_from_checkpoint_roundtrip():
    fixture = _load_fixture("processor_reference.pth")
    processor = FloodGINODataProcessor.from_checkpoint(str(FIXTURE_DIR / "processor.mdlus"))

    processed = processor.preprocess(_clone_structure(fixture["sample"]))
    for key, expected in fixture["processed"].items():
        if expected is not None:
            torch.testing.assert_close(processed[key], expected, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize(
    ("output_queries", "error_type", "message"),
    [
        (["bad"], TypeError, "torch.Tensor or a non-empty dict"),
        ({}, ValueError, "must not be empty"),
        ({"main": "bad"}, TypeError, "must be a torch.Tensor"),
        (torch.zeros(1, 2, 3, 2), ValueError, "rank 2 or 3"),
        (torch.zeros(6, 3), ValueError, "coordinate dimension"),
        (
            {"a": torch.zeros(6, 2), "b": torch.zeros(1, 6, 2)},
            ValueError,
            "same batch style",
        ),
    ],
)
def test_gino_wrapper_rejects_invalid_output_queries(output_queries, error_type, message):
    wrapper = GINOWrapper(FakeGINOBackbone())
    fixture = _load_fixture("gino_wrapper_reference.pth")

    with pytest.raises(error_type, match=message):
        wrapper(
            input_geom=fixture["input_geom"],
            latent_queries=fixture["latent_queries"],
            output_queries=output_queries,
            x=fixture["x"],
        )


# ---- Data Processing Coverage ----

class MockGINOModelForCheckpoint(physicsnemo.Module):
    """Simple test GINO model for checkpoint testing."""
    def __init__(self):
        super().__init__(meta=physicsnemo.ModelMetaData(name="MockGINOForCheckpoint"))
        self.fno_hidden_channels = 64
        self.out_channels = 3
        self.gno_coord_dim = 2
        self.latent_feature_channels = None
        self.in_coord_dim_reverse_order = [2, 3]
        self.out_gno_tanh = None
        
        # Create minimal layers for the model to work
        self.gno_in = nn.Linear(2, 64)
        self.gno_out = nn.Linear(64, 64)
        self.projection = nn.Linear(64, 3)
        self.latent_embedding = nn.Identity()

@pytest.fixture
def sample_dict():
    """Create sample dictionary for preprocessing."""
    batch_size = 2
    num_cells = 100
    n_history = 3
    # query_points should be (B, H, W, 2) or (H, W, 2) for latent queries
    # Using a simple grid: (8, 8, 2) for 2D
    H, W = 8, 8
    geometry = torch.rand(num_cells, 2).unsqueeze(0).expand(batch_size, -1, -1).clone()
    query_points = torch.rand(H, W, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

    return {
        "geometry": geometry,
        "static": torch.rand(batch_size, num_cells, 7),
        "boundary": torch.rand(batch_size, n_history, 1),
        "dynamic": torch.rand(batch_size, n_history, num_cells, 3),
        "target": torch.rand(batch_size, num_cells, 3),
        "query_points": query_points,  # Required for preprocessing
    }

@pytest.fixture
def fake_gino_model():
    """Create an importable deterministic GINO-like backend."""
    return FakeGINOBackbone(fno_hidden_channels=64, out_channels=3, scale=0.1)

@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_init(device):
    """Test FloodGINODataProcessor initialization."""
    processor = FloodGINODataProcessor(device=device)
    # device property returns torch.device, so compare string representations
    assert str(processor.device) == str(torch.device(device))
    assert processor.target_norm is None
    assert processor.inverse_test is True
    assert processor.model is None
    assert isinstance(processor, nn.Module)

@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_preprocess(sample_dict, device):
    """Test preprocessing with batched input."""
    processor = FloodGINODataProcessor(device=device)
    result = processor.preprocess(sample_dict)

    # Check required keys exist
    assert "input_geom" in result
    assert "latent_queries" in result
    assert "output_queries" in result
    assert "x" in result
    assert "y" in result

    # Check geometry has no batch dim (shared)
    assert result["input_geom"].dim() == 2  # (n_cells, 2)
    assert result["latent_queries"].dim() == 3  # (H, W, 2)
    assert result["output_queries"].dim() == 2  # (n_cells, 2)

    # Check x has batch dim
    assert result["x"].dim() == 3  # (B, n_cells, features)
    assert result["x"].shape == (2, 100, 19)
    expected_boundary = sample_dict["boundary"].to(device).reshape(2, 1, 3).expand(-1, 100, -1)
    torch.testing.assert_close(result["x"][..., 7:10], expected_boundary)

@pytest.mark.parametrize(
    ("mutated_key", "message"),
    [
        ("geometry", "geometry is batched with non-identical entries"),
        ("query_points", "query_points is batched with non-identical entries"),
    ],
)
@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_rejects_non_identical_batched_shared_tensors(
    sample_dict,
    mutated_key,
    message,
    device,
):
    """Shared geometry/query batches must be exact copies before collapsing to item zero."""
    processor = FloodGINODataProcessor(device=device)
    bad_sample = _clone_structure(sample_dict)
    bad_sample[mutated_key][1] = bad_sample[mutated_key][1] + 1.0

    with pytest.raises(ValueError, match=message):
        processor.preprocess(bad_sample)

@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_postprocess(device):
    """Test postprocessing in training and eval modes."""
    mock_norm = MagicMock()
    mock_norm.inverse_transform = MagicMock(return_value=torch.ones(2, 100, 3) * 2)

    processor = FloodGINODataProcessor(device=device, target_norm=mock_norm, inverse_test=True)
    out = torch.ones(2, 100, 3)
    sample = {"y": torch.ones(2, 100, 3)}

    # Training mode: no inverse transform
    processor.train()
    result_out, _ = processor.postprocess(out, sample)
    mock_norm.inverse_transform.assert_not_called()

    # Eval mode: applies inverse transform
    processor.eval()
    result_out, _ = processor.postprocess(out, sample)
    assert torch.allclose(result_out, torch.ones(2, 100, 3) * 2)

@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_init(fake_gino_model, device):
    """Test GINOWrapper initialization."""
    wrapper = GINOWrapper(fake_gino_model).to(device)
    assert isinstance(wrapper.model, ImportableTorchModuleAdapter)
    assert isinstance(wrapper.model.inner_model, FakeGINOBackbone)
    assert isinstance(wrapper, nn.Module)
    assert wrapper.fno_hidden_channels == 64
    assert wrapper.autoregressive is False  # Default value

@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_forward(fake_gino_model, device):
    """Test GINOWrapper forward pass and autoregressive residual mode."""
    base_wrapper = GINOWrapper(fake_gino_model).to(device)
    residual_wrapper = GINOWrapper(
        FakeGINOBackbone(fno_hidden_channels=64, out_channels=3, scale=0.1),
        autoregressive=True,
        residual_output=True,
    ).to(device)
    residual_wrapper.model.load_state_dict(base_wrapper.model.state_dict())

    num_cells = 64
    input_geom = torch.rand(1, num_cells, 2, device=device)
    latent_queries = torch.rand(1, 8, 8, 2, device=device)
    output_queries = torch.rand(1, num_cells, 2, device=device)
    x = torch.rand(1, num_cells, 10, device=device)

    base_result = base_wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
    )
    result = residual_wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
    )

    # Verify result shape and autoregressive residual
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, num_cells, 3)
    expected = base_result + x[..., -3:]
    assert torch.allclose(result, expected, atol=1e-5)

    # Test return_features
    out, features = base_wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        return_features=True,
    )
    assert isinstance(features, torch.Tensor)
    assert features.shape == (1, 64, 8, 8)  # (B, hidden_channels, H, W)

@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_feature_grid_feeds_domain_classifier(device):
    """GINOWrapper feature returns should satisfy the adversarial DA classifier contract."""
    wrapper = GINOWrapper(FakeGINOBackbone(fno_hidden_channels=4)).to(device)
    classifier = CNNDomainClassifier(
        in_channels=4,
        da_cfg={"conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}], "fc_dim": 1},
    ).to(device)
    num_cells = 16
    input_geom = torch.rand(1, num_cells, 2, device=device)
    latent_queries = torch.rand(1, 4, 4, 2, device=device)
    output_queries = torch.rand(1, num_cells, 2, device=device)
    x = torch.rand(1, num_cells, 10, device=device)

    _, features = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        return_features=True,
    )
    logits = classifier(features)

    assert features.shape == (1, 4, 4, 4)
    assert logits.shape == (1, 1)

@pytest.mark.parametrize("device", _DEVICES)
def test_lploss_wrapper(device):
    """Test LpLossWrapper filters kwargs correctly."""
    mock_loss = MagicMock(return_value=torch.tensor(0.5))
    wrapper = LpLossWrapper(mock_loss)

    y_pred = torch.rand(2, 100, 3)
    y = torch.rand(2, 100, 3)

    # Extra kwargs should be ignored
    wrapper(y_pred, y=y, input_geom=torch.rand(100, 2), extra_arg="ignored")

    # Should only pass y_pred and y
    mock_loss.assert_called_once_with(y_pred, y)

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
def test_ginowrapper_supported_checkpoint_path(device, tmp_path):
    """The supported load path is instantiate-from-config plus load_checkpoint."""
    checkpoint_dir = tmp_path / "checkpoint_gino_wrapper"
    gino_model = MockGINOModelForCheckpoint()
    model_orig = GINOWrapper(gino_model, autoregressive=False).to(device)

    save_checkpoint(path=str(checkpoint_dir), models=model_orig, epoch=0)

    reloaded = GINOWrapper(MockGINOModelForCheckpoint(), autoregressive=False).to(device)
    load_checkpoint(path=str(checkpoint_dir), models=reloaded, epoch=0, device=device)

    assert torch.allclose(reloaded.model.gno_in.weight, model_orig.model.gno_in.weight)
    assert torch.allclose(reloaded.model.gno_out.weight, model_orig.model.gno_out.weight)
    assert torch.allclose(reloaded.model.projection.weight, model_orig.model.projection.weight)

def test_ginowrapper_does_not_advertise_legacy_from_checkpoint():
    """GINOWrapper should inherit the PhysicsNeMo checkpoint signature, not the legacy neuralop one."""
    import inspect

    signature = inspect.signature(GINOWrapper.from_checkpoint)
    assert "file_name" in signature.parameters
    assert "save_folder" not in signature.parameters
    assert "save_name" not in signature.parameters


# ---- Trainer Coverage ----

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

@pytest.mark.parametrize("device", _DEVICES)
def test_trainer_filters_loss_only_targets_before_model_forward(device):
    """Processed target tensors should stay available for the loss but never reach model.forward."""

    class StrictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 3)
            self.seen_kwargs = None

        def forward(self, x=None, **kwargs):
            self.seen_kwargs = dict(kwargs)
            assert "y" not in kwargs
            return self.linear(x)

    class Processor:
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

    model = StrictModel().to(device)
    trainer = NeuralOperatorTrainer(
        model=model,
        n_epochs=1,
        device=device,
        data_processor=Processor(),
        verbose=False,
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer.regularizer = None

    loss, processed = trainer._train_one_batch(
        0,
        {
            "x": torch.rand(2, 10),
            "target": torch.rand(2, 3),
        },
        lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y),
    )

    assert torch.is_tensor(loss)
    assert "y" in processed
    assert model.seen_kwargs == {}

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

def test_trainer_save_checkpoint_uses_native_save_and_best_sidecar(tmp_path, monkeypatch):
    """Best-checkpoint saving should use native checkpointing and sidecar metadata."""
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

        @staticmethod
        def is_initialized():
            return True

    def fake_save_checkpoint(**kwargs):
        saved["save_checkpoint"] = kwargs

    def fake_write_best_checkpoint_metadata(path, **kwargs):
        saved["write_best"] = {"path": Path(path), **kwargs}

    monkeypatch.setattr(trainer_module, "DistributedManager", FakeDistributedManager)
    monkeypatch.setattr(trainer_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(trainer_module, "write_best_checkpoint_metadata", fake_write_best_checkpoint_metadata)

    save_dir = tmp_path / "trainer_ckpt"
    trainer._save_checkpoint(save_dir, is_best=True)

    assert saved["save_checkpoint"]["models"] is model
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

def test_trainer_resume_from_checkpoint_requires_physicsnemo_files(tmp_path, monkeypatch):
    """Resume should fail fast when a PhysicsNeMo checkpoint cannot be resolved."""
    trainer = NeuralOperatorTrainer(
        model=nn.Linear(4, 2),
        n_epochs=1,
        device="cpu",
        verbose=False,
    )
    trainer.optimizer = "optimizer"
    trainer.scheduler = "scheduler"

    def fail_resolve_checkpoint_epoch(path, mode):
        raise FileNotFoundError("missing physicsnemo checkpoint")

    monkeypatch.setattr(trainer_module, "resolve_checkpoint_epoch", fail_resolve_checkpoint_epoch)

    with pytest.raises(FileNotFoundError, match="missing physicsnemo checkpoint"):
        trainer._resume_from_checkpoint(tmp_path)

def test_trainer_save_every_none_saves_best_and_final_latest(tmp_path, monkeypatch):
    """save_every=None should suppress interval saves but still write best and final latest."""

    class TinyKeywordModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x=None):
            return self.linear(x)

    model = TinyKeywordModel()
    samples = [{"x": torch.ones(2), "y": torch.ones(1)} for _ in range(2)]
    loader = DataLoader(_DictDataset(samples), batch_size=2, shuffle=False)
    trainer = NeuralOperatorTrainer(model=model, n_epochs=1, device="cpu", verbose=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    saved = []

    def fake_save_checkpoint(save_dir, is_best=False):
        saved.append((trainer.epoch, is_best))

    monkeypatch.setattr(trainer, "_save_checkpoint", fake_save_checkpoint)

    trainer.train(
        train_loader=loader,
        test_loaders={"val": loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y),
        eval_losses={"l2": lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y)},
        save_dir=tmp_path / "ckpt",
        save_best="val_l2",
        save_every=None,
    )

    assert saved == [(0, True), (0, False)]


# ---- Pretraining And Domain Adaptation Coverage ----

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
        self.grl = GradientReversal(lambda_=1.0)
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

def test_adapt_model_constructs_domain_classifier_with_keyword_args(tmp_path, monkeypatch):
    """adapt_model should not pass da_lambda_max positionally as da_cfg."""

    class FakeTargetDataset(Dataset):
        def __init__(self, **kwargs):
            self.samples = [{"x": torch.rand(10), "y": torch.rand(3)} for _ in range(4)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    class FakeLazyNormalizedDataset(Dataset):
        def __init__(self, base_dataset, normalizers, query_res, apply_noise):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            return self.base_dataset[idx]

    class FakeAdaptTrainer:
        def __init__(self, **kwargs):
            captured["trainer_domain_classifier"] = kwargs["domain_classifier"]

        def train_domain_adaptation(self, **kwargs):
            captured["class_loss_weight"] = kwargs["class_loss_weight"]
            Path(kwargs["save_dir"]).mkdir(parents=True, exist_ok=True)

    captured = {}
    real_classifier_cls = domain_adaptation_module.CNNDomainClassifier

    def fake_classifier(*args, **kwargs):
        captured["classifier_args"] = args
        captured["classifier_kwargs"] = kwargs
        return real_classifier_cls(*args, **kwargs)

    model = _SimpleDARegressionModel(feature_channels=4)
    model.fno_hidden_channels = 4
    source_samples = [{"x": torch.rand(10), "y": torch.rand(3)} for _ in range(4)]
    source_loader = DataLoader(_DictDataset(source_samples), batch_size=2, shuffle=False)
    config = OmegaConf.create(
        {
            "training": {
                "da_classifier": {
                    "conv_layers": [{"out_channels": 4, "kernel_size": 1, "pool_size": 1}],
                    "fc_dim": 1,
                },
                "da_lambda_max": 0.5,
                "adapt_learning_rate": 1e-3,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "training_loss": "l2",
                "testing_loss": "l2",
                "amp_autocast": False,
                "n_epochs_adapt": 0,
                "da_class_loss_weight": 0.1,
                "eval_interval": 1,
            },
            "checkpoint": {
                "save_dir": str(tmp_path),
                "resume_from_adapt": None,
            },
            "distributed": {"seed": 123},
            "data_io": {},
            "wandb": {"eval_interval": 1},
        }
    )
    target_data_config = OmegaConf.create(
        {
            "root": "target-root",
            "n_history": 3,
            "xy_file": "M40_XY.txt",
            "query_res": [2, 2],
            "static_files": [],
            "dynamic_patterns": {},
            "boundary_patterns": {},
            "skip_before_timestep": 0,
            "noise_type": "none",
            "noise_std": None,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    )
    normalizers = {"target": nn.Identity()}

    monkeypatch.setattr(domain_adaptation_module, "FloodDatasetWithQueryPoints", FakeTargetDataset)
    monkeypatch.setattr(domain_adaptation_module, "LazyNormalizedDataset", FakeLazyNormalizedDataset)
    monkeypatch.setattr(domain_adaptation_module, "CNNDomainClassifier", fake_classifier)
    monkeypatch.setattr(domain_adaptation_module, "DomainAdaptationTrainer", FakeAdaptTrainer)

    domain_adaptation_module.adapt_model(
        model=model,
        normalizers=normalizers,
        data_processor=None,
        config=config,
        device="cpu",
        is_logger=False,
        target_data_config=target_data_config,
        source_train_loader=source_loader,
        source_val_loader=source_loader,
        logger=MagicMock(),
    )

    assert captured["classifier_args"] == ()
    assert captured["classifier_kwargs"]["in_channels"] == 4
    assert captured["classifier_kwargs"]["da_cfg"] == config.training.da_classifier
    assert captured["classifier_kwargs"]["lambda_max"] == pytest.approx(0.5)
    assert captured["class_loss_weight"] == pytest.approx(0.1)

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
def test_domain_adaptation_filters_loss_only_targets_before_model_forward(device, tmp_path):
    """DA training should keep processor targets for the loss without forwarding them to the model."""

    class StrictDAProcessor:
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

    class StrictDARegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 3)
            self.seen_kwargs = []

        def forward(self, x=None, return_features: bool = False, **kwargs):
            self.seen_kwargs.append(dict(kwargs))
            assert "y" not in kwargs
            out = self.linear(x)
            if return_features:
                features = torch.ones(x.shape[0], 4, 2, 2, device=x.device)
                return out, features
            return out

    model = StrictDARegressionModel().to(device)
    classifier = _ClassifierShouldNotRun().to(device)
    trainer = DomainAdaptationTrainer(
        model=model,
        data_processor=StrictDAProcessor(),
        domain_classifier=classifier,
        device=device,
        verbose=False,
    )

    samples = [{"x": torch.rand(10), "target": torch.rand(3)} for _ in range(4)]
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
        adaptation_epochs=1,
        save_dir=tmp_path / "adapt_filters_targets",
        val_loaders={"target_val": loader},
    )

    assert model.seen_kwargs
    assert all("y" not in kwargs for kwargs in model.seen_kwargs)

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
def test_domain_adaptation_save_every_none_saves_best_and_final_latest(device, tmp_path, monkeypatch):
    """DA save_every=None should suppress interval saves but keep best and final latest checkpoints."""
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
    saved = []

    def fake_save_checkpoint(
        save_dir,
        optimizer,
        scheduler,
        epoch,
        save_classifier=True,
        is_best=False,
        metric_name=None,
    ):
        saved.append((epoch, is_best))

    monkeypatch.setattr(trainer, "_save_checkpoint", fake_save_checkpoint)

    trainer.train_domain_adaptation(
        src_loader=loader,
        tgt_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=lambda pred, y=None, **_: torch.nn.functional.mse_loss(pred, y),
        class_loss_weight=0.0,
        adaptation_epochs=1,
        save_every=None,
        save_dir=tmp_path / "adapt_no_interval",
        val_loaders={"target_val": loader},
    )

    assert saved == [(0, True), (0, False)]

@pytest.mark.parametrize("device", _DEVICES)
def test_pretrain_model_wires_datasets_loaders_and_normalizer_save(device, tmp_path, monkeypatch):
    """Pretraining should split data, build loaders, invoke the trainer, and save normalizers on rank 0."""
    from omegaconf import OmegaConf

    fake_samples = [{"index": idx} for idx in range(10)]

    class FakeLazyNormalizedDataset:
        def __init__(self, base_dataset, normalizers, query_res, apply_noise):
            self.base_dataset = base_dataset
            self.normalizers = normalizers
            self.query_res = query_res
            self.apply_noise = apply_noise

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            return self.base_dataset[idx]

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

    def fake_fit_normalizers_from_sample_index(dataset):
        assert len(dataset) == 9
        return {
            "dynamic": "dynamic_norm",
            "target": "target_norm",
            "boundary": "boundary_norm",
            "static": "static_norm",
        }

    def fake_create_loader_from_config(dataset, data_config, shuffle):
        loader = {
            "dataset": dataset,
            "shuffle": shuffle,
            "batch_size": data_config.batch_size,
        }
        loader_calls.append(loader)
        return loader

    monkeypatch.setattr(pretraining_module, "FloodDatasetWithQueryPoints", lambda **kwargs: fake_samples)
    monkeypatch.setattr(pretraining_module, "LazyNormalizedDataset", FakeLazyNormalizedDataset)
    monkeypatch.setattr(pretraining_module, "fit_normalizers_from_sample_index", fake_fit_normalizers_from_sample_index)
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
    assert trainer.init_kwargs["eval_interval"] == 1
    assert trainer.train_kwargs["save_best"] == "source_val_l2"
    assert trainer.train_kwargs["test_loaders"]["source_val"]["shuffle"] is False
    assert trainer.source_train_loader["shuffle"] is True
    assert trainer.source_val_loader["shuffle"] is False
    assert len(trainer.source_train_dataset) == 9
    assert len(trainer.source_val_dataset) == 1
    assert trainer.source_train_dataset.apply_noise is True
    assert trainer.source_val_dataset.apply_noise is False
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

def test_domain_adaptation_resume_requires_physicsnemo_checkpoint(tmp_path, monkeypatch):
    """DA resume should not fall back to neuralop-format checkpoints."""
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
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def fail_resolve_checkpoint_epoch(path, mode):
        raise FileNotFoundError("missing physicsnemo checkpoint")

    monkeypatch.setattr(domain_adaptation_module, "resolve_checkpoint_epoch", fail_resolve_checkpoint_epoch)

    with pytest.raises(FileNotFoundError, match="missing physicsnemo checkpoint"):
        trainer._resume_from_checkpoint(tmp_path, optimizer, scheduler)

def test_train_domain_adaptation_classifier_resume_requires_physicsnemo_checkpoint(tmp_path, monkeypatch):
    """Separate classifier resume should require PhysicsNeMo model files."""
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

    with pytest.raises(FileNotFoundError, match="missing classifier checkpoint"):
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


# ---- Modern integration smoke coverage ----


def _make_compact_flood_samples(device: str, count: int = 4):
    samples = []
    num_cells = 8
    geometry = torch.rand(num_cells, 2, device=device)
    query_points = torch.rand(4, 4, 2, device=device)
    for _ in range(count):
        samples.append(
            {
                "geometry": geometry.clone(),
                "static": torch.rand(num_cells, 7, device=device),
                "boundary": torch.rand(3, 1, device=device),
                "dynamic": torch.rand(3, num_cells, 3, device=device),
                "target": torch.rand(num_cells, 3, device=device),
                "query_points": query_points.clone(),
            }
        )
    return samples


@pytest.mark.parametrize("device", _DEVICES)
def test_gino_wrapper_trainer_integration_smoke(device, tmp_path):
    """Exercise the current processor -> GINOWrapper -> trainer path with compact boundary input."""

    model = GINOWrapper(FakeGINOBackbone()).to(device)
    processor = FloodGINODataProcessor(device=device)
    processor.wrap(model)

    class FloodSmokeDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    loader = DataLoader(FloodSmokeDataset(_make_compact_flood_samples(device)), batch_size=2, shuffle=False)
    trainer = NeuralOperatorTrainer(
        model=model,
        n_epochs=1,
        device=device,
        data_processor=processor,
        verbose=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    metrics = trainer.train(
        train_loader=loader,
        test_loaders={"val": loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=LpLoss(d=2, p=2),
        eval_losses={"l2": LpLoss(d=2, p=2)},
        save_dir=tmp_path / "integrated_trainer",
        save_best=None,
        save_every=None,
    )

    assert "avg_loss" in metrics
    assert "train_err" in metrics
    assert "val_l2" in metrics
