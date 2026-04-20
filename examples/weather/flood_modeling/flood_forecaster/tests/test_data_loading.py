# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import sys
import time
from pathlib import Path

import torch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))
REPO_ROOT = EXAMPLE_ROOT.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import (  # noqa: E402
    FloodDatasetWithQueryPoints,
    FloodRolloutTestDatasetNew,
    LazyNormalizedDataset,
    LazyNormalizedRolloutDataset,
    prepare_flood_cache,
)
from data_processing import FloodGINODataProcessor  # noqa: E402
from utils.normalization import fit_normalizers_from_sample_index  # noqa: E402
from utils.runtime import create_loader_from_config, resolve_eval_interval  # noqa: E402


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
