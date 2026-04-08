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

"""Unit tests for the HydroGraphNet dataset and helper utilities."""

import hashlib
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from test.conftest import requires_module


def _load_example_utils():
    utils_path = Path(__file__).resolve().parents[2] / "examples" / "weather" / "flood_modeling" / "hydrographnet" / "utils.py"
    spec = importlib.util.spec_from_file_location("hydrographnet_example_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_matrix(path: Path, array: np.ndarray) -> None:
    np.savetxt(path, array, delimiter="\t")


class _Cfg(dict):
    def __getattr__(self, key):
        return self[key]


def _build_toy_hydrograph_dir(root: Path, hydro_ids=("001", "002"), num_steps=140) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    prefix = "M80"
    num_nodes = 4
    times = np.arange(num_steps, dtype=np.float64)

    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    area = np.array([2.0, 2.5, 3.0, 3.5]).reshape(-1, 1)
    elevation = np.array([100.0, 101.0, 102.0, 103.0]).reshape(-1, 1)
    slope = np.array([0.01, 0.015, 0.02, 0.025]).reshape(-1, 1)
    aspect = np.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1)
    curvature = np.array([0.1, 0.2, 0.3, 0.4]).reshape(-1, 1)
    manning = np.array([0.03, 0.031, 0.032, 0.033]).reshape(-1, 1)
    flow_accum = np.array([5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)
    infiltration = np.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1)

    _write_matrix(root / f"{prefix}_XY.txt", xy)
    _write_matrix(root / f"{prefix}_CA.txt", area)
    _write_matrix(root / f"{prefix}_CE.txt", elevation)
    _write_matrix(root / f"{prefix}_CS.txt", slope)
    _write_matrix(root / f"{prefix}_A.txt", aspect)
    _write_matrix(root / f"{prefix}_CU.txt", curvature)
    _write_matrix(root / f"{prefix}_N.txt", manning)
    _write_matrix(root / f"{prefix}_FA.txt", flow_accum)
    _write_matrix(root / f"{prefix}_IP.txt", infiltration)

    with open(root / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(hydro_ids))
    with open(root / "test.txt", "w", encoding="utf-8") as f:
        f.write(hydro_ids[0])

    for offset, hydro_id in enumerate(hydro_ids):
        inflow = 5.0 + 20.0 * np.exp(-((times - (90 + offset)) / 8.0) ** 2)
        precipitation = 0.001 + 0.0002 * np.sin(times / 8.0 + offset)
        base_depth = 0.05 * np.maximum(times - 70.0, 0.0)
        node_offsets = np.linspace(0.1, 0.4, num_nodes)
        water_depth = base_depth[:, None] + node_offsets[None, :] + 0.01 * offset
        volume = water_depth * area.reshape(1, -1)

        _write_matrix(root / f"{prefix}_WD_{hydro_id}.txt", water_depth)
        _write_matrix(
            root / f"{prefix}_US_InF_{hydro_id}.txt",
            np.column_stack([times, inflow]),
        )
        _write_matrix(root / f"{prefix}_V_{hydro_id}.txt", volume)
        _write_matrix(root / f"{prefix}_Pr_{hydro_id}.txt", precipitation.reshape(-1, 1))

    return root


@pytest.fixture()
def hydrograph_dirs(tmp_path):
    train_dir = _build_toy_hydrograph_dir(tmp_path / "train_data")
    test_dir = _build_toy_hydrograph_dir(tmp_path / "test_data")
    return train_dir, test_dir


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_dataset_utility_functions_and_error_paths(tmp_path, monkeypatch):
    import physicsnemo.datapipes.gnn.hydrographnet_dataset as hydro_ds

    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        HydroGraphDataset,
        calculate_md5,
        check_integrity,
        check_md5,
        ensure_data_available,
    )

    payload = tmp_path / "payload.txt"
    payload.write_text("hydrographnet", encoding="utf-8")
    md5 = calculate_md5(payload)
    assert check_md5(payload, md5)
    assert check_integrity(payload, md5)
    assert check_integrity(payload, None)
    assert not check_integrity(payload.with_name("missing.txt"))

    existing_dir = tmp_path / "existing_data"
    existing_dir.mkdir()
    monkeypatch.setattr(
        "physicsnemo.datapipes.gnn.hydrographnet_dataset.download_from_zenodo_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("download should not be called for an existing directory")
        ),
    )
    ensure_data_available(existing_dir)

    called = {}

    def _fake_download(record_id, root, files_to_download):
        called["record_id"] = record_id
        called["root"] = Path(root)
        called["files"] = files_to_download
        Path(root).mkdir(parents=True, exist_ok=True)

    missing_dir = tmp_path / "missing_data"
    monkeypatch.setattr(
        "physicsnemo.datapipes.gnn.hydrographnet_dataset.download_from_zenodo_record",
        _fake_download,
    )
    ensure_data_available(missing_dir)
    assert called["root"] == missing_dir

    monkeypatch.setattr(hydro_ds.sys, "version_info", (3, 8))

    class _DummyMd5:
        def __init__(self):
            self.data = b""

        def update(self, chunk):
            self.data += chunk

        def hexdigest(self):
            return "dummy"

    monkeypatch.setattr(hydro_ds.hashlib, "md5", lambda: _DummyMd5())
    assert hydro_ds.calculate_md5(payload) == "dummy"

    toy_dir = _build_toy_hydrograph_dir(tmp_path / "toy_errors")
    with pytest.raises(ValueError):
        HydroGraphDataset(data_dir=toy_dir, split="validation")

    with pytest.raises(FileNotFoundError):
        HydroGraphDataset(
            data_dir=toy_dir,
            split="train",
            hydrograph_ids_file="missing_ids.txt",
        )


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_download_helpers_with_mocked_network(tmp_path, monkeypatch):
    import physicsnemo.datapipes.gnn.hydrographnet_dataset as hydro_ds

    def _make_zip_bytes():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("zip_payload.txt", "zip-data")
        return buf.getvalue()

    def _make_tar_bytes():
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as archive:
            data = b"tar-data"
            info = tarfile.TarInfo(name="tar_payload.txt")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    class _FakeResponse:
        def __init__(self, payload: bytes, status_code: int = 200):
            self.payload = payload
            self.status_code = status_code
            self.headers = {"content-length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("request failed")

        def iter_content(self, chunk_size=8192):
            for idx in range(0, len(self.payload), chunk_size):
                yield self.payload[idx : idx + chunk_size]

        def json(self):
            raise AssertionError("json() should not be used here")

    zip_bytes = _make_zip_bytes()
    tar_bytes = _make_tar_bytes()

    def _fake_get(url, *args, **kwargs):
        if url.endswith(".zip"):
            return _FakeResponse(zip_bytes)
        if url.endswith(".tar"):
            return _FakeResponse(tar_bytes)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(hydro_ds.requests, "get", _fake_get)

    existing = tmp_path / "existing.txt"
    existing.write_text("already-here", encoding="utf-8")
    existing_md5 = hashlib.md5(existing.read_bytes(), usedforsecurity=False).hexdigest()
    hydro_ds.download_from_url(
        url="https://example.com/existing.txt",
        root=tmp_path,
        filename="existing.txt",
        md5=existing_md5,
        extract=False,
    )

    zip_md5 = hashlib.md5(zip_bytes, usedforsecurity=False).hexdigest()
    hydro_ds.download_from_url(
        url="https://example.com/archive.zip",
        root=tmp_path / "zip_case",
        md5=zip_md5,
        size=len(zip_bytes),
        extract=True,
    )
    assert (tmp_path / "zip_case" / "zip_payload.txt").exists()

    tar_md5 = hashlib.md5(tar_bytes, usedforsecurity=False).hexdigest()
    hydro_ds.download_from_url(
        url="https://example.com/archive.tar",
        root=tmp_path / "tar_case",
        md5=tar_md5,
        size=len(tar_bytes),
        extract=True,
    )
    assert (tmp_path / "tar_case" / "tar_payload.txt").exists()


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_download_from_zenodo_record_with_mocked_metadata(tmp_path, monkeypatch):
    import physicsnemo.datapipes.gnn.hydrographnet_dataset as hydro_ds

    calls = []

    class _ZenodoResponse:
        status_code = 200

        def json(self):
            return {
                "files": [
                    {
                        "key": "wanted.txt",
                        "links": {"self": "https://example.com/wanted.txt"},
                        "checksum": "md5:abc123",
                        "size": 10,
                    },
                    {
                        "key": "skip.txt",
                        "links": {"self": "https://example.com/skip.txt"},
                        "checksum": "md5:def456",
                        "size": 20,
                    },
                ]
            }

    monkeypatch.setattr(hydro_ds.requests, "get", lambda url: _ZenodoResponse())
    monkeypatch.setattr(
        hydro_ds,
        "download_from_url",
        lambda **kwargs: calls.append(kwargs),
    )

    hydro_ds.download_from_zenodo_record(
        record_id="12345",
        root=tmp_path,
        files_to_download=["wanted.txt"],
    )
    assert calls == [
        {
            "url": "https://example.com/wanted.txt",
            "root": tmp_path,
            "filename": "wanted.txt",
            "md5": "abc123",
            "size": 10,
            "extract": True,
        }
    ]

    class _FailedResponse:
        status_code = 500

        def json(self):
            return {}

    monkeypatch.setattr(hydro_ds.requests, "get", lambda url: _FailedResponse())
    with pytest.raises(RuntimeError):
        hydro_ds.download_from_zenodo_record("999", tmp_path)


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_feature_layout_and_rain_helpers():
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        compute_effective_rain_area_sum,
        get_hydrograph_feature_layout,
        get_hydrograph_input_dim,
    )

    layout = get_hydrograph_feature_layout(3)
    assert layout.mesh_slice == slice(0, 10)
    assert layout.forcing_slice == slice(10, 12)
    assert layout.water_depth_slice == slice(12, 15)
    assert layout.volume_slice == slice(15, 18)
    assert get_hydrograph_input_dim(3) == 18

    with pytest.raises(ValueError):
        get_hydrograph_feature_layout(0)

    area = np.array([[2.0], [3.0], [5.0]])
    infiltration_percent = np.array([[10.0], [20.0], [40.0]])
    expected = 2.0 * 0.9 + 3.0 * 0.8 + 5.0 * 0.6
    assert compute_effective_rain_area_sum(area, infiltration_percent) == pytest.approx(
        expected
    )


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_autodiscovery_sampling_and_noise_variants(tmp_path):
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    toy_dir = _build_toy_hydrograph_dir(tmp_path / "autodiscovery", hydro_ids=("001", "002", "003"))
    sampled = HydroGraphDataset(
        data_dir=toy_dir,
        stats_dir=toy_dir,
        split="train",
        num_samples=2,
        n_time_steps=2,
        k=2,
        noise_type="none",
        hydrograph_ids_file=None,
    )
    assert len(sampled.hydrograph_ids) == 2
    assert set(sampled.hydrograph_ids).issubset({"001", "002", "003"})

    base = np.ones((4, 3), dtype=np.float64)
    none_noise = sampled.apply_noise_to_feature(base.copy(), "none", 0.5)
    correlated = sampled.apply_noise_to_feature(base.copy(), "correlated", 0.5)
    uncorrelated = sampled.apply_noise_to_feature(base.copy(), "uncorrelated", 0.5)
    random_walk = sampled.apply_noise_to_feature(base.copy(), "random_walk", 0.5)
    unknown = sampled.apply_noise_to_feature(base.copy(), "mystery", 0.5)

    assert np.array_equal(none_noise, base)
    assert correlated.shape == base.shape
    assert uncorrelated.shape == base.shape
    assert random_walk.shape == base.shape
    assert np.array_equal(unknown, base)
    assert not np.array_equal(correlated, base)
    assert not np.array_equal(uncorrelated, base)
    assert not np.array_equal(random_walk, base)


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_hydrograph_dataset_contract(hydrograph_dirs):
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        HydroGraphDataset,
        get_hydrograph_feature_layout,
        get_hydrograph_input_dim,
    )

    train_dir, test_dir = hydrograph_dirs
    train_dataset = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=2,
        n_time_steps=2,
        k=2,
        noise_type="none",
        return_physics=True,
        hydrograph_ids_file="train.txt",
    )

    sample = train_dataset[0]
    layout = get_hydrograph_feature_layout(2)
    assert isinstance(sample, Data)
    assert sample.x.shape[1] == get_hydrograph_input_dim(2)
    assert sample.x.shape[1] == layout.input_dim
    assert sample.y.shape[1] == 2
    assert sample.area_denorm.shape[0] == sample.num_nodes
    assert sample.current_water_depth_denorm.shape[0] == sample.num_nodes
    assert sample.current_volume_denorm.shape[0] == sample.num_nodes
    assert hasattr(sample, "physics_current_total_volume")
    assert hasattr(sample, "physics_future_total_volume")
    assert hasattr(sample, "physics_avg_net_source")
    assert hasattr(sample, "physics_next_avg_net_source")
    assert hasattr(sample, "total_area")

    test_dataset = HydroGraphDataset(
        data_dir=test_dir,
        stats_dir=train_dir,
        split="test",
        num_samples=1,
        n_time_steps=2,
        k=2,
        rollout_length=5,
        hydrograph_ids_file="test.txt",
    )
    graph, rollout = test_dataset[0]
    assert graph.x.shape[1] == get_hydrograph_input_dim(2)
    assert rollout["water_depth_gt"].shape[0] == 5
    assert rollout["volume_gt"].shape[0] == 5

    expected_first_depth = HydroGraphDataset.denormalize(
        test_dataset.dynamic_data[0]["water_depth"][2, :],
        test_dataset.dynamic_stats["water_depth"]["mean"],
        test_dataset.dynamic_stats["water_depth"]["std"],
    )
    assert torch.allclose(
        rollout["water_depth_gt"][0],
        torch.tensor(expected_first_depth, dtype=torch.float),
    )


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_tail_windows_are_excluded_when_extra_horizons_are_required(hydrograph_dirs):
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    train_dir, _ = hydrograph_dirs
    base_dataset = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=1,
        n_time_steps=2,
        k=2,
        noise_type="none",
        return_physics=False,
        hydrograph_ids_file="test.txt",
    )
    physics_dataset = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=1,
        n_time_steps=2,
        k=2,
        noise_type="none",
        return_physics=True,
        hydrograph_ids_file="test.txt",
    )
    pushforward_dataset = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=1,
        n_time_steps=2,
        k=2,
        noise_type="pushforward",
        return_physics=False,
        hydrograph_ids_file="test.txt",
    )

    T = base_dataset.dynamic_data[0]["water_depth"].shape[0]
    assert len(base_dataset) == T - 2
    assert len(physics_dataset) == T - 3
    assert len(pushforward_dataset) == T - 3
    assert base_dataset.sample_index[-1][1] == T - 2
    assert physics_dataset.sample_index[-1][1] == T - 3
    assert pushforward_dataset.sample_index[-1][1] == T - 3


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_training_sample_alignment_and_physics_values(hydrograph_dirs):
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        HydroGraphDataset,
        get_hydrograph_feature_layout,
    )

    train_dir, _ = hydrograph_dirs
    dataset = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=1,
        n_time_steps=2,
        k=2,
        noise_type="none",
        return_physics=True,
        hydrograph_ids_file="test.txt",
    )

    graph = dataset[0]
    hydro_idx, anchor_time = dataset.sample_index[0]
    dyn = dataset.dynamic_data[hydro_idx]
    layout = get_hydrograph_feature_layout(2)

    assert torch.allclose(
        graph.x[:, layout.forcing_slice],
        torch.tensor(
            np.tile(
                np.array(
                    [
                        dyn["inflow_hydrograph"][anchor_time],
                        dyn["precipitation"][anchor_time],
                    ]
                ),
                (graph.num_nodes, 1),
            ),
            dtype=torch.float,
        ),
    )

    expected_target = np.stack(
        [
            dyn["water_depth"][anchor_time + 1, :] - dyn["water_depth"][anchor_time, :],
            dyn["volume"][anchor_time + 1, :] - dyn["volume"][anchor_time, :],
        ],
        axis=1,
    )
    assert torch.allclose(graph.y, torch.tensor(expected_target, dtype=torch.float))

    volume_stats = dataset.dynamic_stats["volume"]
    inflow_stats = dataset.dynamic_stats["inflow_hydrograph"]
    precip_stats = dataset.dynamic_stats["precipitation"]
    effective_rain_area_sum = dataset.static_data["effective_rain_area_sum"]

    expected_current_total_volume = float(
        np.sum(
            HydroGraphDataset.denormalize(
                dyn["volume"][anchor_time, :],
                volume_stats["mean"],
                volume_stats["std"],
            )
        )
    )
    expected_future_total_volume = float(
        np.sum(
            HydroGraphDataset.denormalize(
                dyn["volume"][anchor_time + 2, :],
                volume_stats["mean"],
                volume_stats["std"],
            )
        )
    )
    inflow_t = HydroGraphDataset.denormalize(
        dyn["inflow_hydrograph"][anchor_time],
        inflow_stats["mean"],
        inflow_stats["std"],
    )
    inflow_t1 = HydroGraphDataset.denormalize(
        dyn["inflow_hydrograph"][anchor_time + 1],
        inflow_stats["mean"],
        inflow_stats["std"],
    )
    inflow_t2 = HydroGraphDataset.denormalize(
        dyn["inflow_hydrograph"][anchor_time + 2],
        inflow_stats["mean"],
        inflow_stats["std"],
    )
    precip_t = HydroGraphDataset.denormalize(
        dyn["precipitation"][anchor_time],
        precip_stats["mean"],
        precip_stats["std"],
    )
    precip_t1 = HydroGraphDataset.denormalize(
        dyn["precipitation"][anchor_time + 1],
        precip_stats["mean"],
        precip_stats["std"],
    )
    precip_t2 = HydroGraphDataset.denormalize(
        dyn["precipitation"][anchor_time + 2],
        precip_stats["mean"],
        precip_stats["std"],
    )

    expected_avg_source = 0.5 * (inflow_t + inflow_t1) + 0.5 * (
        precip_t + precip_t1
    ) * effective_rain_area_sum
    expected_next_avg_source = 0.5 * (inflow_t1 + inflow_t2) + 0.5 * (
        precip_t1 + precip_t2
    ) * effective_rain_area_sum

    assert graph.physics_current_total_volume.item() == pytest.approx(
        expected_current_total_volume
    )
    assert graph.physics_future_total_volume.item() == pytest.approx(
        expected_future_total_volume
    )
    assert graph.physics_avg_net_source.item() == pytest.approx(expected_avg_source)
    assert graph.physics_next_avg_net_source.item() == pytest.approx(
        expected_next_avg_source
    )


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_pushforward_targets_and_noise_are_correct(hydrograph_dirs):
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    train_dir, _ = hydrograph_dirs
    pushforward_dataset = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=1,
        n_time_steps=3,
        k=2,
        noise_type="pushforward",
        return_physics=False,
        hydrograph_ids_file="test.txt",
    )
    sample = pushforward_dataset[0]
    assert isinstance(sample, Data)
    assert hasattr(sample, "y_pushforward")
    assert sample.y_pushforward.shape == sample.y.shape
    assert sample.next_inflow.shape == torch.Size([1])
    assert sample.next_precipitation.shape == torch.Size([1])

    noisy_dataset = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=1,
        n_time_steps=2,
        k=2,
        noise_type="only_last",
        noise_std=0.5,
        hydrograph_ids_file="test.txt",
    )
    before_depth = noisy_dataset.dynamic_data[0]["water_depth"].copy()
    before_volume = noisy_dataset.dynamic_data[0]["volume"].copy()
    _ = noisy_dataset[0]
    assert np.array_equal(noisy_dataset.dynamic_data[0]["water_depth"], before_depth)
    assert np.array_equal(noisy_dataset.dynamic_data[0]["volume"], before_volume)


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_test_split_rollout_alignment_and_validation(hydrograph_dirs, tmp_path):
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        HydroGraphDataset,
        get_hydrograph_feature_layout,
    )

    train_dir, test_dir = hydrograph_dirs
    _ = HydroGraphDataset(
        data_dir=train_dir,
        stats_dir=train_dir,
        split="train",
        num_samples=1,
        n_time_steps=2,
        k=2,
        hydrograph_ids_file="test.txt",
    )
    dataset = HydroGraphDataset(
        data_dir=test_dir,
        stats_dir=train_dir,
        split="test",
        num_samples=1,
        n_time_steps=2,
        k=2,
        rollout_length=5,
        hydrograph_ids_file="test.txt",
    )
    graph, rollout = dataset[0]
    layout = get_hydrograph_feature_layout(2)
    anchor_time = dataset.n_time_steps - 1
    dyn = dataset.dynamic_data[0]

    assert torch.allclose(
        graph.x[:, layout.forcing_slice],
        torch.tensor(
            np.tile(
                np.array(
                    [
                        dyn["inflow_hydrograph"][anchor_time],
                        dyn["precipitation"][anchor_time],
                    ]
                ),
                (graph.num_nodes, 1),
            ),
            dtype=torch.float,
        ),
    )
    assert torch.allclose(
        rollout["inflow"][0],
        torch.tensor(dyn["inflow_hydrograph"][dataset.n_time_steps], dtype=torch.float),
    )
    assert torch.allclose(
        rollout["precipitation"][0],
        torch.tensor(dyn["precipitation"][dataset.n_time_steps], dtype=torch.float),
    )

    short_dir = _build_toy_hydrograph_dir(
        tmp_path / "short_data", hydro_ids=("001",), num_steps=90
    )
    _ = HydroGraphDataset(
        data_dir=short_dir,
        stats_dir=short_dir,
        split="train",
        num_samples=1,
        n_time_steps=2,
        k=2,
        hydrograph_ids_file="test.txt",
    )
    with pytest.raises(ValueError):
        HydroGraphDataset(
            data_dir=short_dir,
            stats_dir=short_dir,
            split="test",
            num_samples=1,
            n_time_steps=2,
            k=2,
            rollout_length=50,
            hydrograph_ids_file="test.txt",
        )


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_hydrographnet_helper_losses_and_rollout():
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        get_hydrograph_feature_layout,
        get_hydrograph_input_dim,
    )

    utils = _load_example_utils()
    layout = get_hydrograph_feature_layout(2)

    x = torch.zeros(1, get_hydrograph_input_dim(2))
    x[:, layout.forcing_slice] = torch.tensor([[0.1, 0.2]])
    x[:, layout.water_depth_slice] = torch.tensor([[1.0, 2.0]])
    x[:, layout.volume_slice] = torch.tensor([[10.0, 11.0]])
    rolled = utils.roll_feature_window(
        x,
        torch.tensor([[0.5, 1.0]]),
        torch.tensor([0.3]),
        torch.tensor([0.4]),
        2,
        torch.zeros(1, dtype=torch.long),
    )
    assert torch.allclose(rolled[:, layout.forcing_slice], torch.tensor([[0.3, 0.4]]))
    assert torch.allclose(rolled[:, layout.water_depth_slice], torch.tensor([[2.0, 2.5]]))
    assert torch.allclose(rolled[:, layout.volume_slice], torch.tensor([[11.0, 12.0]]))

    graph = Data(
        x=torch.zeros(2, get_hydrograph_input_dim(2)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros(2, 3),
        area_denorm=torch.tensor([[2.0], [2.0]]),
        current_water_depth_denorm=torch.tensor([[1.0], [1.5]]),
        current_volume_denorm=torch.tensor([[2.0], [3.0]]),
        water_depth_std=torch.tensor([1.0]),
        volume_std=torch.tensor([1.0]),
        physics_current_total_volume=torch.tensor([5.0]),
        physics_future_total_volume=torch.tensor([7.0]),
        physics_avg_net_source=torch.tensor([1.5]),
        physics_next_avg_net_source=torch.tensor([0.5]),
        total_area=torch.tensor([4.0]),
    )
    graph.batch = torch.tensor([0, 0], dtype=torch.long)

    pred = torch.tensor([[0.25, 0.5], [0.5, 1.0]])
    target = pred.clone()
    prediction = utils.compute_prediction_loss(pred, target, graph)
    assert prediction["prediction_loss"].item() == pytest.approx(0.0)
    assert utils.compute_depth_volume_penalty(pred, graph).item() == pytest.approx(0.0)
    assert utils.compute_physics_loss(pred, graph, delta_t=1.0).item() == pytest.approx(0.0)


@requires_module(["torch_geometric", "scipy", "tqdm"])
def test_helper_weighted_loss_and_batched_rollout():
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import (
        get_hydrograph_feature_layout,
        get_hydrograph_input_dim,
    )

    utils = _load_example_utils()
    layout = get_hydrograph_feature_layout(2)

    x = torch.zeros(3, get_hydrograph_input_dim(2))
    x[:, layout.water_depth_slice] = torch.tensor(
        [[1.0, 2.0], [1.5, 2.5], [3.0, 4.0]]
    )
    x[:, layout.volume_slice] = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
    )
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    rolled = utils.roll_feature_window(
        x,
        torch.tensor([[0.5, 1.0], [1.0, 2.0], [0.25, 0.5]]),
        torch.tensor([0.3, 0.8]),
        torch.tensor([0.4, 0.9]),
        2,
        batch,
    )
    assert torch.allclose(
        rolled[:, layout.forcing_slice],
        torch.tensor([[0.3, 0.4], [0.3, 0.4], [0.8, 0.9]]),
    )
    assert torch.allclose(
        rolled[:, layout.water_depth_slice],
        torch.tensor([[2.0, 2.5], [2.5, 3.5], [4.0, 4.25]]),
    )
    assert torch.allclose(
        rolled[:, layout.volume_slice],
        torch.tensor([[11.0, 12.0], [21.0, 23.0], [31.0, 31.5]]),
    )

    graph = Data(
        x=torch.zeros(2, get_hydrograph_input_dim(2)),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros(2, 3),
        area_denorm=torch.tensor([[1.0], [1.0]]),
        current_water_depth_denorm=torch.tensor([[0.0], [0.0]]),
        current_volume_denorm=torch.tensor([[0.0], [0.0]]),
        water_depth_std=torch.tensor([1.0]),
        volume_std=torch.tensor([1.0]),
        physics_current_total_volume=torch.tensor([0.0]),
        physics_future_total_volume=torch.tensor([0.0]),
        physics_avg_net_source=torch.tensor([0.0]),
        physics_next_avg_net_source=torch.tensor([0.0]),
        total_area=torch.tensor([1.0]),
    )
    graph.batch = torch.tensor([0, 0], dtype=torch.long)
    pred = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    target = torch.zeros_like(pred)

    total_loss, loss_dict = utils.compute_one_step_loss(
        pred,
        target,
        graph,
        use_physics_loss=True,
        delta_t=1.0,
        physics_penalty_weight=2.0,
        depth_volume_penalty_weight=3.0,
    )
    assert loss_dict["prediction_loss"].item() == pytest.approx(0.5)
    assert loss_dict["depth_volume_penalty"].item() == pytest.approx(0.5)
    assert loss_dict["physics_loss"].item() == pytest.approx(1.0)
    assert total_loss.item() == pytest.approx(4.0)


@requires_module(
    ["torch_geometric", "scipy", "tqdm", "jaxtyping", "nvtx"]
)
def test_utils_batch_helpers_and_build_model():
    utils = _load_example_utils()

    graph = Data(x=torch.randn(3, 4))
    batch = utils.get_batch_vector(graph)
    assert torch.equal(batch, torch.zeros(3, dtype=torch.long))

    zero_sum = utils._sum_by_graph(
        torch.zeros(0, dtype=torch.float), torch.zeros(0, dtype=torch.long)
    )
    assert zero_sum.numel() == 0

    class _FakeDDP:
        def __init__(self, module):
            self.module = module

    wrapped = object()
    original_ddp = utils.DistributedDataParallel
    utils.DistributedDataParallel = _FakeDDP
    try:
        assert utils.unwrap_model(_FakeDDP(wrapped)) is wrapped
    finally:
        utils.DistributedDataParallel = original_ddp

    cfg = _Cfg(
        n_time_steps=2,
        num_input_features=None,
        num_edge_features=3,
        num_output_features=2,
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
        recompute_activation=True,
    )
    model = utils.build_model(cfg)
    assert model.node_encoder is not None


@requires_module(
    ["torch_geometric", "scipy", "tqdm", "jaxtyping", "nvtx"]
)
def test_build_model_validates_input_feature_count():
    utils = _load_example_utils()
    cfg = _Cfg(
        n_time_steps=2,
        num_input_features=999,
        num_edge_features=3,
        num_output_features=2,
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
        recompute_activation=False,
    )

    with pytest.raises(ValueError):
        utils.build_model(cfg)
