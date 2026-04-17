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

r"""Shared data-loading backends for FloodForecaster datasets."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is an explicit dependency for the example
    h5py = None


SCHEMA_VERSION = 1
CACHE_FILE_NAME = "flood_forecaster_v1.h5"
MANIFEST_FILE_NAME = "manifest.json"


def _supports_tqdm_output(stream: Any) -> bool:
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except OSError:
            return False
    return False


def _cache_status_message(message: str) -> None:
    print(f"[FloodForecaster cache] {message}")


def _read_run_ids(list_path: Path) -> List[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"Expected run list at {list_path}, not found.")
    try:
        with open(list_path, "r", encoding="utf-8-sig") as f:
            lines = [line.strip().lstrip("\ufeff") for line in f if line.strip()]
    except OSError as e:
        raise IOError(f"Failed to read run list from {list_path}: {e}") from e

    run_ids = lines[0].split(",") if len(lines) == 1 and "," in lines[0] else lines
    run_ids = [run_id.strip().lstrip("\ufeff") for run_id in run_ids if run_id.strip()]
    if not run_ids:
        raise ValueError(f"No valid run IDs found in {list_path}")
    return run_ids


def _path_record(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "size": None, "mtime_ns": None}
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _gather_file_records(
    data_root: Path,
    list_file_name: str,
    xy_file: Optional[str],
    static_files: Iterable[str],
    dynamic_patterns: Dict[str, str],
    boundary_patterns: Dict[str, str],
    run_ids: Iterable[str],
    dynamic_keys: Iterable[str],
    boundary_keys: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}

    def add_record(path: Path) -> None:
        rel_path = path.relative_to(data_root).as_posix()
        records[rel_path] = _path_record(path)

    add_record(data_root / list_file_name)
    if xy_file:
        add_record(data_root / xy_file)
    for fname in static_files:
        add_record(data_root / fname)
    for run_id in run_ids:
        for key in dynamic_keys:
            pattern = dynamic_patterns.get(key)
            if pattern:
                add_record(data_root / pattern.format(run_id))
        for key in boundary_keys:
            pattern = boundary_patterns.get(key)
            if pattern:
                add_record(data_root / pattern.format(run_id))
    return records


def _normalize_geometry_array(xy_arr: np.ndarray) -> np.ndarray:
    min_xy = xy_arr.min(axis=0)
    max_xy = xy_arr.max(axis=0)
    range_xy = max_xy - min_xy
    range_xy[range_xy == 0] = 1.0
    return (xy_arr - min_xy) / range_xy


def _load_xy_tensor(data_root: Path, xy_file: Optional[str]) -> torch.Tensor:
    if not xy_file:
        raise ValueError("xy_file was not provided. Please specify it in the config.")
    xy_path = data_root / xy_file
    if not xy_path.exists():
        raise FileNotFoundError(f"Reference XY file not found: {xy_path}")

    xy_arr = np.loadtxt(str(xy_path), delimiter="\t", dtype=np.float32)
    if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
        raise ValueError(f"{xy_file} must have shape (num_cells, 2). Got {xy_arr.shape}.")
    return torch.from_numpy(_normalize_geometry_array(xy_arr)).float()


def _trim_feature_rows(
    arr: np.ndarray,
    reference_cell_count: int,
    fname: str,
    raise_on_smaller: bool,
) -> Optional[np.ndarray]:
    if arr.ndim == 1:
        arr = arr[:, None]

    n_rows = arr.shape[0]
    if n_rows < reference_cell_count:
        msg = f"{fname} has {n_rows} < {reference_cell_count}"
        if raise_on_smaller:
            raise ValueError(msg)
        warnings.warn(msg + " -> skipping.")
        return None
    if n_rows > reference_cell_count:
        arr = arr[:reference_cell_count, :]
    return arr.astype(np.float32, copy=False)


def _trim_dynamic_columns(
    arr: np.ndarray,
    reference_cell_count: int,
    fname: str,
    raise_on_smaller: bool,
) -> Optional[np.ndarray]:
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    n_cols = arr.shape[1]
    if n_cols < reference_cell_count:
        msg = f"{fname} has {n_cols} < {reference_cell_count}"
        if raise_on_smaller:
            raise ValueError(msg)
        warnings.warn(msg + " -> skipping.")
        return None
    if n_cols > reference_cell_count:
        arr = arr[:, :reference_cell_count]
    return arr.astype(np.float32, copy=False)


def _load_static_tensor(
    data_root: Path,
    static_files: Iterable[str],
    reference_cell_count: int,
    raise_on_smaller: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
    static_arrays: List[np.ndarray] = []
    static_feature_names: List[str] = []
    cell_area_tensor: Optional[torch.Tensor] = None

    for fname in static_files:
        fpath = data_root / fname
        if not fpath.exists():
            warnings.warn(f"Static file not found: {fpath}, skipping.")
            continue
        arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)
        arr = _trim_feature_rows(arr, reference_cell_count, fname, raise_on_smaller)
        if arr is None:
            continue
        static_arrays.append(arr)
        static_feature_names.append(str(fname))
        if Path(fname).name == "M40_CA.txt":
            cell_area_tensor = torch.from_numpy(arr.reshape(-1).copy()).float()

    if static_arrays:
        static_tensor = torch.from_numpy(np.concatenate(static_arrays, axis=1).copy()).float()
    else:
        static_tensor = torch.zeros((reference_cell_count, 0), dtype=torch.float32)
    return static_tensor, cell_area_tensor, static_feature_names


def _load_run_arrays(
    data_root: Path,
    run_id: str,
    dynamic_patterns: Dict[str, str],
    boundary_patterns: Dict[str, str],
    reference_cell_count: int,
    raise_on_smaller: bool,
    dynamic_keys: List[str],
    boundary_keys: List[str],
) -> Dict[str, Any]:
    dynamic_arrays: List[np.ndarray] = []
    boundary_arrays: List[np.ndarray] = []
    available_dynamic_keys: List[str] = []
    available_boundary_keys: List[str] = []

    for key in dynamic_keys:
        pattern = dynamic_patterns.get(key)
        if pattern is None:
            continue
        fpath = data_root / pattern.format(run_id)
        if not fpath.exists():
            warnings.warn(f"Dynamic file not found: {fpath}, skipping {key}.")
            continue
        arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)
        arr = _trim_dynamic_columns(arr, reference_cell_count, fpath.name, raise_on_smaller)
        if arr is None:
            continue
        dynamic_arrays.append(arr)
        available_dynamic_keys.append(key)

    for key in boundary_keys:
        pattern = boundary_patterns.get(key)
        if pattern is None:
            continue
        fpath = data_root / pattern.format(run_id)
        if not fpath.exists():
            warnings.warn(f"Boundary file not found: {fpath}, skipping {key}.")
            continue
        arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] == 2:
            arr = arr[:, 1:2]
        boundary_arrays.append(arr.astype(np.float32, copy=False))
        available_boundary_keys.append(key)

    dynamic_tensor: Optional[torch.Tensor] = None
    boundary_tensor: Optional[torch.Tensor] = None
    dynamic_length = 0
    boundary_length = 0

    if dynamic_arrays:
        dynamic_lengths = {arr.shape[0] for arr in dynamic_arrays}
        if len(dynamic_lengths) != 1:
            raise ValueError(f"Run {run_id} has inconsistent dynamic sequence lengths: {sorted(dynamic_lengths)}")
        dynamic_length = next(iter(dynamic_lengths))
        dynamic_tensor = torch.from_numpy(np.stack(dynamic_arrays, axis=-1).copy()).float()

    if boundary_arrays:
        boundary_lengths = {arr.shape[0] for arr in boundary_arrays}
        if len(boundary_lengths) != 1:
            raise ValueError(f"Run {run_id} has inconsistent boundary sequence lengths: {sorted(boundary_lengths)}")
        boundary_length = next(iter(boundary_lengths))
        boundary_tensor = torch.from_numpy(np.concatenate(boundary_arrays, axis=1).copy()).float()

    sequence_length = 0
    if dynamic_tensor is not None and boundary_tensor is not None:
        sequence_length = min(dynamic_tensor.shape[0], boundary_tensor.shape[0])
        if dynamic_tensor.shape[0] != boundary_tensor.shape[0]:
            warnings.warn(
                f"Run {run_id} has mismatched dynamic/boundary lengths "
                f"({dynamic_tensor.shape[0]} vs {boundary_tensor.shape[0]}). "
                f"Truncating both to {sequence_length}."
            )
        dynamic_tensor = dynamic_tensor[:sequence_length]
        boundary_tensor = boundary_tensor[:sequence_length]
        dynamic_length = sequence_length
        boundary_length = sequence_length

    return {
        "dynamic": dynamic_tensor,
        "boundary": boundary_tensor,
        "available_dynamic_keys": available_dynamic_keys,
        "available_boundary_keys": available_boundary_keys,
        "dynamic_length": dynamic_length,
        "boundary_length": boundary_length,
        "sequence_length": sequence_length,
    }


def _build_manifest_payload(
    data_root: Path,
    list_file_name: str,
    run_ids: List[str],
    reference_cell_count: int,
    static_feature_names: List[str],
    dynamic_keys: List[str],
    boundary_keys: List[str],
    dynamic_patterns: Dict[str, str],
    boundary_patterns: Dict[str, str],
    file_records: Dict[str, Dict[str, Any]],
    run_metadata: Dict[str, Dict[str, Any]],
    has_cell_area: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "data_root": str(data_root.resolve()),
        "list_file_name": list_file_name,
        "run_ids": run_ids,
        "sequence_lengths": {
            run_id: int(meta.get("sequence_length", 0))
            for run_id, meta in run_metadata.items()
        },
        "reference_cell_count": int(reference_cell_count),
        "static_feature_names": static_feature_names,
        "requested_static_files": static_feature_names,
        "dynamic_keys": dynamic_keys,
        "boundary_keys": boundary_keys,
        "dynamic_patterns": dynamic_patterns,
        "boundary_patterns": boundary_patterns,
        "files": file_records,
        "run_metadata": run_metadata,
        "has_cell_area": bool(has_cell_area),
        "cache_file_name": CACHE_FILE_NAME,
    }


def _manifest_matches_inputs(
    manifest: Dict[str, Any],
    *,
    data_root: Path,
    list_file_name: str,
    run_ids: List[str],
    static_files: List[str],
    dynamic_keys: List[str],
    boundary_keys: List[str],
    dynamic_patterns: Dict[str, str],
    boundary_patterns: Dict[str, str],
    file_records: Dict[str, Dict[str, Any]],
) -> bool:
    return (
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("data_root") == str(data_root.resolve())
        and manifest.get("list_file_name") == list_file_name
        and manifest.get("run_ids") == run_ids
        and manifest.get("dynamic_keys") == dynamic_keys
        and manifest.get("boundary_keys") == boundary_keys
        and manifest.get("dynamic_patterns") == dynamic_patterns
        and manifest.get("boundary_patterns") == boundary_patterns
        and manifest.get("requested_static_files") == list(static_files)
        and manifest.get("files") == file_records
    )


@contextmanager
def _cache_lock(lock_path: Path, timeout_seconds: float = 900.0):
    start_time = time.time()
    lock_fd: Optional[int] = None
    while lock_fd is None:
        try:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(lock_fd, str(os.getpid()).encode("utf-8"))
        except FileExistsError:
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for cache lock {lock_path}")
            time.sleep(0.5)
    try:
        yield
    finally:
        try:
            if lock_fd is not None:
                os.close(lock_fd)
        finally:
            if lock_path.exists():
                lock_path.unlink()


def _load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(manifest_path)


def _resolve_dynamic_keys(dynamic_patterns: Dict[str, str]) -> List[str]:
    default_order = ["WD", "VX", "VY"]
    keys = [key for key in default_order if key in dynamic_patterns]
    return keys if keys else list(dynamic_patterns.keys())


def _resolve_boundary_keys(boundary_patterns: Dict[str, str]) -> List[str]:
    keys = sorted(boundary_patterns.keys())
    return keys


def prepare_flood_cache(
    data_root,
    *,
    list_file_name: str,
    xy_file: Optional[str],
    static_files: Optional[List[str]] = None,
    dynamic_patterns: Optional[Dict[str, str]] = None,
    boundary_patterns: Optional[Dict[str, str]] = None,
    raise_on_smaller: bool = True,
    cache_dir_name: str = ".flood_cache",
    rebuild: bool = False,
) -> Dict[str, Any]:
    r"""Build or reuse the shared HDF5 cache for a FloodForecaster data root."""

    if h5py is None:
        raise ImportError("h5py is required to build the FloodForecaster cache backend.")

    data_root = Path(data_root)
    static_files = list(static_files or [])
    dynamic_patterns = dict(dynamic_patterns or {})
    boundary_patterns = dict(boundary_patterns or {})
    dynamic_keys = _resolve_dynamic_keys(dynamic_patterns)
    boundary_keys = _resolve_boundary_keys(boundary_patterns)

    cache_dir = data_root / cache_dir_name
    manifest_path = cache_dir / MANIFEST_FILE_NAME
    h5_path = cache_dir / CACHE_FILE_NAME
    lock_path = cache_dir / ".build.lock"
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_ids = _read_run_ids(data_root / list_file_name)
    file_records = _gather_file_records(
        data_root,
        list_file_name,
        xy_file,
        static_files,
        dynamic_patterns,
        boundary_patterns,
        run_ids,
        dynamic_keys,
        boundary_keys,
    )

    manifest = _load_manifest(manifest_path)
    if (
        not rebuild
        and manifest is not None
        and h5_path.exists()
        and _manifest_matches_inputs(
            manifest,
            data_root=data_root,
            list_file_name=list_file_name,
            run_ids=run_ids,
            static_files=static_files,
            dynamic_keys=dynamic_keys,
            boundary_keys=boundary_keys,
            dynamic_patterns=dynamic_patterns,
            boundary_patterns=boundary_patterns,
            file_records=file_records,
        )
    ):
        _cache_status_message(f"Reusing cache at {h5_path}")
        return manifest

    with _cache_lock(lock_path):
        manifest = _load_manifest(manifest_path)
        if (
            not rebuild
            and manifest is not None
            and h5_path.exists()
            and _manifest_matches_inputs(
                manifest,
                data_root=data_root,
                list_file_name=list_file_name,
                run_ids=run_ids,
                static_files=static_files,
                dynamic_keys=dynamic_keys,
                boundary_keys=boundary_keys,
                dynamic_patterns=dynamic_patterns,
                boundary_patterns=boundary_patterns,
                file_records=file_records,
            )
        ):
            _cache_status_message(f"Reusing cache at {h5_path}")
            return manifest

        _cache_status_message(
            f"Building cache at {h5_path} from {len(run_ids)} runs"
        )
        geometry = _load_xy_tensor(data_root, xy_file)
        static_tensor, cell_area_tensor, static_feature_names = _load_static_tensor(
            data_root,
            static_files,
            reference_cell_count=geometry.shape[0],
            raise_on_smaller=raise_on_smaller,
        )

        tmp_h5_path = h5_path.with_suffix(".tmp")
        if tmp_h5_path.exists():
            tmp_h5_path.unlink()

        run_metadata: Dict[str, Dict[str, Any]] = {}
        with h5py.File(tmp_h5_path, "w") as h5_file:
            h5_file.create_dataset("geometry", data=geometry.numpy(), dtype=np.float32)
            h5_file.create_dataset("static", data=static_tensor.numpy(), dtype=np.float32)
            if cell_area_tensor is not None:
                h5_file.create_dataset("cell_area", data=cell_area_tensor.numpy(), dtype=np.float32)

            runs_group = h5_file.create_group("runs")
            progress = tqdm(
                run_ids,
                desc="Caching flood runs",
                file=sys.stdout,
                disable=not _supports_tqdm_output(sys.stdout),
            )
            for run_id in progress:
                run_data = _load_run_arrays(
                    data_root=data_root,
                    run_id=run_id,
                    dynamic_patterns=dynamic_patterns,
                    boundary_patterns=boundary_patterns,
                    reference_cell_count=geometry.shape[0],
                    raise_on_smaller=raise_on_smaller,
                    dynamic_keys=dynamic_keys,
                    boundary_keys=boundary_keys,
                )
                run_metadata[run_id] = {
                    "available_dynamic_keys": run_data["available_dynamic_keys"],
                    "available_boundary_keys": run_data["available_boundary_keys"],
                    "dynamic_length": int(run_data["dynamic_length"]),
                    "boundary_length": int(run_data["boundary_length"]),
                    "sequence_length": int(run_data["sequence_length"]),
                }
                if (
                    run_data["dynamic"] is None
                    or run_data["boundary"] is None
                    or run_data["available_dynamic_keys"] != dynamic_keys
                    or run_data["available_boundary_keys"] != boundary_keys
                    or run_data["sequence_length"] <= 0
                ):
                    continue

                run_group = runs_group.create_group(run_id)
                run_group.create_dataset(
                    "dynamic",
                    data=run_data["dynamic"].numpy(),
                    dtype=np.float32,
                )
                run_group.create_dataset(
                    "boundary",
                    data=run_data["boundary"].numpy(),
                    dtype=np.float32,
                )

        tmp_h5_path.replace(h5_path)

        manifest = _build_manifest_payload(
            data_root=data_root,
            list_file_name=list_file_name,
            run_ids=run_ids,
            reference_cell_count=geometry.shape[0],
            static_feature_names=static_feature_names,
            dynamic_keys=dynamic_keys,
            boundary_keys=boundary_keys,
            dynamic_patterns=dynamic_patterns,
            boundary_patterns=boundary_patterns,
            file_records=file_records,
            run_metadata=run_metadata,
            has_cell_area=cell_area_tensor is not None,
        )
        manifest["requested_static_files"] = static_files
        _write_manifest(manifest_path, manifest)
        _cache_status_message(f"Finished cache build at {h5_path}")
        return manifest


class HDF5RunStore:
    r"""Run-backed accessor for HDF5 cached FloodForecaster data."""

    def __init__(self, cache_path, manifest: Dict[str, Any]):
        if h5py is None:
            raise ImportError("h5py is required to use the HDF5 FloodForecaster cache.")
        self.cache_path = Path(cache_path)
        self.manifest = manifest
        self._file = None
        self._pid = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def _ensure_open(self):
        current_pid = os.getpid()
        if self._file is None or self._pid != current_pid:
            self.close()
            self._file = h5py.File(self.cache_path, "r")
            self._pid = current_pid
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._pid = None

    def load_shared_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        with h5py.File(self.cache_path, "r") as h5_file:
            geometry = torch.from_numpy(h5_file["geometry"][...].copy()).float()
            static = torch.from_numpy(h5_file["static"][...].copy()).float()
            cell_area = None
            if "cell_area" in h5_file:
                cell_area = torch.from_numpy(h5_file["cell_area"][...].copy()).float()
        return geometry, static, cell_area

    def load_run(self, run_id: str) -> Dict[str, torch.Tensor]:
        h5_file = self._ensure_open()
        group = h5_file["runs"][run_id]
        return {
            "dynamic": torch.from_numpy(group["dynamic"][...].copy()).float(),
            "boundary": torch.from_numpy(group["boundary"][...].copy()).float(),
        }


class RawTextRunStore:
    r"""On-demand raw text loader for FloodForecaster data."""

    def __init__(
        self,
        data_root,
        *,
        xy_file: Optional[str],
        static_files: Optional[List[str]] = None,
        dynamic_patterns: Optional[Dict[str, str]] = None,
        boundary_patterns: Optional[Dict[str, str]] = None,
        raise_on_smaller: bool = True,
    ):
        self.data_root = Path(data_root)
        self.xy_file = xy_file
        self.static_files = list(static_files or [])
        self.dynamic_patterns = dict(dynamic_patterns or {})
        self.boundary_patterns = dict(boundary_patterns or {})
        self.raise_on_smaller = raise_on_smaller
        self.dynamic_keys = _resolve_dynamic_keys(self.dynamic_patterns)
        self.boundary_keys = _resolve_boundary_keys(self.boundary_patterns)
        self._shared_cache: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[str]]] = None

    def load_shared_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self._shared_cache is None:
            geometry = _load_xy_tensor(self.data_root, self.xy_file)
            static, cell_area, static_feature_names = _load_static_tensor(
                self.data_root,
                self.static_files,
                reference_cell_count=geometry.shape[0],
                raise_on_smaller=self.raise_on_smaller,
            )
            self._shared_cache = (geometry, static, cell_area, static_feature_names)
        geometry, static, cell_area, _ = self._shared_cache
        return geometry, static, cell_area

    def get_static_feature_names(self) -> List[str]:
        self.load_shared_tensors()
        assert self._shared_cache is not None
        return list(self._shared_cache[3])

    def inspect_runs(self, list_file_name: str) -> Dict[str, Any]:
        geometry, _, cell_area = self.load_shared_tensors()
        run_ids = _read_run_ids(self.data_root / list_file_name)
        file_records = _gather_file_records(
            self.data_root,
            list_file_name,
            self.xy_file,
            self.static_files,
            self.dynamic_patterns,
            self.boundary_patterns,
            run_ids,
            self.dynamic_keys,
            self.boundary_keys,
        )

        run_metadata: Dict[str, Dict[str, Any]] = {}
        for run_id in run_ids:
            run_data = _load_run_arrays(
                data_root=self.data_root,
                run_id=run_id,
                dynamic_patterns=self.dynamic_patterns,
                boundary_patterns=self.boundary_patterns,
                reference_cell_count=geometry.shape[0],
                raise_on_smaller=self.raise_on_smaller,
                dynamic_keys=self.dynamic_keys,
                boundary_keys=self.boundary_keys,
            )
            run_metadata[run_id] = {
                "available_dynamic_keys": run_data["available_dynamic_keys"],
                "available_boundary_keys": run_data["available_boundary_keys"],
                "dynamic_length": int(run_data["dynamic_length"]),
                "boundary_length": int(run_data["boundary_length"]),
                "sequence_length": int(run_data["sequence_length"]),
            }

        manifest = _build_manifest_payload(
            data_root=self.data_root,
            list_file_name=list_file_name,
            run_ids=run_ids,
            reference_cell_count=geometry.shape[0],
            static_feature_names=self.get_static_feature_names(),
            dynamic_keys=self.dynamic_keys,
            boundary_keys=self.boundary_keys,
            dynamic_patterns=self.dynamic_patterns,
            boundary_patterns=self.boundary_patterns,
            file_records=file_records,
            run_metadata=run_metadata,
            has_cell_area=cell_area is not None,
        )
        manifest["requested_static_files"] = list(self.static_files)
        return manifest

    def load_run(self, run_id: str) -> Dict[str, torch.Tensor]:
        geometry, _, _ = self.load_shared_tensors()
        run_data = _load_run_arrays(
            data_root=self.data_root,
            run_id=run_id,
            dynamic_patterns=self.dynamic_patterns,
            boundary_patterns=self.boundary_patterns,
            reference_cell_count=geometry.shape[0],
            raise_on_smaller=self.raise_on_smaller,
            dynamic_keys=self.dynamic_keys,
            boundary_keys=self.boundary_keys,
        )
        if run_data["dynamic"] is None or run_data["boundary"] is None:
            raise KeyError(f"Run {run_id} is missing required dynamic or boundary data.")
        return {
            "dynamic": run_data["dynamic"],
            "boundary": run_data["boundary"],
        }


def create_run_store(
    data_root,
    *,
    list_file_name: str,
    xy_file: Optional[str],
    static_files: Optional[List[str]] = None,
    dynamic_patterns: Optional[Dict[str, str]] = None,
    boundary_patterns: Optional[Dict[str, str]] = None,
    raise_on_smaller: bool = True,
    backend: str = "auto",
    cache_dir_name: str = ".flood_cache",
    rebuild_cache: bool = False,
) -> Tuple[Any, Dict[str, Any], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""Create a run store plus dataset metadata for the requested backend."""

    static_files = list(static_files or [])
    dynamic_patterns = dict(dynamic_patterns or {})
    boundary_patterns = dict(boundary_patterns or {})

    normalized_backend = str(backend).lower()
    if normalized_backend == "auto":
        normalized_backend = "hdf5" if h5py is not None else "raw_txt"

    if normalized_backend == "hdf5":
        manifest = prepare_flood_cache(
            data_root,
            list_file_name=list_file_name,
            xy_file=xy_file,
            static_files=static_files,
            dynamic_patterns=dynamic_patterns,
            boundary_patterns=boundary_patterns,
            raise_on_smaller=raise_on_smaller,
            cache_dir_name=cache_dir_name,
            rebuild=rebuild_cache,
        )
        cache_path = Path(data_root) / cache_dir_name / CACHE_FILE_NAME
        store = HDF5RunStore(cache_path, manifest)
        geometry, static, cell_area = store.load_shared_tensors()
        return store, manifest, geometry, static, cell_area

    if normalized_backend != "raw_txt":
        raise ValueError(f"Unknown FloodForecaster data backend '{backend}'.")

    store = RawTextRunStore(
        data_root,
        xy_file=xy_file,
        static_files=static_files,
        dynamic_patterns=dynamic_patterns,
        boundary_patterns=boundary_patterns,
        raise_on_smaller=raise_on_smaller,
    )
    geometry, static, cell_area = store.load_shared_tensors()
    manifest = store.inspect_runs(list_file_name=list_file_name)
    return store, manifest, geometry, static, cell_area
