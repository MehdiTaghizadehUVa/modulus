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

import errno
import json
import os
import socket
import sys
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is an explicit dependency for the example
    h5py = None


SCHEMA_VERSION = 1
CACHE_FILE_NAME = "flood_forecaster_v1.h5"
MANIFEST_FILE_NAME = "manifest.json"
BUILD_STATUS_FILE_NAME = ".build.status.json"
CACHE_LOCK_TIMEOUT_SECONDS = 7200.0
CACHE_LOCK_STALE_SECONDS = 300.0
CACHE_LOCK_HEARTBEAT_SECONDS = 5.0


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


def _read_lock_owner(
    lock_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    r"""Read lock metadata and its last heartbeat time."""
    try:
        stat = lock_path.stat()
        with open(lock_path, "r", encoding="utf-8") as lock_file:
            owner = json.load(lock_file)
    except FileNotFoundError:
        return None, None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        try:
            return None, lock_path.stat().st_mtime
        except OSError:
            return None, None
    return owner if isinstance(owner, dict) else None, stat.st_mtime


def _local_process_is_alive(pid: Any) -> bool:
    r"""Return whether ``pid`` identifies a live process on this host."""
    try:
        resolved_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if resolved_pid <= 0:
        return False

    if os.name == "nt":
        # os.kill(pid, 0) is not a safe liveness probe on Windows. Query the
        # process handle without requesting termination or mutation rights.
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        process_handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            resolved_pid,
        )
        if not process_handle:
            return ctypes.get_last_error() == 5  # Access denied implies it exists.
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(
                process_handle,
                ctypes.byref(exit_code),
            ):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(process_handle)

    try:
        os.kill(resolved_pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _lock_is_stale(
    lock_path: Path,
    *,
    stale_after_seconds: float,
) -> bool:
    r"""Determine whether a lock owner is dead or stopped heartbeating."""
    owner, heartbeat_time = _read_lock_owner(lock_path)
    if heartbeat_time is None:
        return False

    if owner and owner.get("hostname") == socket.gethostname():
        return not _local_process_is_alive(owner.get("pid"))

    return time.time() - heartbeat_time > stale_after_seconds


def _lock_owner_description(lock_path: Path) -> str:
    owner, heartbeat_time = _read_lock_owner(lock_path)
    if owner is None:
        return "owner metadata unavailable"
    heartbeat_age = (
        max(0.0, time.time() - heartbeat_time)
        if heartbeat_time is not None
        else float("nan")
    )
    return (
        f"token={owner.get('token', 'unknown')}, pid={owner.get('pid', 'unknown')}, "
        f"host={owner.get('hostname', 'unknown')}, heartbeat_age={heartbeat_age:.1f}s"
    )


def _create_owned_lock(lock_path: Path, token: str) -> None:
    owner = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "token": token,
        "created_at": time.time(),
    }
    lock_fd: Optional[int] = None
    created = False
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        created = True
        os.write(lock_fd, json.dumps(owner, sort_keys=True).encode("utf-8"))
        os.fsync(lock_fd)
    except Exception:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_fd = None
        if created:
            try:
                lock_path.unlink()
            except OSError:
                pass
        raise
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def _remove_lock_if_owned(lock_path: Path, token: str) -> bool:
    owner, _ = _read_lock_owner(lock_path)
    if owner is None or owner.get("token") != token:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return False
    return True


def _try_reclaim_stale_lock(
    lock_path: Path,
    *,
    stale_after_seconds: float,
) -> bool:
    r"""Serialize stale-lock recovery and remove only a revalidated stale lock."""
    recovery_path = lock_path.with_name(f"{lock_path.name}.recovery")
    recovery_token = uuid.uuid4().hex
    try:
        _create_owned_lock(recovery_path, recovery_token)
    except FileExistsError:
        if _lock_is_stale(
            recovery_path,
            stale_after_seconds=max(30.0, stale_after_seconds),
        ):
            stale_owner, _ = _read_lock_owner(recovery_path)
            if stale_owner is None:
                try:
                    recovery_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                _remove_lock_if_owned(recovery_path, str(stale_owner.get("token")))
        return False

    try:
        if not _lock_is_stale(
            lock_path,
            stale_after_seconds=stale_after_seconds,
        ):
            return False
        stale_owner = _lock_owner_description(lock_path)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            return True
        _cache_status_message(f"Reclaimed stale cache lock ({stale_owner})")
        return True
    finally:
        _remove_lock_if_owned(recovery_path, recovery_token)


def _heartbeat_cache_lock(
    lock_path: Path,
    token: str,
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    while not stop_event.wait(interval_seconds):
        owner, _ = _read_lock_owner(lock_path)
        if owner is None or owner.get("token") != token:
            return
        try:
            os.utime(lock_path, None)
        except OSError:
            return


@contextmanager
def _cache_lock(
    lock_path: Path,
    timeout_seconds: float = CACHE_LOCK_TIMEOUT_SECONDS,
    stale_after_seconds: float = CACHE_LOCK_STALE_SECONDS,
    heartbeat_interval_seconds: float = CACHE_LOCK_HEARTBEAT_SECONDS,
):
    r"""Acquire a heartbeat-backed, stale-recoverable cache build lock."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive.")
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive.")
    if heartbeat_interval_seconds >= stale_after_seconds:
        raise ValueError(
            "heartbeat_interval_seconds must be shorter than stale_after_seconds."
        )

    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            _create_owned_lock(lock_path, token)
            break
        except FileExistsError:
            if _try_reclaim_stale_lock(
                lock_path,
                stale_after_seconds=stale_after_seconds,
            ):
                continue
            if time.monotonic() >= deadline:
                owner = _lock_owner_description(lock_path)
                raise TimeoutError(
                    f"Timed out after {timeout_seconds:.1f}s waiting for cache lock "
                    f"{lock_path} ({owner})."
                )
            time.sleep(min(0.5, max(0.01, deadline - time.monotonic())))

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_cache_lock,
        args=(lock_path, token, stop_event, heartbeat_interval_seconds),
        name="flood-cache-lock-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        yield token
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=max(1.0, heartbeat_interval_seconds * 2.0))
        _remove_lock_if_owned(lock_path, token)


def _load_manifest(manifest_path: Path) -> Optional[Dict[str, Any]]:
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
    except json.JSONDecodeError:
        warnings.warn(
            f"Ignoring malformed cache metadata at {manifest_path}; it will be rebuilt."
        )
        return None
    return payload if isinstance(payload, dict) else None


def _write_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    _write_json_atomic(manifest_path, payload)


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.flush()
            os.fsync(output_file.fileno())
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _distributed_cache_context() -> Optional[Tuple[int, int]]:
    if not dist.is_available() or not dist.is_initialized():
        return None
    world_size = dist.get_world_size()
    if world_size <= 1:
        return None
    return dist.get_rank(), world_size


def _broadcast_cache_plan(
    plan: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    plan_holder = [plan]
    dist.broadcast_object_list(plan_holder, src=0)
    resolved_plan = plan_holder[0]
    if not isinstance(resolved_plan, dict):
        raise RuntimeError("Rank zero broadcast an invalid FloodForecaster cache plan.")
    return resolved_plan


def _distributed_cache_barrier() -> None:
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def _write_build_status(
    status_path: Path,
    *,
    build_id: str,
    state: str,
    lock_token: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    payload = {
        "build_id": build_id,
        "state": state,
        "rank": 0,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "updated_at": time.time(),
    }
    if lock_token is not None:
        payload["lock_token"] = lock_token
    if error is not None:
        payload["error"] = error
    _write_json_atomic(status_path, payload)


@contextmanager
def _cache_build_status(status_path: Path, build_id: Optional[str]):
    try:
        yield
    except BaseException as exc:
        if build_id is not None:
            try:
                _write_build_status(
                    status_path,
                    build_id=build_id,
                    state="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            except OSError:
                pass
        raise


def _wait_for_distributed_cache(
    status_path: Path,
    lock_path: Path,
    *,
    build_id: str,
    timeout_seconds: float,
    stale_after_seconds: float,
) -> None:
    r"""Wait for rank zero without holding a long-running distributed collective."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            status = _load_manifest(status_path)
        except (OSError, json.JSONDecodeError):
            status = None

        if status is not None and status.get("build_id") == build_id:
            state = status.get("state")
            if state == "complete":
                return
            if state == "error":
                raise RuntimeError(
                    "Rank zero failed while building the FloodForecaster cache: "
                    f"{status.get('error', 'unknown error')}"
                )
            if state == "building" and status.get("lock_token"):
                owner, _ = _read_lock_owner(lock_path)
                if (
                    owner is None or owner.get("token") == status.get("lock_token")
                ) and _lock_is_stale(
                    lock_path,
                    stale_after_seconds=stale_after_seconds,
                ):
                    raise RuntimeError(
                        "Rank zero stopped updating the FloodForecaster cache lock "
                        f"at {lock_path}; the cache build was abandoned."
                    )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.1f}s waiting for rank zero to "
                f"finish FloodForecaster cache build {build_id}."
            )
        time.sleep(min(0.5, max(0.01, deadline - time.monotonic())))


def _resolve_dynamic_keys(dynamic_patterns: Dict[str, str]) -> List[str]:
    default_order = ["WD", "VX", "VY"]
    keys = [key for key in default_order if key in dynamic_patterns]
    return keys if keys else list(dynamic_patterns.keys())


def _resolve_boundary_keys(boundary_patterns: Dict[str, str]) -> List[str]:
    keys = sorted(boundary_patterns.keys())
    return keys


def _cache_manifest_is_valid(
    manifest: Optional[Dict[str, Any]],
    h5_path: Path,
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
        manifest is not None
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
    )


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
    wait_timeout_seconds: float = CACHE_LOCK_TIMEOUT_SECONDS,
    stale_lock_seconds: float = CACHE_LOCK_STALE_SECONDS,
) -> Dict[str, Any]:
    r"""Build or reuse the shared HDF5 cache for a FloodForecaster data root.

    In distributed jobs, rank zero exclusively owns cache construction. Other
    ranks monitor a filesystem heartbeat and enter a short process-group barrier
    only after construction completes, avoiding collective timeouts during long
    first-time builds.
    """

    if h5py is None:
        raise ImportError("h5py is required to build the FloodForecaster cache backend.")
    if wait_timeout_seconds <= 0:
        raise ValueError("wait_timeout_seconds must be positive.")
    if stale_lock_seconds <= 0:
        raise ValueError("stale_lock_seconds must be positive.")

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
    status_path = cache_dir / BUILD_STATUS_FILE_NAME
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
    cache_valid = _cache_manifest_is_valid(
        manifest,
        h5_path,
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
    distributed_context = _distributed_cache_context()
    build_id: Optional[str] = None
    if distributed_context is not None:
        rank, _ = distributed_context
        plan = None
        if rank == 0:
            needs_build = bool(rebuild or not cache_valid)
            build_id = uuid.uuid4().hex if needs_build else None
            plan = {"build": needs_build, "build_id": build_id}
            if build_id is not None:
                _write_build_status(
                    status_path,
                    build_id=build_id,
                    state="pending",
                )
        plan = _broadcast_cache_plan(plan)
        if not bool(plan.get("build")):
            if not cache_valid or manifest is None:
                manifest = _load_manifest(manifest_path)
                cache_valid = _cache_manifest_is_valid(
                    manifest,
                    h5_path,
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
            if not cache_valid or manifest is None:
                raise RuntimeError(
                    "Rank zero selected FloodForecaster cache reuse, but this rank "
                    "cannot validate the shared cache files."
                )
            _cache_status_message(f"Reusing cache at {h5_path}")
            return manifest

        planned_build_id = plan.get("build_id")
        if not isinstance(planned_build_id, str) or not planned_build_id:
            raise RuntimeError(
                "Rank zero requested a FloodForecaster cache build without a valid build ID."
            )
        build_id = planned_build_id
        if rank != 0:
            _wait_for_distributed_cache(
                status_path,
                lock_path,
                build_id=build_id,
                timeout_seconds=wait_timeout_seconds,
                stale_after_seconds=stale_lock_seconds,
            )
            _distributed_cache_barrier()
            manifest = _load_manifest(manifest_path)
            if not _cache_manifest_is_valid(
                manifest,
                h5_path,
                data_root=data_root,
                list_file_name=list_file_name,
                run_ids=run_ids,
                static_files=static_files,
                dynamic_keys=dynamic_keys,
                boundary_keys=boundary_keys,
                dynamic_patterns=dynamic_patterns,
                boundary_patterns=boundary_patterns,
                file_records=file_records,
            ):
                raise RuntimeError(
                    "Rank zero reported a complete FloodForecaster cache build, but "
                    "the resulting cache failed validation on this rank."
                )
            _cache_status_message(f"Reusing cache at {h5_path}")
            return manifest
    elif not rebuild and cache_valid and manifest is not None:
        _cache_status_message(f"Reusing cache at {h5_path}")
        return manifest

    with (
        _cache_build_status(status_path, build_id),
        _cache_lock(
            lock_path,
            timeout_seconds=wait_timeout_seconds,
            stale_after_seconds=stale_lock_seconds,
            heartbeat_interval_seconds=min(
                CACHE_LOCK_HEARTBEAT_SECONDS,
                stale_lock_seconds / 3.0,
            ),
        ) as lock_token,
    ):
        if build_id is not None:
            _write_build_status(
                status_path,
                build_id=build_id,
                state="building",
                lock_token=lock_token,
            )
        manifest = _load_manifest(manifest_path)
        if not rebuild and _cache_manifest_is_valid(
            manifest,
            h5_path,
            data_root=data_root,
            list_file_name=list_file_name,
            run_ids=run_ids,
            static_files=static_files,
            dynamic_keys=dynamic_keys,
            boundary_keys=boundary_keys,
            dynamic_patterns=dynamic_patterns,
            boundary_patterns=boundary_patterns,
            file_records=file_records,
        ):
            if build_id is not None:
                _write_build_status(
                    status_path,
                    build_id=build_id,
                    state="complete",
                    lock_token=lock_token,
                )
                _distributed_cache_barrier()
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
        if build_id is not None:
            _write_build_status(
                status_path,
                build_id=build_id,
                state="complete",
                lock_token=lock_token,
            )
            _distributed_cache_barrier()
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
    cache_wait_timeout_seconds: float = CACHE_LOCK_TIMEOUT_SECONDS,
    stale_lock_seconds: float = CACHE_LOCK_STALE_SECONDS,
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
            wait_timeout_seconds=cache_wait_timeout_seconds,
            stale_lock_seconds=stale_lock_seconds,
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
