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

r"""Utility modules for flood prediction.

This module provides normalization utilities specific to FloodForecaster.
For logging, use physicsnemo.launch.logging (PythonLogger, RankZeroLoggingWrapper).
For configuration, use Hydra (already integrated in train.py and inference.py).
"""

from .normalization import (
    collect_all_fields,
    fit_normalizers_from_dataset,
    fit_normalizers_from_sample_index,
    stack_and_fit_transform,
    transform_with_existing_normalizers,
)
from .checkpointing import (
    BEST_CHECKPOINT_FILENAME,
    resolve_checkpoint_epoch,
    validate_checkpoint_files,
    write_best_checkpoint_metadata,
)
from .runtime import (
    create_data_loader,
    create_loader_from_config,
    make_torch_generator,
    resolve_amp_autocast_enabled,
    seed_everything,
    set_loader_epoch,
    split_dataset,
)

__all__ = [
    "BEST_CHECKPOINT_FILENAME",
    "collect_all_fields",
    "create_data_loader",
    "create_loader_from_config",
    "fit_normalizers_from_dataset",
    "fit_normalizers_from_sample_index",
    "make_torch_generator",
    "resolve_amp_autocast_enabled",
    "resolve_checkpoint_epoch",
    "seed_everything",
    "set_loader_epoch",
    "split_dataset",
    "stack_and_fit_transform",
    "transform_with_existing_normalizers",
    "validate_checkpoint_files",
    "write_best_checkpoint_metadata",
]

