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

r"""
Synthetic hydrograph generation utilities.

This module provides functions for generating synthetic hydrographs from
base hydrograph templates using random scaling factors.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


def generate_synthetic_hydrographs(
    base_hydrographs: Union[pd.DataFrame, str, Path],
    num_synthetic: int,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    r"""
    Generate synthetic hydrographs from base hydrograph templates.

    This function creates synthetic hydrographs by randomly selecting base
    hydrographs and scaling them by random factors within a specified range.

    Parameters
    ----------
    base_hydrographs : pd.DataFrame or str or Path
        Base hydrograph data. Can be a DataFrame or path to a file containing
        base hydrographs. Each column represents a base hydrograph time series.
    num_synthetic : int
        Number of synthetic hydrographs to generate.
    scale_range : Tuple[float, float], optional, default=(0.8, 1.2)
        Tuple of (min_scale, max_scale) for random scaling factors.
    random_seed : int, optional
        Random seed for reproducibility. If None, uses current random state.

    Returns
    -------
    pd.DataFrame
        DataFrame containing synthetic hydrographs. Each column represents
        a synthetic hydrograph time series with column names 'T1', 'T2', etc.

    Raises
    ------
    ValueError
        If ``num_synthetic`` is not positive or if ``scale_range`` is invalid.
    FileNotFoundError
        If ``base_hydrographs`` is a path and the file does not exist.

    Examples
    --------
    >>> import pandas as pd
    >>> base_df = pd.read_csv("base_hydrographs.txt", delimiter='\t')
    >>> synthetic = generate_synthetic_hydrographs(base_df, num_synthetic=10)
    >>> synthetic.shape
    (n_timesteps, 10)
    """
    if num_synthetic <= 0:
        raise ValueError(f"num_synthetic must be positive, got {num_synthetic}")

    if scale_range[0] >= scale_range[1] or scale_range[0] <= 0:
        raise ValueError(
            f"scale_range must be (min, max) with 0 < min < max, got {scale_range}"
        )

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Load base hydrographs if path provided
    if isinstance(base_hydrographs, (str, Path)):
        base_hydrographs = pd.read_csv(base_hydrographs, delimiter="\t")

    if not isinstance(base_hydrographs, pd.DataFrame):
        raise TypeError(
            f"base_hydrographs must be DataFrame, str, or Path, got {type(base_hydrographs)}"
        )

    num_base = base_hydrographs.shape[1]
    if num_base == 0:
        raise ValueError("base_hydrographs must contain at least one column")

    synthetic_hydrographs = []

    for _ in range(num_synthetic):
        # Select a random base hydrograph
        random_base = base_hydrographs.iloc[:, np.random.randint(0, num_base)]
        # Multiply its values by a random number within scale_range
        random_factor = np.random.uniform(scale_range[0], scale_range[1])
        synthetic_hydrograph = random_base * random_factor
        synthetic_hydrographs.append(synthetic_hydrograph.values)

    # Convert to DataFrame
    synthetic_hydrographs_df = pd.DataFrame(synthetic_hydrographs).transpose()

    headers = [f"T{i}" for i in range(1, 1 + num_synthetic)]
    synthetic_hydrographs_df.columns = headers

    return synthetic_hydrographs_df


def save_hydrographs(
    hydrographs: pd.DataFrame,
    output_path: Union[str, Path],
    delimiter: str = "\t",
) -> None:
    r"""
    Save hydrographs to a text file.

    Parameters
    ----------
    hydrographs : pd.DataFrame
        DataFrame containing hydrographs to save. Each column represents
        a hydrograph time series.
    output_path : str or Path
        Path to output file.
    delimiter : str, optional, default='\t'
        Delimiter to use in output file.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hydrographs.to_csv(output_path, sep=delimiter, index=False, header=True)

