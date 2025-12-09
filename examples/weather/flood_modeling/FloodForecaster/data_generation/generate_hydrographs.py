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
Example script for generating synthetic hydrographs.

This script demonstrates how to generate synthetic hydrographs from base
hydrograph templates for use in flood simulation data generation.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generation.hydrograph_generation import (
    generate_synthetic_hydrographs,
    save_hydrographs,
)


def main():
    r"""Main function for hydrograph generation script."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic hydrographs from base templates"
    )
    parser.add_argument(
        "--base_hydrographs",
        type=str,
        required=True,
        help="Path to base hydrograph file (tab-separated)",
    )
    parser.add_argument(
        "--num_synthetic",
        type=int,
        required=True,
        help="Number of synthetic hydrographs to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for synthetic hydrographs",
    )
    parser.add_argument(
        "--scale_range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        metavar=("MIN", "MAX"),
        help="Scale range for random scaling factors (default: 0.8 1.2)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()

    # Generate synthetic hydrographs
    synthetic_hydrographs = generate_synthetic_hydrographs(
        base_hydrographs=args.base_hydrographs,
        num_synthetic=args.num_synthetic,
        scale_range=tuple(args.scale_range),
        random_seed=args.random_seed,
    )

    # Save to file
    save_hydrographs(synthetic_hydrographs, args.output)

    print(f"Generated {args.num_synthetic} synthetic hydrographs")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

