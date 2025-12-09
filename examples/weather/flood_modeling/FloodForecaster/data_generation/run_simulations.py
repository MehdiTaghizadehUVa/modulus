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
Example script for running batch HEC-RAS simulations.

This script demonstrates how to run a batch of HEC-RAS simulations with
multiple hydrographs, automatically modifying input files and extracting
results for use in FloodForecaster training.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_generation.hec_ras_automation import run_simulation_batch


def main():
    r"""Main function for HEC-RAS simulation batch script."""
    parser = argparse.ArgumentParser(
        description="Run batch HEC-RAS simulations with multiple hydrographs"
    )
    parser.add_argument(
        "--project_path",
        type=str,
        required=True,
        help="Path to HEC-RAS project directory",
    )
    parser.add_argument(
        "--hydrograph_file",
        type=str,
        required=True,
        help="Path to hydrograph file (tab-separated, each column is a hydrograph)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save extracted simulation results",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="M40",
        help="Prefix for output filenames (default: M40)",
    )
    parser.add_argument(
        "--start_col",
        type=int,
        default=0,
        help="Starting column index (zero-based) for hydrographs to process (default: 0)",
    )
    parser.add_argument(
        "--end_col",
        type=int,
        default=None,
        help="Ending column index (inclusive) for hydrographs to process (default: all)",
    )
    parser.add_argument(
        "--project_file",
        type=str,
        default="Flood_GNN.prj",
        help="Name of HEC-RAS project file (default: Flood_GNN.prj)",
    )
    parser.add_argument(
        "--u01_file",
        type=str,
        default="Flood_GNN.u01",
        help="Name of HEC-RAS U01 file to modify (default: Flood_GNN.u01)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Maximum time to wait for each computation in seconds (default: None, wait indefinitely)",
    )

    args = parser.parse_args()

    # Validate paths
    project_path = Path(args.project_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")

    hydrograph_file = Path(args.hydrograph_file)
    if not hydrograph_file.exists():
        raise FileNotFoundError(f"Hydrograph file does not exist: {hydrograph_file}")

    # Run batch simulations
    run_simulation_batch(
        project_path=project_path,
        hydrograph_file=hydrograph_file,
        save_dir=args.save_dir,
        prefix=args.prefix,
        start_col=args.start_col,
        end_col=args.end_col,
        project_file=args.project_file,
        u01_file=args.u01_file,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()

