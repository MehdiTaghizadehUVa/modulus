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
HEC-RAS automation utilities for flood simulation data generation.

This module provides functions for automating HEC-RAS simulations, including
modifying input files, running computations, and extracting results from
HDF5 output files.

Note: This module requires HEC-RAS to be installed and the win32com library
for Windows COM automation.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

try:
    import win32com.client

    HAS_WIN32COM = True
except ImportError:
    HAS_WIN32COM = False
    win32com = None


def read_hydrographs(file_path: Union[str, Path]) -> pd.DataFrame:
    r"""
    Read hydrograph data from a text file.

    Parameters
    ----------
    file_path : str or Path
        Path to hydrograph file. File should be whitespace-separated.

    Returns
    -------
    pd.DataFrame
        DataFrame containing hydrograph data. Each column represents
        a hydrograph time series.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Hydrograph file not found: {file_path}")

    df = pd.read_csv(file_path, sep=r"\s+")
    return df


def format_hydrograph_values(values: Union[List[float], np.ndarray]) -> List[str]:
    r"""
    Format hydrograph values for HEC-RAS U01 file format.

    Formats values to 8-character strings with proper precision for
    HEC-RAS input file compatibility.

    Parameters
    ----------
    values : List[float] or np.ndarray
        Hydrograph values to format.

    Returns
    -------
    List[str]
        List of formatted 8-character strings, right-aligned.

    Examples
    --------
    >>> values = [1234.567, 89.0123, 0.0001]
    >>> formatted = format_hydrograph_values(values)
    >>> len(formatted[0]) == 8
    True
    """
    formatted_values = []
    for value in values:
        str_value = f"{float(value):.6g}"
        if len(str_value) > 8:
            str_value = str_value[:8]
        formatted_values.append(f"{str_value:>8}")
    return formatted_values


def modify_u01_file(
    u01_file_path: Union[str, Path], new_hydrograph_data: List[str]
) -> None:
    r"""
    Modify HEC-RAS U01 file with new hydrograph data.

    Replaces the flow hydrograph section in a HEC-RAS U01 file with
    new hydrograph data.

    Parameters
    ----------
    u01_file_path : str or Path
        Path to the HEC-RAS U01 file to modify.
    new_hydrograph_data : List[str]
        List of formatted hydrograph data strings to insert. Each string
        should contain up to 10 values (80 characters total).

    Raises
    ------
    FileNotFoundError
        If the U01 file does not exist.
    ValueError
        If the 'Flow Hydrograph=' line is not found in the file.
    IOError
        If the file cannot be read or written.
    """
    u01_file_path = Path(u01_file_path)
    if not u01_file_path.exists():
        raise FileNotFoundError(f"U01 file not found: {u01_file_path}")

    with open(u01_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    hydrograph_start_idx = None
    for i, line in enumerate(lines):
        if "Flow Hydrograph=" in line:
            hydrograph_start_idx = i + 1
            break

    if hydrograph_start_idx is None:
        raise ValueError(
            f"'Flow Hydrograph=' line not found in U01 file: {u01_file_path}"
        )

    hydrograph_end_idx = hydrograph_start_idx + len(new_hydrograph_data)
    lines[hydrograph_start_idx:hydrograph_end_idx] = new_hydrograph_data

    with open(u01_file_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


def run_hec_ras_simulation(
    project_path: Union[str, Path],
    project_file: str = "Flood_GNN.prj",
    timeout: Optional[float] = None,
    check_interval: float = 5.0,
) -> Tuple[float, bool]:
    r"""
    Run HEC-RAS simulation using COM automation.

    Opens a HEC-RAS project, runs the current plan computation, and waits
    for completion.

    Parameters
    ----------
    project_path : str or Path
        Path to the HEC-RAS project directory.
    project_file : str, optional, default="Flood_GNN.prj"
        Name of the HEC-RAS project file.
    timeout : float, optional
        Maximum time to wait for computation in seconds. If None, waits
        indefinitely.
    check_interval : float, optional, default=5.0
        Interval in seconds between computation status checks.

    Returns
    -------
    Tuple[float, bool]
        Tuple of (computation_time, success) where computation_time is in
        seconds and success indicates if computation completed successfully.

    Raises
    ------
    ImportError
        If win32com is not available (Windows-only).
    FileNotFoundError
        If the project file does not exist.
    RuntimeError
        If the computation fails or times out.

    Note
    ----
    This function requires HEC-RAS to be installed and the win32com library
    for Windows COM automation. It will only work on Windows systems.
    """
    if not HAS_WIN32COM:
        raise ImportError(
            "win32com is required for HEC-RAS automation. "
            "This functionality is only available on Windows."
        )

    project_path = Path(project_path)
    project_file_path = project_path / project_file

    if not project_file_path.exists():
        raise FileNotFoundError(f"Project file not found: {project_file_path}")

    hec_ras = win32com.client.Dispatch("RAS65.HECRASController")
    hec_ras.Project_Open(str(project_file_path))

    start_time = time.time()
    hec_ras.Compute_CurrentPlan(None, None, False)

    elapsed_time = 0.0
    while True:
        if hec_ras.Compute_Complete() == 1:
            computation_time = time.time() - start_time
            hec_ras.Project_Close()
            hec_ras.QuitRAS()
            return computation_time, True

        time.sleep(check_interval)
        elapsed_time += check_interval

        if timeout is not None and elapsed_time > timeout:
            hec_ras.Project_Close()
            hec_ras.QuitRAS()
            raise RuntimeError(
                f"HEC-RAS computation timed out after {timeout:.1f} seconds"
            )


def extract_and_save_data(
    hdf5_file: Union[str, Path],
    identifier: str,
    save_dir: Union[str, Path],
    prefix: str = "M40",
    save_geometry: bool = True,
    float_format: str = "%.9f",
) -> None:
    r"""
    Extract simulation results from HEC-RAS HDF5 file and save to text files.

    Extracts water depth, velocity components, boundary conditions, and
    geometry data from HEC-RAS HDF5 output files and saves them in a
    format compatible with FloodForecaster datasets.

    Parameters
    ----------
    hdf5_file : str or Path
        Path to the HEC-RAS HDF5 output file (typically .p01.hdf).
    identifier : str
        Identifier string for this simulation (e.g., column name or run ID).
    save_dir : str or Path
        Directory to save extracted data files.
    prefix : str, optional, default="M40"
        Prefix for output filenames.
    save_geometry : bool, optional, default=True
        Whether to save geometry files (XY coordinates, elevations, cell areas).
        Only saved on first run or if explicitly requested.
    float_format : str, optional, default="%.9f"
        Format string for floating-point values in output files.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file does not exist.
    KeyError
        If required datasets are not found in the HDF5 file.
    IOError
        If files cannot be written.

    Note
    ----
    The function extracts the following data:
    - Water depth time series (WD)
    - Velocity X component time series (VX)
    - Velocity Y component time series (VY)
    - Upstream boundary condition (inflow) time series
    - Geometry data (XY coordinates, elevations, cell areas) if save_geometry=True
    """
    hdf5_file = Path(hdf5_file)
    if not hdf5_file.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_file, "r") as f:
        # Extract geometry data (only needed once)
        if save_geometry:
            x_ycoords = f["Geometry/2D Flow Areas/Cell Points"][:]
            elevations = f["Geometry/2D Flow Areas/Perimeter 1/Cells Minimum Elevation"][:]
            area = f["Geometry/2D Flow Areas/Perimeter 1/Cells Surface Area"][:]

            np.savetxt(
                save_dir / f"{prefix}_XY.txt", x_ycoords, delimiter="\t", fmt=float_format
            )
            np.savetxt(
                save_dir / f"{prefix}_CE.txt", elevations, delimiter="\t", fmt=float_format
            )
            np.savetxt(
                save_dir / f"{prefix}_CA.txt", area, delimiter="\t", fmt=float_format
            )

        # Extract time series data
        water_depth_timeseries = f[
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Invert Depth"
        ][:]
        velocity_x_timeseries = f[
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity X"
        ][:]
        velocity_y_timeseries = f[
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Cell Velocity - Velocity Y"
        ][:]
        inflow_timeseries = f[
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Perimeter 1/Boundary Conditions/US_BC"
        ][:]

    # Save time series data
    np.savetxt(
        save_dir / f"{prefix}_WD_{identifier}.txt",
        water_depth_timeseries,
        delimiter="\t",
        fmt=float_format,
    )
    np.savetxt(
        save_dir / f"{prefix}_VX_{identifier}.txt",
        velocity_x_timeseries,
        delimiter="\t",
        fmt=float_format,
    )
    np.savetxt(
        save_dir / f"{prefix}_VY_{identifier}.txt",
        velocity_y_timeseries,
        delimiter="\t",
        fmt=float_format,
    )
    np.savetxt(
        save_dir / f"{prefix}_US_InF_{identifier}.txt",
        inflow_timeseries,
        delimiter="\t",
        fmt=float_format,
    )


def run_simulation_batch(
    project_path: Union[str, Path],
    hydrograph_file: Union[str, Path],
    save_dir: Union[str, Path],
    prefix: str = "M40",
    start_col: int = 0,
    end_col: Optional[int] = None,
    project_file: str = "Flood_GNN.prj",
    u01_file: str = "Flood_GNN.u01",
    timeout: Optional[float] = None,
    save_computation_times: bool = True,
) -> None:
    r"""
    Run a batch of HEC-RAS simulations with multiple hydrographs.

    Processes multiple hydrographs from a file, modifies the U01 file for
    each, runs simulations, and extracts results.

    Parameters
    ----------
    project_path : str or Path
        Path to the HEC-RAS project directory.
    hydrograph_file : str or Path
        Path to file containing hydrograph data. Each column represents
        a hydrograph time series.
    save_dir : str or Path
        Directory to save extracted simulation results.
    prefix : str, optional, default="M40"
        Prefix for output filenames.
    start_col : int, optional, default=0
        Starting column index (zero-based) for hydrographs to process.
    end_col : int, optional
        Ending column index (inclusive) for hydrographs to process.
        If None, processes all columns from start_col to end.
    project_file : str, optional, default="Flood_GNN.prj"
        Name of the HEC-RAS project file.
    u01_file : str, optional, default="Flood_GNN.u01"
        Name of the HEC-RAS U01 file to modify.
    timeout : float, optional
        Maximum time to wait for each computation in seconds.
    save_computation_times : bool, optional, default=True
        Whether to save computation times to a file.

    Raises
    ------
    FileNotFoundError
        If required files are not found.
    ValueError
        If column indices are invalid.

    Examples
    --------
    >>> run_simulation_batch(
    ...     project_path="path/to/project",
    ...     hydrograph_file="hydrographs.txt",
    ...     save_dir="output",
    ...     start_col=0,
    ...     end_col=5
    ... )
    """
    project_path = Path(project_path)
    hydrograph_file = Path(hydrograph_file)
    save_dir = Path(save_dir)

    u01_file_path = project_path / u01_file
    base_hdf5_path = project_path / f"{project_file[:-4]}.p01.hdf"

    if not u01_file_path.exists():
        raise FileNotFoundError(f"U01 file not found: {u01_file_path}")

    if not hydrograph_file.exists():
        raise FileNotFoundError(f"Hydrograph file not found: {hydrograph_file}")

    save_dir.mkdir(parents=True, exist_ok=True)

    hydrographs_df = read_hydrographs(hydrograph_file)

    if start_col < 0 or start_col >= hydrographs_df.shape[1]:
        raise ValueError(
            f"start_col ({start_col}) is out of range [0, {hydrographs_df.shape[1]})"
        )

    if end_col is None:
        end_col = hydrographs_df.shape[1] - 1
    elif end_col < start_col or end_col >= hydrographs_df.shape[1]:
        raise ValueError(
            f"end_col ({end_col}) is invalid. Must be in range [{start_col}, {hydrographs_df.shape[1]})"
        )

    selected_columns = hydrographs_df.iloc[:, start_col : end_col + 1]
    first_run = True

    computation_times_file = save_dir / f"computation_times_{prefix}.txt" if save_computation_times else None

    for column in selected_columns:
        hydrograph = selected_columns[column].astype(str).tolist()
        formatted_hydrograph = format_hydrograph_values(hydrograph)
        formatted_hydrograph = [
            "".join(formatted_hydrograph[i : i + 10]) + "\n"
            for i in range(0, len(formatted_hydrograph), 10)
        ]

        modify_u01_file(u01_file_path, formatted_hydrograph)

        try:
            computation_time, success = run_hec_ras_simulation(
                project_path, project_file=project_file, timeout=timeout
            )

            if save_computation_times and computation_times_file:
                with open(computation_times_file, "a", encoding="utf-8") as times_file:
                    times_file.write(f"Hydrograph {column}: {computation_time:.2f} seconds\n")

            if success:
                extract_and_save_data(
                    base_hdf5_path,
                    str(column),
                    save_dir,
                    prefix=prefix,
                    save_geometry=first_run,
                )
                first_run = False
                print(f"Completed simulation and data extraction for hydrograph {column}")
            else:
                print(f"Warning: Simulation for hydrograph {column} did not complete successfully")

        except Exception as e:
            print(f"Error processing hydrograph {column}: {e}")
            continue

    print("All simulations and data extractions complete.")

