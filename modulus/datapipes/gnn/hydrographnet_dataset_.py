import os
import json
import random
import pickle
from tqdm import tqdm

import numpy as np
from scipy.spatial import KDTree

import torch
import dgl
from dgl.data import DGLDataset


class HydroGraphDataset(DGLDataset):
    """
    DGL Dataset for hydrograph-based graphs with one-step target difference.

    This dataset class loads constant and dynamic data from a given folder (raw text files)
    and creates graphs with node and edge features. The constant files include information such as
    coordinates, area, elevation, slope, aspect, curvature, manning, flow accumulation and infiltration.
    Dynamic files include water depth, inflow hydrograph, volume and precipitation.

    For each hydrograph (specified by its id), the dynamic data is loaded once and an index is built
    to extract sliding time windows on the fly in __getitem__. At each time step, node features are computed
    from a window of dynamic data (of length n_time_steps) and the target is set as the difference between
    the state at time t+n_time_steps and t+n_time_steps-1 for both water depth and volume.

    Additionally, hydrograph IDs can be loaded from a text file. The file path is constructed relative
    to data_dir using the `hydrograph_ids_file` parameter. Each line in the file should contain a single
    hydrograph ID.

    Parameters
    ----------
    name : str, optional
        Name of the dataset. Default is "hydrograph_dataset".
    data_dir : str
        Directory containing the raw data files.
    prefix : str
        Prefix used for file names (e.g. "M80" or "M10").
    num_samples : int, optional
        Number of hydrograph IDs to sample from the folder (or file). Default is 500.
    n_time_steps : int
        Number of time steps to include in the node features window.
    k : int, optional
        Number of nearest neighbors to use when building graph connectivity. Default is 4.
    add_noise : bool, optional
        Whether to add random walk noise to the dynamic features. Default is False.
    noise_std : float, optional
        Standard deviation of the noise added (if add_noise is True). Default is 0.01.
    hydrograph_ids_file : str, optional
        Relative path (to data_dir) to a text file containing hydrograph IDs (one per line). Default is None.
    force_reload : bool, optional
        Whether to force reloading of the dataset. Default is False.
    verbose : bool, optional
        Verbosity flag. Default is False.
    """

    def __init__(
        self,
        name="hydrograph_dataset",
        data_dir=None,
        prefix="M80",
        num_samples=500,
        n_time_steps=10,
        k=4,
        add_noise=False,
        noise_std=0.01,
        hydrograph_ids_file=None,
        force_reload=False,
        verbose=False,
    ):
        self.data_dir = data_dir
        self.prefix = prefix
        self.num_samples = num_samples
        self.n_time_steps = n_time_steps
        self.k = k
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.hydrograph_ids_file = hydrograph_ids_file

        # Will hold the static (constant) data and graph structure.
        self.static_data = {}
        # Will hold dynamic data for each hydrograph (list of dicts).
        self.dynamic_data = []
        # Global sample index: list of tuples (hydrograph_index, time_index)
        self.sample_index = []
        # List of hydrograph IDs.
        self.hydrograph_ids = []

        super().__init__(name=name, force_reload=force_reload, verbose=verbose)

    def process(self):
        # ----------------------------
        # 1. Load static data and build graph structure.
        # ----------------------------
        (xy_coords, area, area_denorm, elevation, slope, aspect, curvature,
         manning, flow_accum, infiltration) = self.load_constant_data(self.data_dir, self.prefix)
        num_nodes = xy_coords.shape[0]

        # Build KDTree and compute edge indices based on k-nearest neighbors.
        kdtree = KDTree(xy_coords)
        _, neighbors = kdtree.query(xy_coords, k=self.k + 1)  # first neighbor is self
        edge_index = np.vstack([
            (i, nbr) for i, nbrs in enumerate(neighbors) for nbr in nbrs if nbr != i
        ]).T  # shape: (2, num_edges)

        # Compute edge features from the static node positions.
        edge_features = self.create_edge_features(xy_coords, edge_index)

        # Save all static arrays and graph structure.
        self.static_data = {
            "xy_coords": xy_coords,
            "area": area,
            "area_denorm": area_denorm,
            "elevation": elevation,
            "slope": slope,
            "aspect": aspect,
            "curvature": curvature,
            "manning": manning,
            "flow_accum": flow_accum,
            "infiltration": infiltration,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

        # ----------------------------
        # 2. Load hydrograph IDs.
        # ----------------------------
        if self.hydrograph_ids_file is not None:
            # Construct the file path relative to data_dir.
            file_path = os.path.join(self.data_dir, self.hydrograph_ids_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    lines = f.readlines()
                self.hydrograph_ids = [line.strip() for line in lines if line.strip()]
            else:
                raise FileNotFoundError(f"Hydrograph IDs file not found: {file_path}")
        else:
            # Scan the folder for hydrograph IDs.
            all_files = os.listdir(self.data_dir)
            self.hydrograph_ids = []
            for f in all_files:
                if f.startswith(f"{self.prefix}_WD_") and f.endswith(".txt"):
                    parts = f.split('_')
                    if len(parts) >= 3:
                        hid = os.path.splitext(parts[2])[0]
                        self.hydrograph_ids.append(hid)

        # Sample if there are more IDs than requested.
        if len(self.hydrograph_ids) > self.num_samples:
            self.hydrograph_ids = random.sample(self.hydrograph_ids, self.num_samples)

        # ----------------------------
        # 3. Load dynamic data for each hydrograph and build sample index.
        # ----------------------------
        for h_idx, hid in enumerate(tqdm(self.hydrograph_ids, desc="Processing Hydrographs")):
            water_depth, inflow_hydrograph, volume, precipitation = self.load_dynamic_data(
                self.data_dir, hid, self.prefix, num_points=num_nodes
            )
            T = water_depth.shape[0]
            self.dynamic_data.append({
                "water_depth": water_depth,
                "inflow_hydrograph": inflow_hydrograph,
                "volume": volume,
                "precipitation": precipitation,
                "hydro_id": hid,
            })
            # Valid time indices: need at least (n_time_steps + 1) steps for a sample.
            for t in range(T - self.n_time_steps):
                self.sample_index.append((h_idx, t))

        self.length = len(self.sample_index)

    def __getitem__(self, idx):
        # Map global index to hydrograph index and time step.
        hydro_idx, t_idx = self.sample_index[idx]
        dyn = self.dynamic_data[hydro_idx]
        sd = self.static_data

        # Compute node features and obtain the inflow value at target time.
        node_features, future_inflow = self.create_node_features(
            sd["xy_coords"],
            sd["area"],
            sd["elevation"],
            sd["slope"],
            sd["aspect"],
            sd["curvature"],
            sd["manning"],
            sd["flow_accum"],
            sd["infiltration"],
            dyn["water_depth"],
            dyn["volume"],
            dyn["precipitation"],
            t_idx,
            self.n_time_steps,
            dyn["inflow_hydrograph"]
        )
        # Compute target as the difference between time t_idx+n_time_steps and t_idx+n_time_steps-1.
        target_time = t_idx + self.n_time_steps
        prev_time = target_time - 1
        target_depth = dyn["water_depth"][target_time, :] - dyn["water_depth"][prev_time, :]
        target_volume = dyn["volume"][target_time, :] - dyn["volume"][prev_time, :]
        target = np.stack([target_depth, target_volume], axis=1)  # shape: (num_nodes, 2)

        # Build new graph using the stored static edge index.
        src, dst = sd["edge_index"]
        g = dgl.graph((src, dst))
        # Assign static edge features.
        g.edata["x"] = torch.tensor(sd["edge_features"], dtype=torch.float)
        # Assign computed node features and target difference.
        g.ndata["x"] = torch.tensor(node_features, dtype=torch.float)
        g.ndata["y"] = torch.tensor(target, dtype=torch.float)
        # Replicate the scalar future_inflow to all nodes.
        num_nodes = g.num_nodes()
        g.ndata["flow_hydrograph"] = torch.full((num_nodes, 1), future_inflow, dtype=torch.float)
        # Store area_denorm directly in ndata (it already has shape (num_nodes, 1)).
        g.ndata["area_denorm"] = torch.tensor(sd["area_denorm"], dtype=torch.float)

        return g

    def __len__(self):
        return self.length

    # ----------------------------
    # Helper functions (static methods)
    # ----------------------------
    @staticmethod
    def min_max_normalize(data, min_val, max_val):
        return (data - min_val) / (max_val - 1e-8)

    @staticmethod
    def load_constant_data(folder, prefix):
        # File paths.
        xy_path = os.path.join(folder, f"{prefix}_XY.txt")
        ca_path = os.path.join(folder, f"{prefix}_CA.txt")
        ce_path = os.path.join(folder, f"{prefix}_CE.txt")
        cs_path = os.path.join(folder, f"{prefix}_CS.txt")
        aspect_path = os.path.join(folder, f"{prefix}_A.txt")
        curvature_path = os.path.join(folder, f"{prefix}_CU.txt")
        manning_path = os.path.join(folder, f"{prefix}_N.txt")
        flow_accum_path = os.path.join(folder, f"{prefix}_FA.txt")
        infiltration_path = os.path.join(folder, f"{prefix}_IP.txt")  # Added infiltration

        # Load static data from text files (assumed tab-delimited).
        xy_coords = np.loadtxt(xy_path, delimiter='\t')
        area_denorm = np.loadtxt(ca_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        area = area_denorm.copy()
        elevation = np.loadtxt(ce_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        slope = np.loadtxt(cs_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        aspect = np.loadtxt(aspect_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        curvature = np.loadtxt(curvature_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        manning = np.loadtxt(manning_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        flow_accum = np.loadtxt(flow_accum_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)
        infiltration = np.loadtxt(infiltration_path, delimiter='\t')[:xy_coords.shape[0]].reshape(-1, 1)

        # Normalize the spatial coordinates and static parameters.
        xy_min, xy_max = xy_coords.min(axis=0), xy_coords.max(axis=0)
        xy_coords = HydroGraphDataset.min_max_normalize(xy_coords, xy_min, xy_max)
        area = HydroGraphDataset.min_max_normalize(area, area.min(), area.max())
        elevation = HydroGraphDataset.min_max_normalize(elevation, elevation.min(), elevation.max())
        slope = HydroGraphDataset.min_max_normalize(slope, slope.min(), slope.max())
        aspect = HydroGraphDataset.min_max_normalize(aspect, aspect.min(), aspect.max())
        curvature = HydroGraphDataset.min_max_normalize(curvature, curvature.min(), curvature.max())
        manning = HydroGraphDataset.min_max_normalize(manning, manning.min(), manning.max())
        flow_accum = HydroGraphDataset.min_max_normalize(flow_accum, flow_accum.min(), flow_accum.max())
        infiltration = HydroGraphDataset.min_max_normalize(infiltration, 0, 100)

        return xy_coords, area, area_denorm, elevation, slope, aspect, curvature, manning, flow_accum, infiltration

    @staticmethod
    def load_dynamic_data(folder, hydrograph_id, prefix, num_points, interval=1, skip=72):
        # File paths for dynamic data.
        wd_path = os.path.join(folder, f"{prefix}_WD_{hydrograph_id}.txt")
        inflow_path = os.path.join(folder, f"{prefix}_US_InF_{hydrograph_id}.txt")
        volume_path = os.path.join(folder, f"{prefix}_V_{hydrograph_id}.txt")
        precipitation_path = os.path.join(folder, f"{prefix}_Pr_{hydrograph_id}.txt")

        # Load dynamic data (assumed tab-delimited).
        water_depth = np.loadtxt(wd_path, delimiter='\t')[skip::interval, :num_points]
        inflow_hydrograph = np.loadtxt(inflow_path, delimiter='\t')[skip::interval, 1]
        volume = np.loadtxt(volume_path, delimiter='\t')[skip::interval, :num_points]
        precipitation = np.loadtxt(precipitation_path, delimiter='\t')[skip::interval]

        # Limit the time window to peak time + 25 steps.
        peak_time_idx = np.argmax(inflow_hydrograph)
        water_depth = water_depth[:peak_time_idx + 25]
        volume = volume[:peak_time_idx + 25]
        precipitation = precipitation[:peak_time_idx + 25] * 2.7778e-7  # conversion: mm/hr -> m/s
        inflow_hydrograph = inflow_hydrograph[:peak_time_idx + 25]

        return water_depth, inflow_hydrograph, volume, precipitation

    @staticmethod
    def add_random_walk_noise(data, noise_std_last_step, num_steps):
        noise_std_each_step = noise_std_last_step / np.sqrt(num_steps)
        noise = np.zeros_like(data)
        noise[0] = np.random.normal(0, noise_std_each_step, size=data.shape[1])
        for i in range(1, len(data)):
            noise[i] = noise[i - 1] + np.random.normal(0, noise_std_each_step, size=data.shape[1])
        return data + noise

    def create_node_features(
        self,
        xy_coords,
        area,
        elevation,
        slope,
        aspect,
        curvature,
        manning,
        flow_accum,
        infiltration,
        water_depth,
        volume,
        precipitation_data,
        time_step,
        n_time_steps,
        inflow_hydrograph,
    ):
        num_steps = n_time_steps
        # Optionally add noise to dynamic features.
        if self.add_noise:
            water_depth[time_step:time_step + num_steps, :] = self.add_random_walk_noise(
                water_depth[time_step:time_step + num_steps, :],
                self.noise_std,
                num_steps,
            )
            volume[time_step:time_step + num_steps, :] = self.add_random_walk_noise(
                volume[time_step:time_step + num_steps, :],
                self.noise_std,
                num_steps,
            )

        num_nodes = xy_coords.shape[0]
        # For the current time step, assign inflow and precipitation (scaled by infiltration).
        flow_hydrograph_current_step = np.full((num_nodes, 1), inflow_hydrograph[time_step])
        precip_current_step = np.full((num_nodes, 1), precipitation_data[time_step])
        precip_current_step *= infiltration

        # Stack all features horizontally.
        features = np.hstack([
            xy_coords,                                # (num_nodes, 2)
            area,                                     # (num_nodes, 1)
            elevation,                                # (num_nodes, 1)
            slope,                                    # (num_nodes, 1)
            aspect,                                   # (num_nodes, 1)
            curvature,                                # (num_nodes, 1)
            manning,                                  # (num_nodes, 1)
            flow_accum,                               # (num_nodes, 1)
            infiltration,                             # (num_nodes, 1)
            flow_hydrograph_current_step,             # (num_nodes, 1)
            precip_current_step,                       # (num_nodes, 1)
            water_depth[time_step:time_step + n_time_steps, :].T,  # (num_nodes, n_time_steps)
            volume[time_step:time_step + n_time_steps, :].T        # (num_nodes, n_time_steps)
        ])
        # Return the inflow value at the target time.
        future_inflow = inflow_hydrograph[time_step + n_time_steps]
        return features, future_inflow

    @staticmethod
    def create_edge_features(xy_coords, edge_index):
        # edge_index is assumed to be a (2, num_edges) numpy array.
        row, col = edge_index
        relative_coords = xy_coords[row] - xy_coords[col]
        distance = np.linalg.norm(relative_coords, axis=1)
        # Normalize relative coordinates.
        rel_min, rel_max = relative_coords.min(axis=0), relative_coords.max(axis=0)
        relative_coords = HydroGraphDataset.min_max_normalize(relative_coords, rel_min, rel_max)
        # Normalize distances.
        dist_min, dist_max = distance.min(), distance.max()
        distance = HydroGraphDataset.min_max_normalize(distance, dist_min, dist_max)
        # Concatenate normalized relative coordinates and distance.
        edge_features = np.hstack([relative_coords, distance[:, None]])
        return edge_features

    def save_processed_data(self, save_path):
        """Save the processed graphs data to a pickle file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump((self.static_data, self.dynamic_data, self.sample_index), f)

    @staticmethod
    def load_processed_data(load_path):
        """Load the processed graphs data from a pickle file."""
        with open(load_path, 'rb') as f:
            static_data, dynamic_data, sample_index = pickle.load(f)
        return static_data, dynamic_data, sample_index
