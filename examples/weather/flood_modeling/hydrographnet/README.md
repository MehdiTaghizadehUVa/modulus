# HydroGraphNet

HydroGraphNet is a physics-informed graph neural network for autoregressive flood forecasting on unstructured meshes. This example trains a `MeshGraphKAN` model to predict one-step changes in water depth and cell volume, optionally adds the paper's mass-balance penalty, and supports a two-step pushforward stability objective.

## What Changed

This example now uses one explicit data contract across dataset construction, training, and rollout:

- Input windows contain `n_time_steps` observed states ending at anchor time `t`
- Node features are ordered as:
  - normalized static mesh features
  - normalized inflow and precipitation at time `t`
  - normalized water-depth history for `[t - n_time_steps + 1, ..., t]`
  - normalized volume history for `[t - n_time_steps + 1, ..., t]`
- One-step targets are the normalized deltas from `t` to `t + 1`
- Pushforward targets are the normalized deltas from `t + 1` to `t + 2`
- Physics loss uses interval-averaged source terms over `[t, t + 1]` and `[t + 1, t + 2]`
- Inference rollout and RMSE reporting are computed in physical water-depth units

## Dataset Interface

The dataset lives at `physicsnemo/datapipes/gnn/hydrographnet_dataset.py`.

Use it with:

```python
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

train_dataset = HydroGraphDataset(
    data_dir="./data",
    stats_dir="./data",
    prefix="M80",
    split="train",
    n_time_steps=2,
    hydrograph_ids_file="train.txt",
    return_physics=True,
)
```

Behavior by split:

- `split="train"` returns a single `torch_geometric.data.Data` object
- `Data.y` is the one-step target delta
- `Data.y_pushforward` is present when `noise_type="pushforward"`
- graph-level physics tensors are attached directly on the `Data` object when `return_physics=True`
- `split="test"` returns `(graph, rollout_data)` where:
  - `graph` contains the observed window ending at time `t = n_time_steps - 1`
  - `rollout_data["inflow"]` and `rollout_data["precipitation"]` are normalized future forcing values used during autoregressive rollout
  - `rollout_data["water_depth_gt"]` and `rollout_data["volume_gt"]` are denormalized physical targets

`stats_dir` controls where normalization statistics are written and read. This lets inference read the training statistics even when `test_dir` points at a different folder.

## Training

Configure `conf/config.yaml`, then run:

```bash
python train.py --config-path conf --config-name config
```

Important config fields:

- `prefix`
- `train_ids_file`
- `test_ids_file`
- `stats_dir`
- `k_neighbors`
- `noise_type`
- `noise_std`
- `physics_penalty_weight`
- `depth_volume_penalty_weight`
- `pushforward_stability_weight`

The model input width is derived from `n_time_steps`. If `num_input_features` is set explicitly and does not match the derived width, training fails fast.

## Inference

Run rollout with:

```bash
python inference.py --config-path conf --config-name config
```

Inference:

- instantiates the model through the same shared builder as training
- loads normalization statistics from `stats_dir`
- runs rollout under `torch.no_grad()`
- updates the autoregressive window with the same forcing and history semantics used during training
- denormalizes predictions before computing RMSE or generating animations

The generated GIFs show predicted water depth, ground truth water depth, absolute error, and RMSE in physical units.

![HydroGraphNet animation](../../../../docs/img/hydrographnet.gif)

## Citation

If you use HydroGraphNet in your research, cite:

```bibtex
@article{taghizadeh2025hydrographnet,
  title   = {Interpretable Physics-Informed Graph Neural Networks for Flood Forecasting},
  author  = {Taghizadeh, Mehdi and Zandsalimi, Zanko and Nabian, Mohammad Amin and Shafiee-Jood, Majid and Alemazkoor, Negin},
  journal = {Computer-Aided Civil and Infrastructure Engineering},
  year    = {2025},
  doi     = {10.1111/mice.13484}
}
```
