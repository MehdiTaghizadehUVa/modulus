# FloodForecaster: Domain-Adaptive GINO for Rapid Flood Forecasting

FloodForecaster is a flood-surrogate example built around a Geometry-Informed Neural Operator (GINO) plus adversarial domain adaptation. It is designed for rapid next-step prediction of water depth and velocity on an unstructured floodplain mesh, followed by autoregressive rollout during inference.

The current implementation is tightly aligned to the GINO path in `neuralop`: the model is instantiated through `neuralop.get_model(...)`, wrapped in the example-local `GINOWrapper`, and trained/evaluated through PhysicsNeMo-style processors, trainers, and checkpointing utilities.

## Problem Overview

Traditional physically based flood solvers are accurate, but expensive when high spatial resolution and many scenarios are required. FloodForecaster addresses that gap with a learned surrogate that:

- handles irregular terrain through geometry-aware operator layers
- predicts water depth and velocity on the native cell set
- supports transfer from a source river/domain to a target river/domain through adversarial fine-tuning
- performs fast autoregressive rollout once trained

## Model Overview

FloodForecaster uses a three-stage workflow:

1. Source-domain pretraining on one-step prediction windows
2. Source-plus-target domain adaptation with a gradient reversal layer and CNN domain classifier
3. Single-device rollout inference for multi-step autoregressive state evaluation

### Core GINO Path

The predictive backbone combines:

- a GNO input/output path for mapping between mesh cells and the latent grid
- an FNO latent core for global spatial processing
- a FloodForecaster-specific `GINOWrapper` that makes the model checkpointable and compatible with the processor/trainer pipeline

The runtime feature contract is:

- `geometry`: unit-box-normalized mesh coordinates with shape `(N, 2)` or batched/shared `(B, N, 2)`
- `static`: per-cell static features with shape `(N, C_static)`
- `boundary`: compact boundary history with shape `(H, C_bc)` or batched `(B, H, C_bc)`
- `dynamic`: state history with shape `(H, N, 3)` or batched `(B, H, N, 3)` for `[WD, VX, VY]`
- `target`: next-step state with shape `(N, 3)` or batched `(B, N, 3)`
- `query_points`: latent grid coordinates with shape `(Hx, Hy, 2)` or batched/shared `(B, Hx, Hy, 2)`

The processor broadcasts compact boundary histories across cells only when building the final per-cell feature tensor, so the dataset does not materialize spatially repeated hydrograph values for every sample.

The model input-channel count must match:

```text
data_channels = C_static + H * C_bc + H * 3
```

With the default static files, `C_static=8` because `M40_XY.txt` contributes two static coordinate channels and the six scalar static files contribute one channel each. With `H=3` and one inflow boundary channel, the default `data_channels` is `8 + 3 * 1 + 3 * 3 = 20`.

### Domain Adaptation

Domain adaptation uses:

- supervised regression on both source and target windows
- a gradient reversal layer (GRL)
- a CNN-based domain classifier in `models/domain_classifier.py`

The GRL schedule is scaled by `training.da_lambda_max`, and setting `training.da_class_loss_weight <= 0` disables the adversarial branch and falls back to supervised fine-tuning.

## Data Generation

Utilities for synthetic hydrograph generation and HEC-RAS automation live in the separate FloodForecaster data-generation repository:

- [FloodForecaster data generation repository](https://github.com/MehdiTaghizadehUVa/FloodForecaster)

That repository is for generating data. This example expects pre-generated flood runs in the layout described below.

## Dataset Layout

FloodForecaster expects one directory per domain split, with shared geometry/static files plus per-run time-series files.

### Shared Mesh and Static Files

- `M40_XY.txt`: cell-center coordinates `(N, 2)`. The loader normalizes this geometry once to the unit box `[0, 1]^2` for GINO.
- `M40_CA.txt`: cell area `(N,)` or `(N, 1)`. When present it is kept both as a static feature and as the optional `cell_area` field used by rollout metrics and plots.
- `M40_CE.txt`: elevation `(N,)` or `(N, C)`
- `M40_CS.txt`: slope `(N,)` or `(N, C)`
- `M40_CU.txt`: curvature `(N,)` or `(N, C)`
- `M40_FA.txt`: Manning roughness `(N,)` or `(N, C)`
- `M40_A.txt`: additional static feature(s), typically `(N,)` or `(N, C)`

Static files with more than `N` rows are trimmed to the reference geometry length. Static files with fewer than `N` rows raise by default because `raise_on_smaller=True`.

### Per-Run Dynamic Files

The `{}` placeholders in `dynamic_patterns` are filled with run IDs, not timestep indices.

- `M40_WD_{run_id}.txt`: full water-depth history `(T, N)`
- `M40_VX_{run_id}.txt`: full x-velocity history `(T, N)`
- `M40_VY_{run_id}.txt`: full y-velocity history `(T, N)`

Dynamic files with extra cell columns are trimmed to `N`; files with fewer than `N` columns raise by default.

### Per-Run Boundary Files

- `M40_US_InF_{run_id}.txt`: upstream inflow history `(T,)`, `(T, 1)`, or `(T, 2)` when the first column is time and the second column is inflow

If a boundary file has two columns, the loader treats the second column as the inflow value and drops the first column. Multiple boundary files are concatenated into compact `(T, C_bc)` tensors. If dynamic and boundary sequence lengths differ for a run, both are truncated to the shorter length with a warning.

### Run Lists

- `train.txt`, `val.txt`, `test.txt`: list run IDs used to build datasets

These run IDs are the values substituted into the filename patterns above. The loader accepts either one run ID per line or a single comma-separated line. Files are read with UTF-8 BOM handling, so accidental BOM characters in `train.txt` do not become part of the run ID.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Prepare source, target, and rollout data directories following the layout above.

### Configuration

The main config is [conf/config.yaml](conf/config.yaml). The example also ships smaller presets:

- [config_smoke.yaml](conf/config_smoke.yaml)
- [config_short.yaml](conf/config_short.yaml)
- [config_full_single_gpu.yaml](conf/config_full_single_gpu.yaml)
- [config_full_single_gpu_small.yaml](conf/config_full_single_gpu_small.yaml)

Important knobs:

- `source_data.root`, `target_data.root`, `rollout_data.root`: dataset roots
- `source_data.batch_size`, `target_data.batch_size`: per-domain batch sizes
- `source_data.num_workers`, `pin_memory`, `persistent_workers`: DataLoader settings
- `source_data.noise_type`, `source_data.noise_std`: dynamic-history training augmentation; supported noise types are `none`, `only_last`, `correlated`, `uncorrelated`, and `random_walk`
- `training.n_epochs_source`, `training.n_epochs_adapt`: stage lengths
- `training.eval_interval`: validation cadence for both pretraining and adaptation
- `training.learning_rate`, `training.adapt_learning_rate`: optimizer learning rates
- `training.amp_autocast`: mixed precision toggle
- `training.da_lambda_max`, `training.da_class_loss_weight`, `training.da_classifier`: domain-adaptation settings
- `checkpoint.*`: save, resume, and inference checkpoint selection; `checkpoint.save_every: null` disables interval saves while still writing a final/latest checkpoint
- `data_io.*`: cache backend and per-process run-cache settings

Hydra overrides work as usual, for example:

```bash
python train.py --config-name config_full_single_gpu_small source_data.root=/path/to/source target_data.root=/path/to/target
```

### Training

Run the full training pipeline:

```bash
python train.py --config-name config
```

This performs:

1. source-domain pretraining
2. target-domain adaptation

`train.py` does not run rollout evaluation. Use `inference.py` for rollout and plots.

### Resuming Training

Resume directories are stage directories under `checkpoint.save_dir`:

- `{save_dir}/pretrain`
- `{save_dir}/adapt`

Set:

- `checkpoint.resume_from_source` to resume pretraining
- `checkpoint.resume_from_adapt` to resume adaptation

FloodForecaster now follows the native PhysicsNeMo checkpoint contract in those stage directories:

- model weights are written as `GINOWrapper.*.mdlus`
- during adaptation, classifier weights are written as `CNNDomainClassifier.*.mdlus`
- optimizer/scheduler/scaler state is written as `checkpoint.*.pt`
- when `checkpoint.save_best` is enabled, the stage directory also records `best_checkpoint.json`

The supported restore workflow is:

1. instantiate the neuralop GINO model from config
2. wrap it with `GINOWrapper`
3. call `physicsnemo.utils.checkpoint.load_checkpoint(...)` on the checkpoint directory

Point resume and inference settings at the checkpoint directory, not at an individual `.mdlus` file.

### Inference

Run rollout inference and figure generation with:

```bash
python inference.py --config-name config
```

Inference:

- restores a source or adapted checkpoint directory
- rebuilds/fits the required normalizers
- loads rollout runs lazily
- performs canonical autoregressive rollout in `inference/rollout.py`
- generates figures, animations, and aggregated metrics

The inference path is single-device only. Distributed rollout splitting is not implemented in this example.

Rollout semantics are:

- the initial window is the normalized history slice `dynamic[skip_before_timestep : skip_before_timestep + history_steps]`
- the model input at each step is built from shared static features plus the current boundary-history window plus the current dynamic-history window
- the predicted next normalized state is appended back into the dynamic window, so state rollout is truly autoregressive
- the boundary window is advanced with the next ground-truth boundary value from the dataset, so inflow forcing is teacher-forced rather than predicted
- metrics and plots are reported after inverse normalization in physical units

## Dataset Loading and Normalization

The dataset stack lives under `datasets/`:

- [flood_dataset.py](datasets/flood_dataset.py): one-step training/validation dataset
- [rollout_dataset.py](datasets/rollout_dataset.py): full-sequence rollout dataset
- [normalized_dataset.py](datasets/normalized_dataset.py): lazy and eager normalization wrappers
- [cache_backend.py](datasets/cache_backend.py): shared raw-text/HDF5 backend

Current runtime behavior:

- raw text is cached into `<data_root>/.flood_cache/flood_forecaster_v1.h5`
- a manifest tracks cache invalidation via file size/mtime and dataset metadata
- geometry and static tensors stay resident
- per-run dynamic and boundary tensors are loaded through an in-process LRU cache
- geometry is normalized once to the unit box and is not renormalized at runtime
- query grids are built directly from that unit-box geometry
- dynamic and target share one per-channel state normalizer for `[WD, VX, VY]`
- boundary stays compact as `(history, bc_dim)` until the processor expands it across cells

Dataset classes:

- `FloodDatasetWithQueryPoints`: one-step windows for training/validation
- `FloodRolloutTestDatasetNew`: one full run at a time for rollout
- `LazyNormalizedDataset` / `LazyNormalizedRolloutDataset`: current runtime wrappers used by training and inference
- `NormalizedDataset` / `NormalizedRolloutTestDataset`: eager wrappers kept for compatibility and tests

The top-level `data_io` config block controls caching:

```yaml
data_io:
  backend: auto
  cache_dir_name: .flood_cache
  rebuild_cache: false
  run_cache_size: 4
```

`backend=auto` selects the HDF5 cache path when `h5py` is installed, otherwise it uses `raw_txt`. Set `backend=raw_txt` explicitly to bypass the cache for debugging or parity checks.

## Configuration Reference

### Data Paths

```yaml
source_data:
  root: "/path/to/source/data"
  xy_file: "M40_XY.txt"
  static_files: ["M40_XY.txt", "M40_CA.txt", "M40_CE.txt", "M40_CS.txt", "M40_FA.txt", "M40_A.txt", "M40_CU.txt"]
  dynamic_patterns:
    WD: "M40_WD_{}.txt"
    VX: "M40_VX_{}.txt"
    VY: "M40_VY_{}.txt"
  boundary_patterns:
    inflow: "M40_US_InF_{}.txt"
```

The `{}` placeholders above are run IDs from the list file, not timestep numbers.

### Model Settings

```yaml
model:
  model_arch: "gino"
  autoregressive: true
  data_channels: 20
  out_channels: 3
  fno_n_modes: [16, 16]
  fno_hidden_channels: 64
  gno_embed_channels: 32
```

This example is GINO-specific. `model_arch` is kept for neuralop config compatibility, but the surrounding wrapper, processor, checkpointing, and rollout code assume the GINO contract.

### Training Settings

```yaml
source_data:
  batch_size: 64

target_data:
  batch_size: 64

training:
  n_epochs_source: 30
  n_epochs_adapt: 20
  eval_interval: 1
  learning_rate: 1e-4
  adapt_learning_rate: 1e-4
  amp_autocast: false
  da_lambda_max: 1.0
  da_class_loss_weight: 0.1
```

The full/default configs enable adversarial domain adaptation with `da_class_loss_weight: 0.1`. `config_short.yaml` and `config_smoke.yaml` keep `da_class_loss_weight: 0.0` for fast supervised fine-tuning checks. The full-dataset single-GPU presets reduce batch and resolution for memory and speed. For example, `config_full_single_gpu.yaml` uses `batch_size: 1` with `query_res: [32, 32]`, and `config_full_single_gpu_small.yaml` keeps `query_res: [32, 32]` while reducing the model width/depth and using `batch_size: 2`.

### Domain Adaptation Classifier

```yaml
training:
  da_classifier:
    conv_layers:
      - out_channels: 64
        kernel_size: 3
        pool_size: 2
    fc_dim: 1
```

### Distributed Computing

FloodForecaster uses PhysicsNeMo's `DistributedManager`, which automatically detects single-process, `torchrun`, OpenMPI, and SLURM execution environments.

The shipped distributed block is intentionally small:

```yaml
distributed:
  seed: 123
  device: "cuda:0"
```

Behavior:

- in distributed CUDA runs, the effective device comes from `DistributedManager`
- the configured `distributed.device` value is only a single-process fallback
- loaders attach `DistributedSampler` automatically in distributed mode

Example multi-GPU launch:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --config-name config
```

## Evaluation Metrics

FloodForecaster computes both time-series and event-level metrics, including:

- RMSE for water depth and velocity components
- CSI at multiple water-depth thresholds
- arrival time and inundation duration error
- maximum momentum flux error
- flood hazard classification accuracy
- total water volume comparison
- conditional error analysis by water depth and velocity magnitude

Rollout outputs are saved as figures, animations, and `.npz` metric files.

## Logging

The example supports:

- standard Python/PhysicsNeMo logging
- optional Weights and Biases logging through the `wandb` config block

Tracked quantities include training/validation losses, learning rate, and domain-classification loss when the adversarial branch is enabled.

## Project Structure

```text
FloodForecaster/
|-- conf/
|   |-- config.yaml
|   |-- config_smoke.yaml
|   |-- config_short.yaml
|   |-- config_full_single_gpu.yaml
|   `-- config_full_single_gpu_small.yaml
|-- datasets/
|   |-- cache_backend.py
|   |-- flood_dataset.py
|   |-- normalized_dataset.py
|   `-- rollout_dataset.py
|-- models/
|   |-- gino_wrapper.py
|   `-- domain_classifier.py
|-- data_processing/
|   `-- data_processor.py
|-- training/
|   |-- pretraining.py
|   `-- domain_adaptation.py
|-- inference/
|   `-- rollout.py
|-- utils/
|   |-- checkpointing.py
|   |-- normalization.py
|   |-- runtime.py
|   `-- plotting.py
|-- tests/
|   |-- data/
|   `-- model_fixtures.py
|-- train.py
|-- inference.py
`-- README.md
```

Collected FloodForecaster pytest coverage lives in
`test/models/test_flood_forecaster.py`; the example-local `tests/` package holds
only reusable fixtures and reference artifacts.

## Notes

- GINO batching assumes shared geometry within a batch. If geometry varies across samples, use `batch_size: 1`.
- Large meshes can still be GPU-memory bound by the GINO forward pass even after the data-loading refactor.
- Mixed precision may be disabled automatically for unsupported latent-grid shapes.
- Training and adaptation are one-step supervised stages; autoregressive state updates happen only in [inference/rollout.py](inference/rollout.py), not in the trainer.
- The shipped rollout is autoregressive in the flood state but uses observed future boundary forcing from the dataset.

## Citation

If you use FloodForecaster in your research, please cite:

```bibtex
@article{taghizadeh2025floodforecaster,
  title     = {FloodForecaster: A domain-adaptive geometry-informed neural operator framework for rapid flood forecasting},
  author    = {Taghizadeh, Mehdi and Zandsalimi, Zanko and Nabian, Mohammad Amin and Goodall, Jonathan L. and Alemazkoor, Negin},
  journal   = {Journal of Hydrology},
  volume    = {664},
  pages     = {134512},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.jhydrol.2025.134512},
  url       = {https://doi.org/10.1016/j.jhydrol.2025.134512}
}
```

## Contact

- Mehdi Taghizadeh: <jrj6wm@virginia.edu>
- Zanko Zandsalimi: <mfx2uq@virginia.edu>
- Mohammad Amin Nabian: <mnabian@nvidia.com>
- Jonathan L. Goodall: <jlg7h@virginia.edu>
- Negin Alemazkoor: <na7fp@virginia.edu>
