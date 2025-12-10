<!-- markdownlint-disable MD013-->

# PhysicsNeMo Pull Request

## Description

This PR integrates **FloodForecaster**, a domain-adaptive Geometry-Informed Neural Operator (GINO) framework for rapid, high-resolution flood forecasting. The framework enables accurate, real-time flood predictions by learning from source domain data and adapting to target domains through adversarial training.

### Key Features

- **Domain-Adaptive Training**: Three-stage pipeline (pretraining → domain adaptation → rollout evaluation) for transfer learning from data-rich to data-scarce domains
- **Gradient Reversal Layer (GRL)**: Adversarial domain adaptation using CNN-based domain classifier
- **GINO Integration**: Combines Graph Neural Operators (GNO) and Fourier Neural Operators (FNO) for irregular terrain processing
- **Physics-Informed Metrics**: Volume conservation, arrival time, inundation duration, CSI, and FHCA
- **PhysicsNeMo Module Compliance**: All components inherit from `physicsnemo.Module` with checkpointing support

### Components Added

- Training modules: domain adaptation trainer, pretraining pipeline
- Data processing: GINO wrapper, preprocessing/postprocessing utilities
- Datasets: Custom dataset classes for flood data loading
- Inference: Rollout evaluation and visualization pipeline
- Configuration: Hydra-based configuration system
- Documentation: Complete README with usage examples

## Checklist

- [x] I am familiar with the [Contributing Guidelines](https://github.com/NVIDIA/physicsnemo/blob/main/CONTRIBUTING.md).
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [ ] The [CHANGELOG.md](https://github.com/NVIDIA/physicsnemo/blob/main/CHANGELOG.md) is up to date with these changes.
- [ ] An [issue](https://github.com/NVIDIA/physicsnemo/issues) is linked to this pull request.

## Dependencies

No new dependencies required. All packages are either already in PhysicsNeMo or standard scientific computing libraries:
- `neuralop>=2.0.0` (existing)
- `hydra-core>=1.2.0` (existing)
- `wandb>=0.12.0` (optional, for logging)
- Standard packages: `matplotlib`, `tqdm`, `numpy`, `torch`, `omegaconf`, `pandas`, `h5py`
