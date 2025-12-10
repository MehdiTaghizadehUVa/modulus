### Added

- **FloodForecaster**: Added a domain-adaptive Geometry-Informed Neural Operator (GINO) framework for rapid, high-resolution flood forecasting in `examples/weather/flood_modeling/flood_forecaster/`. The framework enables accurate, real-time flood predictions by learning from source domain data and adapting to target domains through adversarial training. Key features include:
  - Three-stage training pipeline (pretraining → domain adaptation → rollout evaluation)
  - Gradient Reversal Layer (GRL) for adversarial domain adaptation
  - Integration with GINO models combining Graph Neural Operators (GNO) and Fourier Neural Operators (FNO)
  - Comprehensive physics-informed evaluation metrics (volume conservation, arrival time, inundation duration, CSI, FHCA)
  - Autoregressive multi-step forecasting capabilities
  - Full PhysicsNeMo Module compliance with checkpointing and serialization support
  - Comprehensive test suite with 57 tests covering all core components
  - Complete documentation with usage examples and dataset format specifications

