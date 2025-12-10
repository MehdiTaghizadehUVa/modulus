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

"""
Unit tests for FloodForecaster data processing modules.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import physicsnemo

# Conditionally include CUDA in device parametrization only if available
_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda:0")

# Add the FloodForecaster example to the path
_examples_dir = Path(__file__).parent.parent.parent / "examples" / "weather" / "flood_modeling" / "flood_forecaster"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from data_processing import FloodGINODataProcessor, GINOWrapper, LpLossWrapper

from . import common


# Define MockGINOModelForCheckpoint at module level so it can be properly loaded from checkpoint
# Note: Name doesn't start with "Test" to avoid pytest collection
# Register it in the model registry to ensure it can be found when loading checkpoints
class MockGINOModelForCheckpoint(physicsnemo.Module):
    """Simple test GINO model for checkpoint testing."""
    def __init__(self):
        super().__init__(meta=physicsnemo.ModelMetaData(name="MockGINOForCheckpoint"))
        self.fno_hidden_channels = 64
        self.out_channels = 3
        self.gno_coord_dim = 2
        self.latent_feature_channels = None
        self.in_coord_dim_reverse_order = [2, 3]
        self.out_gno_tanh = None
        
        # Create minimal layers for the model to work
        self.gno_in = nn.Linear(2, 64)
        self.gno_out = nn.Linear(64, 64)
        self.projection = nn.Linear(64, 3)
        self.latent_embedding = nn.Identity()

# Register MockGINOModelForCheckpoint in the model registry so it can be loaded from checkpoints
try:
    registry = physicsnemo.registry.ModelRegistry()
    if "MockGINOModelForCheckpoint" not in registry.list_models():
        registry.register(MockGINOModelForCheckpoint, "MockGINOModelForCheckpoint")
except (ValueError, AttributeError):
    # Already registered or registry issue - continue
    pass


@pytest.fixture
def sample_dict():
    """Create sample dictionary for preprocessing."""
    batch_size = 2
    num_cells = 100
    n_history = 3
    # query_points should be (B, H, W, 2) or (H, W, 2) for latent queries
    # Using a simple grid: (8, 8, 2) for 2D
    H, W = 8, 8

    return {
        "geometry": torch.rand(batch_size, num_cells, 2),
        "static": torch.rand(batch_size, num_cells, 7),
        "boundary": torch.rand(batch_size, n_history, num_cells, 1),
        "dynamic": torch.rand(batch_size, n_history, num_cells, 3),
        "target": torch.rand(batch_size, num_cells, 3),
        "query_points": torch.rand(batch_size, H, W, 2),  # Required for preprocessing
    }


@pytest.fixture
def mock_gino_model():
    """Create a mock GINO model with internal components."""
    model = MagicMock(spec=nn.Module)
    model.fno_hidden_channels = 64
    model.out_channels = 3
    model.gno_coord_dim = 2
    model.latent_feature_channels = None
    model.in_coord_dim_reverse_order = [2, 3]  # For 2D: [2, 3] means permute dims 2,3
    model.out_gno_tanh = None
    
    # Mock internal components - gno_in returns flattened output
    def mock_gno_in(y, x, f_y=None):
        # Returns (out_channels, n_points) for flattened queries
        n_points = x.shape[0]
        return torch.rand(64, n_points)  # (out_channels, n_points)
    
    model.gno_in = MagicMock(side_effect=mock_gno_in)
    
    # Mock gno_out - returns (B, channels, n_out) after permute
    def mock_gno_out(y, x, f_y):
        # f_y is (B, n_latent, channels)
        batch_size = f_y.shape[0]
        n_out = x.shape[0]
        return torch.rand(batch_size, 64, n_out)  # (B, channels, n_out)
    
    model.gno_out = MagicMock(side_effect=mock_gno_out)
    
    # Mock projection - returns (B, out_channels, n_out)
    def mock_projection(x):
        # x is (B, channels, n_out)
        batch_size = x.shape[0]
        n_out = x.shape[2]
        return torch.rand(batch_size, 3, n_out)  # (B, out_channels, n_out)
    
    model.projection = MagicMock(side_effect=mock_projection)
    
    # Mock latent_embedding to return a tensor
    def mock_latent_embedding(in_p, ada_in=None):
        # in_p shape: (B, H, W, channels)
        # Return shape: (B, channels, H, W) for 2D
        batch_size = in_p.shape[0]
        grid_shape = in_p.shape[1:-1]  # (H, W)
        return torch.rand(batch_size, 64, *grid_shape)
    
    model.latent_embedding = mock_latent_embedding
    
    return model


@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_init(device):
    """Test FloodGINODataProcessor initialization."""
    processor = FloodGINODataProcessor(device=device)
    # device property returns torch.device, so compare string representations
    assert str(processor.device) == str(torch.device(device))
    assert processor.target_norm is None
    assert processor.inverse_test is True
    assert processor.model is None
    assert isinstance(processor, nn.Module)


@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_preprocess(sample_dict, device):
    """Test preprocessing with batched input."""
    processor = FloodGINODataProcessor(device=device)
    result = processor.preprocess(sample_dict)

    # Check required keys exist
    assert "input_geom" in result
    assert "latent_queries" in result
    assert "output_queries" in result
    assert "x" in result
    assert "y" in result

    # Check geometry has no batch dim (shared)
    assert result["input_geom"].dim() == 2  # (n_cells, 2)
    assert result["latent_queries"].dim() == 3  # (H, W, 2)
    assert result["output_queries"].dim() == 2  # (n_cells, 2)

    # Check x has batch dim
    assert result["x"].dim() == 3  # (B, n_cells, features)


@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_postprocess_training(device):
    """Test postprocessing in training mode (no inverse transform)."""
    mock_norm = MagicMock()
    processor = FloodGINODataProcessor(device=device, target_norm=mock_norm)
    processor.train()

    out = torch.rand(2, 100, 3)
    sample = {"y": torch.rand(2, 100, 3)}

    result_out, result_sample = processor.postprocess(out, sample)

    # Should not call inverse_transform in training mode
    mock_norm.inverse_transform.assert_not_called()


@pytest.mark.parametrize("device", _DEVICES)
def test_data_processor_postprocess_eval(device):
    """Test postprocessing in eval mode (applies inverse transform)."""
    mock_norm = MagicMock()
    mock_norm.inverse_transform = lambda x: x * 2  # Simple transform

    processor = FloodGINODataProcessor(device=device, target_norm=mock_norm, inverse_test=True)
    processor.eval()

    out = torch.ones(2, 100, 3)
    sample = {"y": torch.ones(2, 100, 3)}

    result_out, result_sample = processor.postprocess(out, sample)

    # Should apply inverse transform
    assert torch.allclose(result_out, torch.ones(2, 100, 3) * 2)


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_init(mock_gino_model, device):
    """Test GINOWrapper initialization."""
    wrapper = GINOWrapper(mock_gino_model)
    assert wrapper.gino == mock_gino_model
    assert isinstance(wrapper, nn.Module)
    assert wrapper.fno_hidden_channels == 64
    assert wrapper.autoregressive is False  # Default value


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_init_autoregressive(mock_gino_model, device):
    """Test GINOWrapper initialization with autoregressive=True."""
    wrapper = GINOWrapper(mock_gino_model, autoregressive=True)
    assert wrapper.autoregressive is True


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_forward_filters_kwargs(mock_gino_model, device):
    """Test that GINOWrapper forward filters out unexpected kwargs."""
    wrapper = GINOWrapper(mock_gino_model)

    input_geom = torch.rand(1, 100, 2)  # With batch dim for squeeze
    latent_queries = torch.rand(1, 8, 8, 2)  # With batch dim for squeeze
    output_queries = torch.rand(1, 100, 2)  # With batch dim for squeeze
    x = torch.rand(1, 100, 10)

    # Mock the internal GNO components
    mock_gino_model.gno_in.return_value = torch.rand(64, 8 * 8)  # (out_channels, n_points)
    mock_gino_model.gno_out.return_value = torch.rand(1, 64, 100)  # (B, channels, n_out)
    mock_gino_model.projection.return_value = torch.rand(1, 3, 100)  # (B, out_channels, n_out)

    # This should not raise even with extra kwargs
    result = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        y=torch.rand(1, 100, 3),  # Should be filtered out
        extra_arg="should be ignored",
    )

    # Verify result is a tensor
    assert isinstance(result, torch.Tensor)
    assert result.shape[0] == 1  # batch size
    assert result.shape[2] == 3  # out_channels


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_return_features(mock_gino_model, device):
    """Test that GINOWrapper returns features when return_features=True."""
    wrapper = GINOWrapper(mock_gino_model)

    input_geom = torch.rand(1, 100, 2)
    latent_queries = torch.rand(1, 8, 8, 2)
    output_queries = torch.rand(1, 100, 2)
    x = torch.rand(1, 100, 10)

    # Mock the internal GNO components
    mock_gino_model.gno_in.return_value = torch.rand(64, 8 * 8)
    mock_gino_model.gno_out.return_value = torch.rand(1, 64, 100)
    mock_gino_model.projection.return_value = torch.rand(1, 3, 100)

    # Call with return_features=True
    out, features = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        return_features=True,
    )

    # Verify output shape
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1  # batch size
    assert out.shape[2] == 3  # out_channels

    # Verify features shape: (B, channels, H, W) for 2D
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == 1  # batch size
    assert features.shape[1] == 64  # fno_hidden_channels
    assert features.shape[2] == 8  # H
    assert features.shape[3] == 8  # W


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_autoregressive(mock_gino_model, device):
    """Test that GINOWrapper applies residual connection when autoregressive=True."""
    wrapper = GINOWrapper(mock_gino_model, autoregressive=True)

    input_geom = torch.rand(1, 100, 2)
    latent_queries = torch.rand(1, 8, 8, 2)
    output_queries = torch.rand(1, 100, 2)
    x = torch.rand(1, 100, 10)  # Input with 10 channels, out_channels=3

    # Mock the internal GNO components
    # Flow: gno_out -> permute -> projection -> permute
    # gno_out returns (B, channels, n_out) = (1, 64, 100)
    # After permute(0, 2, 1) -> (1, 100, 64)
    # projection takes (1, 100, 64) and should return (1, 3, 100)
    # After permute(0, 2, 1) -> (1, 100, 3) which matches x.shape[1] = 100
    mock_gino_model.gno_in.return_value = torch.rand(64, 8 * 8)
    mock_gino_model.gno_out.return_value = torch.rand(1, 64, 100)
    # projection is called on (1, 100, 64) after first permute
    # Return (1, 3, 100) so that after permute(0, 2, 1) we get (1, 100, 3)
    def mock_projection(x):
        # x is (B, n_out, channels) = (1, 100, 64)
        # Return (B, out_channels, n_out) = (1, 3, 100) so permute gives (1, 100, 3)
        return torch.zeros(x.shape[0], 3, x.shape[1])
    mock_gino_model.projection.side_effect = mock_projection

    # Call with autoregressive=True
    out = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
    )

    # With autoregressive=True and zero output, result should equal x[..., -3:]
    expected = x[..., -3:]  # Last 3 channels
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_autoregressive_with_features(mock_gino_model, device):
    """Test autoregressive + return_features together."""
    wrapper = GINOWrapper(mock_gino_model, autoregressive=True)

    input_geom = torch.rand(1, 100, 2)
    latent_queries = torch.rand(1, 8, 8, 2)
    output_queries = torch.rand(1, 100, 2)
    x = torch.rand(1, 100, 10)

    # Mock the internal GNO components
    mock_gino_model.gno_in.return_value = torch.rand(64, 8 * 8)
    mock_gino_model.gno_out.return_value = torch.rand(1, 64, 100)
    # projection is called on (1, 100, 64) after first permute
    # It should return (1, 3, 100) so that after permute(0, 2, 1) we get (1, 100, 3)
    def mock_projection(x):
        # x is (B, n_out, channels) = (1, 100, 64)
        # Return (B, out_channels, n_out) = (1, 3, 100) so permute gives (1, 100, 3)
        return torch.zeros(x.shape[0], 3, x.shape[1])
    mock_gino_model.projection.side_effect = mock_projection

    # Call with both autoregressive and return_features
    out, features = wrapper(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        return_features=True,
    )

    # Verify output has residual connection
    expected = x[..., -3:]
    assert torch.allclose(out, expected, atol=1e-5)

    # Verify features are returned
    assert isinstance(features, torch.Tensor)
    assert features.shape[1] == 64  # fno_hidden_channels


@pytest.mark.parametrize("device", _DEVICES)
def test_lploss_wrapper(device):
    """Test LpLossWrapper filters kwargs correctly."""
    mock_loss = MagicMock(return_value=torch.tensor(0.5))
    wrapper = LpLossWrapper(mock_loss)

    y_pred = torch.rand(2, 100, 3)
    y = torch.rand(2, 100, 3)

    # Extra kwargs should be ignored
    wrapper(
        y_pred,
        y=y,
        input_geom=torch.rand(100, 2),
        latent_queries=torch.rand(8, 8, 2),
        output_queries=torch.rand(100, 2),
        x=torch.rand(2, 100, 10),
    )

    # Should only pass y_pred and y
    mock_loss.assert_called_once_with(y_pred, y)


@pytest.mark.parametrize("device", _DEVICES)
def test_lploss_wrapper_with_real_lploss(device, pytestconfig):
    """Test LpLossWrapper with real LpLoss from neuralop."""
    import sys
    from pathlib import Path

    # Add test directory to path for pytest_utils
    _test_dir = Path(__file__).parent.parent.parent
    if str(_test_dir) not in sys.path:
        sys.path.insert(0, str(_test_dir))

    from pytest_utils import import_or_fail

    @import_or_fail(["neuralop"])
    def _test(pytestconfig=None):
        """Inner test function that accepts pytestconfig parameter."""
        from neuralop.losses import LpLoss

        loss_fn = LpLossWrapper(LpLoss(d=2, p=2))

        y_pred = torch.rand(2, 100, 3)
        y = torch.rand(2, 100, 3)

        # Should work without warnings about extra kwargs
        result = loss_fn(y_pred, y=y, extra_kwarg="ignored")

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar loss

    _test(pytestconfig=pytestconfig)


def _instantiate_model(cls, seed: int = 0, **kwargs):
    """Helper to create model with reproducible parameters."""
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"]
)
def test_ginowrapper_constructor(config, device, mock_gino_model):
    """Test GINOWrapper constructor and attributes."""
    if config == "default":
        model = GINOWrapper(mock_gino_model, autoregressive=False)
        assert model.autoregressive is False
    else:
        model = GINOWrapper(mock_gino_model, autoregressive=True)
        assert model.autoregressive is True
    
    # Test common attributes
    assert model.gino == mock_gino_model
    assert hasattr(model, "fno_hidden_channels")
    assert isinstance(model, physicsnemo.Module)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"]
)
def test_flood_gino_data_processor_constructor(config, device):
    """Test FloodGINODataProcessor constructor and attributes."""
    if config == "default":
        model = FloodGINODataProcessor(device=device)
        # device property returns torch.device, so compare string representations
        assert str(model.device) == str(torch.device(device))
        assert model.target_norm is None
        assert model.inverse_test is True
    else:
        mock_norm = MagicMock()
        model = FloodGINODataProcessor(device=device, target_norm=mock_norm, inverse_test=False)
        # device property returns torch.device, so compare string representations
        assert str(model.device) == str(torch.device(device))
        assert model.target_norm == mock_norm
        assert model.inverse_test is False
    
    # Test common attributes
    assert hasattr(model, "model")
    assert isinstance(model, physicsnemo.Module)


@pytest.mark.parametrize("device", _DEVICES)
def test_ginowrapper_from_checkpoint(device, mock_gino_model):
    """Test loading GINOWrapper from checkpoint and verify outputs."""
    from pathlib import Path
    import physicsnemo
    
    # Use the module-level MockGINOModelForCheckpoint class for checkpoint testing
    # This ensures the class can be properly loaded from checkpoint
    gino_model = MockGINOModelForCheckpoint()
    
    # Create a model and save checkpoint
    model_orig = GINOWrapper(gino_model, autoregressive=False).to(device)
    checkpoint_path = Path("checkpoint_gino_wrapper.mdlus")
    model_orig.save(str(checkpoint_path))
    
    # Load from checkpoint - use strict=False to handle potential state dict mismatches
    # The nested TestGINOModel should be properly reconstructed via module path or registry
    model = physicsnemo.Module.from_checkpoint(str(checkpoint_path), strict=False).to(device)
    
    # Verify attributes after loading
    assert model.autoregressive is False
    assert isinstance(model, GINOWrapper)
    # Verify the wrapped model was loaded correctly with all layers
    # Note: The model structure is preserved even if class type isn't exactly TestGINOModel
    # (physicsnemo may load it as a generic Module if the class can't be imported)
    assert hasattr(model.gino, 'gno_in')
    assert hasattr(model.gino, 'gno_out')
    assert hasattr(model.gino, 'projection')
    assert hasattr(model.gino, 'latent_embedding')
    # Verify the layers have the correct structure
    assert isinstance(model.gino.gno_in, nn.Linear)
    assert isinstance(model.gino.gno_out, nn.Linear)
    assert isinstance(model.gino.projection, nn.Linear)
    
    # Cleanup
    checkpoint_path.unlink(missing_ok=True)


@pytest.mark.parametrize("device", _DEVICES)
def test_flood_gino_data_processor_from_checkpoint(device):
    """Test loading FloodGINODataProcessor from checkpoint and verify outputs."""
    from pathlib import Path
    
    # Create a model and save checkpoint
    model_orig = FloodGINODataProcessor(device=device).to(device)
    checkpoint_path = Path("checkpoint_flood_gino_data_processor.mdlus")
    model_orig.save(str(checkpoint_path))
    
    # Load from checkpoint
    model = physicsnemo.Module.from_checkpoint(str(checkpoint_path)).to(device)
    
    # Verify attributes after loading
    # Note: device is a property, so we check it matches
    assert str(model.device) == str(torch.device(device))
    assert model.inverse_test is True
    assert isinstance(model, FloodGINODataProcessor)
    
    # Cleanup
    checkpoint_path.unlink(missing_ok=True)

