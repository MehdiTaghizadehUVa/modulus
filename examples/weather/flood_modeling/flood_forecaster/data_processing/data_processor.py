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
Data processor for GINO flood prediction model.

Compatible with neuralop 2.0.0 API.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

import physicsnemo
from physicsnemo.models.meta import ModelMetaData

try:
    from jaxtyping import Float
    HAS_JAXTYPING = True
except ImportError:
    HAS_JAXTYPING = False
    # Fallback type alias for when jaxtyping is not available
    Float = None


class LpLossWrapper:
    r"""
    Wrapper around LpLoss that filters out unexpected kwargs.
    
    The neuralop Trainer calls loss(out, **sample) where sample contains
    all keys including model inputs. This wrapper filters to only pass 'y'.
    """
    
    def __init__(self, loss_fn):
        r"""
        Initialize LpLoss wrapper.

        Parameters
        ----------
        loss_fn : callable
            The underlying loss function to wrap.

        Raises
        ------
        ValueError
            If ``loss_fn`` is None.
        """
        if loss_fn is None:
            raise ValueError("loss_fn cannot be None")
        self.loss_fn = loss_fn
        
    def __call__(self, y_pred: torch.Tensor, y: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        r"""
        Compute loss, filtering out unexpected kwargs.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values of shape :math:`(B, D)` where :math:`B` is batch size
            and :math:`D` is the number of output dimensions.
        y : torch.Tensor, optional
            Ground truth values of shape :math:`(B, D)`, optional.
        **kwargs
            Additional arguments (filtered out, not used).

        Returns
        -------
        torch.Tensor
            Loss value as a scalar tensor.
        """
        # Ignore all kwargs except y - silently filter out model input keys
        return self.loss_fn(y_pred, y)


class GINOWrapper(physicsnemo.Module):
    r"""
    Enhanced wrapper around GINO model that adds enhanced functionality.
    
    This wrapper adds:
    1. Autoregressive residual connection support
    2. Feature extraction support (return_features)
    3. Filters out unexpected kwargs
    
    The neuralop Trainer calls model(**sample) where sample contains
    both model inputs AND 'y' for loss computation. This wrapper
    filters out 'y' before passing to GINO to avoid warnings.
    
    This wrapper restores functionality from the older GINO version:
    - autoregressive: If True, adds residual connection: ``out = x[..., :out_channels] + predicted_delta``
    - return_features: If True, returns (out, latent_embed) tuple for domain adaptation

    Parameters
    ----------
    model : nn.Module
        The GINO model to wrap.
    autoregressive : bool, optional, default=False
        If True, enable residual connection for autoregressive time-stepping.

    Forward
    -------
    input_geom : torch.Tensor
        Input geometry tensor of shape :math:`(n_{in}, D)` or :math:`(1, n_{in}, D)`
        where :math:`n_{in}` is the number of input points and :math:`D` is the
        coordinate dimension (typically 2).
    latent_queries : torch.Tensor
        Latent query points of shape :math:`(H, W, D)` or :math:`(1, H, W, D)`
        where :math:`H, W` are spatial dimensions.
    output_queries : torch.Tensor or Dict[str, torch.Tensor]
        Output query points of shape :math:`(n_{out}, D)` or :math:`(1, n_{out}, D)`,
        or a dictionary of such tensors.
    x : torch.Tensor, optional
        Input features of shape :math:`(B, n_{in}, C_{in})` where :math:`B` is batch size
        and :math:`C_{in}` is the number of input channels.
    latent_features : torch.Tensor, optional
        Latent features of shape :math:`(B, H, W, C_{feat})` where :math:`C_{feat}` is
        the number of feature channels.
    ada_in : torch.Tensor, optional
        Adaptive input tensor.
    return_features : bool, optional, default=False
        If True, return (output, latent_embed) tuple for domain adaptation.

    Outputs
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Model output tensor of shape :math:`(B, n_{out}, C_{out})`, or
        (output, features) tuple if ``return_features=True``. Features are the
        latent embedding from FNO blocks of shape :math:`(B, C, H, W)` for 2D.
    """
    
    def __init__(self, model: nn.Module, autoregressive: bool = False):
        r"""
        Initialize GINO wrapper.

        Parameters
        ----------
        model : nn.Module
            The GINO model to wrap.
        autoregressive : bool, optional, default=False
            If True, enable residual connection for autoregressive time-stepping.

        Raises
        ------
        ValueError
            If ``model`` is None.
        """
        super().__init__(meta=ModelMetaData(name="GINOWrapper"))
        if model is None:
            raise ValueError("model cannot be None")
        # Register as a submodule so it's properly tracked by PyTorch
        # This ensures the model is properly stored and accessible
        # Always store as both submodule (for PyTorch) and direct attribute (for easy access)
        if isinstance(model, nn.Module):
            self.add_module('gino', model)
        # Also store as direct attribute for easy access (works for both Module and non-Module)
        self.gino = model
        self.autoregressive = autoregressive
        
    def forward(
        self, 
        input_geom: torch.Tensor, 
        latent_queries: torch.Tensor, 
        output_queries: torch.Tensor, 
        x: Optional[torch.Tensor] = None, 
        latent_features: Optional[torch.Tensor] = None, 
        ada_in: Optional[torch.Tensor] = None,
        return_features: bool = False,
        **kwargs
    ):
        r"""
        Forward pass through wrapped GINO model with enhanced features.
        
        This method replicates the new GINO's forward logic but adds:
        1. Autoregressive residual connection support
        2. Feature extraction support (return_features)

        Parameters
        ----------
        input_geom : torch.Tensor
            Input geometry tensor of shape :math:`(n_{in}, D)` or :math:`(1, n_{in}, D)`.
        latent_queries : torch.Tensor
            Latent query points of shape :math:`(H, W, D)` or :math:`(1, H, W, D)`.
        output_queries : torch.Tensor or Dict[str, torch.Tensor]
            Output query points of shape :math:`(n_{out}, D)` or :math:`(1, n_{out}, D)`,
            or a dictionary of such tensors.
        x : torch.Tensor, optional
            Input features of shape :math:`(B, n_{in}, C_{in})`.
        latent_features : torch.Tensor, optional
            Latent features of shape :math:`(B, H, W, C_{feat})`.
        ada_in : torch.Tensor, optional
            Adaptive input tensor.
        return_features : bool, optional, default=False
            If True, return (output, latent_embed) tuple.
        **kwargs
            Additional arguments (filtered out, including 'y').

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Model output tensor of shape :math:`(B, n_{out}, C_{out})`, or
            (output, features) tuple if ``return_features=True``. Features are the
            latent embedding from FNO blocks of shape :math:`(B, C, H, W)` for 2D.
        """
        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            # Validate input_geom shape
            if input_geom.ndim not in [2, 3]:
                raise ValueError(
                    f"Expected input_geom to be 2D or 3D tensor, got {input_geom.ndim}D tensor "
                    f"with shape {tuple(input_geom.shape)}"
                )
            if input_geom.ndim == 3 and input_geom.shape[0] != 1:
                raise ValueError(
                    f"Expected input_geom with batch dim to have shape (1, n_in, D), "
                    f"got {tuple(input_geom.shape)}"
                )
            
            # Validate latent_queries shape
            if latent_queries.ndim not in [3, 4]:
                raise ValueError(
                    f"Expected latent_queries to be 3D or 4D tensor, got {latent_queries.ndim}D tensor "
                    f"with shape {tuple(latent_queries.shape)}"
                )
            if latent_queries.ndim == 4 and latent_queries.shape[0] != 1:
                raise ValueError(
                    f"Expected latent_queries with batch dim to have shape (1, H, W, D), "
                    f"got {tuple(latent_queries.shape)}"
                )
            
            # Validate x shape if provided
            if x is not None:
                if x.ndim != 3:
                    raise ValueError(
                        f"Expected x to be 3D tensor (B, n_in, C_in), got {x.ndim}D tensor "
                        f"with shape {tuple(x.shape)}"
                    )
                # Check consistency with input_geom
                n_in_geom = input_geom.shape[-2] if input_geom.ndim == 2 else input_geom.shape[1]
                if x.shape[1] != n_in_geom:
                    raise ValueError(
                        f"Expected x.shape[1] ({x.shape[1]}) to match input_geom n_in ({n_in_geom})"
                    )
        
        # Filter out unexpected kwargs (like 'y') - just ignore them silently
        
        # Determine batch size (matching new GINO logic)
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        # Handle latent_features validation (matching new GINO)
        if latent_features is not None:
            if self.gino.latent_feature_channels is None:
                raise ValueError("if passing latent features, latent_feature_channels must be set.")
            if latent_features.shape[-1] != self.gino.latent_feature_channels:
                raise ValueError(f"latent_features.shape[-1] must equal latent_feature_channels")
            if latent_features.ndim != self.gino.gno_coord_dim + 2:
                raise ValueError(
                    f"Latent features must be of shape (batch, n_gridpts_1, ...n_gridpts_n, feat_dim), "
                    f"got {latent_features.shape}"
                )
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = latent_features.repeat(batch_size, *[1]*(latent_features.ndim-1))
        
        # Handle input geometry and queries (matching new GINO: squeeze(0))
        input_geom = input_geom.squeeze(0)
        latent_queries = latent_queries.squeeze(0)
        
        # Pass through input GNOBlock (matching new GINO exactly)
        in_p = self.gino.gno_in(
            y=input_geom, 
            x=latent_queries.view((-1, latent_queries.shape[-1])), 
            f_y=x
        )
        
        # Reshape to grid format (matching new GINO)
        grid_shape = latent_queries.shape[:-1]  # (H, W) for 2D
        in_p = in_p.view((batch_size, *grid_shape, -1))
        
        # Concatenate latent features if provided (matching new GINO)
        if latent_features is not None:
            in_p = torch.cat((in_p, latent_features), dim=-1)
        
        # Get latent embedding (this is what we need for feature extraction)
        # This matches new GINO's latent_embedding call
        latent_embed = self.gino.latent_embedding(in_p=in_p, ada_in=ada_in)
        # latent_embed shape: (B, channels, H, W) for 2D - keep this for feature return
        
        # Prepare for output GNO (matching new GINO exactly)
        # latent_embed shape (b, c, n_1, n_2, ..., n_k)
        batch_size = latent_embed.shape[0]
        # permute to (b, n_1, n_2, ...n_k, c) then reshape to (b, n_1 * n_2 * ...n_k, c)
        # Note: new GINO reassigns latent_embed here, but we keep both versions
        latent_embed_flat = latent_embed.permute(
            0, *self.gino.in_coord_dim_reverse_order, 1
        ).reshape(batch_size, -1, self.gino.fno_hidden_channels)
        
        # Apply tanh if needed (matching new GINO - applied after reshape to flattened version)
        if self.gino.out_gno_tanh in ["latent_embed", "both"]:
            latent_embed_flat = torch.tanh(latent_embed_flat)
            # Also apply to unflattened latent_embed for feature return consistency
            latent_embed = torch.tanh(latent_embed)
        
        # Handle output_queries (matching new GINO logic)
        if isinstance(output_queries, dict):
            # Multiple output queries - handle each separately
            out = {}
            for key, out_p in output_queries.items():
                out_p = out_p.squeeze(0)
                
                sub_output = self.gino.gno_out(
                    y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                    x=out_p,
                    f_y=latent_embed_flat,
                )
                sub_output = sub_output.permute(0, 2, 1)
                sub_output = self.gino.projection(sub_output).permute(0, 2, 1)
                
                # Apply residual connection if autoregressive (NEW FUNCTIONALITY)
                if self.autoregressive and (x is not None):
                    if sub_output.shape[1] != x.shape[1]:
                        raise ValueError(
                            f"Autoregressive skip requires out.shape[1] == x.shape[1], "
                            f"got {sub_output.shape[1]} vs {x.shape[1]}."
                        )
                    if self.gino.out_channels > x.shape[2]:
                        raise ValueError(
                            f"Cannot skip-add: out_channels {self.gino.out_channels} > in_channels {x.shape[2]}."
                        )
                    prev_step = x[..., -self.gino.out_channels:]
                    sub_output = sub_output + prev_step
                
                out[key] = sub_output
        else:
            # Single output query (matching new GINO)
            output_queries = output_queries.squeeze(0)
            
            out = self.gino.gno_out(
                y=latent_queries.reshape((-1, latent_queries.shape[-1])),
                x=output_queries,
                f_y=latent_embed_flat,
            )
            out = out.permute(0, 2, 1)
            out = self.gino.projection(out).permute(0, 2, 1)
            
            # Apply residual connection if autoregressive (NEW FUNCTIONALITY)
            if self.autoregressive and (x is not None):
                if out.shape[1] != x.shape[1]:
                    raise ValueError(
                        f"Autoregressive skip requires out.shape[1] == x.shape[1], "
                        f"got {out.shape[1]} vs {x.shape[1]}."
                    )
                if self.gino.out_channels > x.shape[2]:
                    raise ValueError(
                        f"Cannot skip-add: out_channels {self.gino.out_channels} > in_channels {x.shape[2]}."
                    )
                prev_step = x[..., -self.gino.out_channels:]
                out = out + prev_step
        
        # Return features if requested (NEW FUNCTIONALITY)
        if return_features:
            return out, latent_embed
        else:
            return out
    
    @property
    def fno_hidden_channels(self) -> int:
        r"""Expose GINO's fno_hidden_channels for domain classifier."""
        return self.gino.fno_hidden_channels
    
    @property  
    def model(self) -> nn.Module:
        r"""Alias for the wrapped model."""
        return self.gino
    
    def __getattr__(self, name):
        r"""
        Delegate attribute access to the underlying GINO model.
        
        This allows the wrapper to be used as a drop-in replacement.
        """
        # 'gino' and 'autoregressive' are wrapper attributes, should be accessible directly
        # If we get here, it means normal attribute lookup failed
        # For 'gino', try to get from _modules (if registered as submodule)
        if name == 'gino':
            if 'gino' in self._modules:
                return self._modules['gino']
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name == 'autoregressive':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # For other attributes, delegate to gino
        # Get gino (should exist as attribute or in _modules)
        gino = None
        if hasattr(self, 'gino'):
            gino = self.gino
        elif 'gino' in self._modules:
            gino = self._modules['gino']
        
        if gino is not None:
            try:
                return getattr(gino, name)
            except AttributeError:
                pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def save_checkpoint(self, save_folder: str, save_name: str) -> None:
        r"""
        Delegate checkpoint saving to wrapped GINO model.

        Parameters
        ----------
        save_folder : str
            Directory to save checkpoint.
        save_name : str
            Name of checkpoint file.

        Raises
        ------
        AttributeError
            If wrapped model does not have ``save_checkpoint`` method.
        """
        if not hasattr(self.gino, 'save_checkpoint'):
            raise AttributeError(f"Wrapped model {type(self.gino).__name__} does not have save_checkpoint method")
        return self.gino.save_checkpoint(save_folder, save_name)
    
    @classmethod
    def from_checkpoint(cls, save_folder: str, save_name: str, map_location: Optional[str] = None):
        r"""
        Load from checkpoint - delegate to GINO.

        Parameters
        ----------
        save_folder : str
            Directory containing checkpoint.
        save_name : str
            Name of checkpoint file.
        map_location : str, optional
            Device to map checkpoint to.

        Returns
        -------
        GINOWrapper
            GINOWrapper instance with loaded model.
        """
        from neuralop.models import GINO
        gino = GINO.from_checkpoint(save_folder, save_name, map_location)
        return cls(gino)


class FloodGINODataProcessor(physicsnemo.Module):
    r"""
    Data processor for flood GINO model that handles preprocessing and postprocessing.
    
    Compatible with neuralop 2.0.0 DataProcessor interface.
    
    GINO batching note:
        GINO supports batching only when geometry is SHARED across the batch.
        - input_geom, latent_queries, output_queries: NO batch dimension (shared)
        - x (features): HAS batch dimension :math:`(B, n_{in}, C_{in})`
        - y (target): HAS batch dimension :math:`(B, n_{out}, C_{out})`

    Parameters
    ----------
    device : str or torch.device, optional, default="cuda"
        Device to move tensors to (string like "cuda" or "cpu", or torch.device object).
    target_norm : nn.Module, optional
        Target normalizer for inverse transform.
    inverse_test : bool, optional, default=True
        Whether to apply inverse transform during testing.
    """

    def __init__(
        self, 
        device: Union[str, torch.device] = "cuda", 
        target_norm: Optional[nn.Module] = None, 
        inverse_test: bool = True
    ):
        r"""
        Initialize flood GINO data processor.

        Parameters
        ----------
        device : str or torch.device, optional, default="cuda"
            Device to move tensors to (string like "cuda" or "cpu", or torch.device object).
        target_norm : nn.Module, optional
            Target normalizer for inverse transform.
        inverse_test : bool, optional, default=True
            Whether to apply inverse transform during testing.

        Raises
        ------
        TypeError
            If ``device`` is not a string or torch.device.
        """
        super().__init__(meta=ModelMetaData(name="FloodGINODataProcessor"))
        # Accept both string and torch.device objects - preserve original type
        if not isinstance(device, (str, torch.device)):
            raise TypeError(f"device must be a string or torch.device, got {type(device)}")
        # Store device string for reference, but use super().to() to actually move module
        # Note: physicsnemo.Module has a read-only 'device' property, so we can't set self.device directly
        self._device_str = str(device) if isinstance(device, torch.device) else device
        # Move module to device using parent class method
        super().to(device)
        self.model: Optional[nn.Module] = None
        self.target_norm = target_norm
        self.inverse_test = inverse_test

    def preprocess(self, sample: Dict, batched: bool = True) -> Dict:
        r"""
        Preprocess sample for GINO model input.
        
        GINO expects:
        - input_geom: :math:`(n_{in}, 2)` - NO batch dim, shared geometry
        - latent_queries: :math:`(H, W, 2)` - NO batch dim
        - output_queries: :math:`(n_{out}, 2)` - NO batch dim
        - x: :math:`(B, n_{in}, C_{in})` - HAS batch dim
        - y: :math:`(B, n_{out}, C_{out})` - HAS batch dim (for loss)

        Parameters
        ----------
        sample : Dict
            Sample dictionary with geometry, static, boundary, dynamic, etc.
        batched : bool, optional, default=True
            Whether the data is batched (currently unused but kept for API compatibility).

        Returns
        -------
        Dict
            New dictionary with GINO-compatible keys + y for loss.

        Raises
        ------
        KeyError
            If required keys are missing from sample.
        RuntimeError
            If tensor shapes are incompatible.
        """
        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            # Validate required keys
            required_keys = ["geometry", "static", "boundary", "dynamic", "query_points"]
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                raise KeyError(f"Missing required keys in sample: {missing_keys}")
            
            # Validate tensor shapes
            for key in ["geometry", "static", "boundary", "dynamic", "query_points"]:
                if key in sample and isinstance(sample[key], torch.Tensor):
                    if sample[key].ndim < 2:
                        raise ValueError(
                            f"Expected {key} to be at least 2D tensor, got {sample[key].ndim}D tensor "
                            f"with shape {tuple(sample[key].shape)}"
                        )
        
        # Move all tensors to device
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(self.device)

        # Get batch dimension info
        dyn_ = sample["dynamic"]
        # dynamic comes as (B, n_history, num_cells, 3) or (n_history, num_cells, 3)
        if dyn_.dim() == 3:
            # Single sample: (n_history, num_cells, 3) -> add batch dim
            dyn_ = dyn_.unsqueeze(0)
        # Now dyn_ is (B, n_history, num_cells, 3)
        # Reshape to (B, num_cells, n_history * 3)
        dyn_ = dyn_.permute(0, 2, 1, 3)  # (B, num_cells, n_history, 3)
        B, N, H, D = dyn_.shape
        dyn_ = dyn_.reshape(B, N, H * D)

        # boundary: (B, n_history, num_cells, bc_dim) or (n_history, num_cells, bc_dim)
        bc_ = sample["boundary"]
        if bc_.dim() == 3:
            bc_ = bc_.unsqueeze(0)
        bc_ = bc_.permute(0, 2, 1, 3)  # (B, num_cells, n_history, bc_dim)
        B2, N2, H2, C2 = bc_.shape
        bc_ = bc_.reshape(B2, N2, H2 * C2)

        # static: (B, num_cells, static_dim) or (num_cells, static_dim)
        st_ = sample["static"]
        if st_.dim() == 2:
            st_ = st_.unsqueeze(0)

        # Concatenate all features: [static, boundary, dynamic]
        # x shape: (B, num_cells, total_features)
        x_ = torch.cat([st_, bc_, dyn_], dim=2)

        # geometry: (B, num_cells, 2) or (num_cells, 2)
        # GINO expects geometry WITHOUT batch dim - use first sample's geometry (shared)
        geom_ = sample["geometry"]
        if geom_.dim() == 3:
            # Take first sample's geometry (all should be same for GINO batching)
            geom_ = geom_[0]
        # geom_ is now (num_cells, 2) - NO batch dim

        # target (y): (B, num_cells, 3) or (num_cells, 3)
        y_ = sample.get("target", None)
        if y_ is not None and y_.dim() == 2:
            y_ = y_.unsqueeze(0)

        # query_points: (B, H, W, 2) or (H, W, 2)
        # GINO expects latent_queries WITHOUT batch dim
        q_ = sample["query_points"]
        if q_.dim() == 4:
            # Take first sample's query points (should be same for all)
            q_ = q_[0]
        # q_ is now (H, W, 2) - NO batch dim

        # Return ONLY the keys needed (GINO inputs + y for loss)
        return {
            "input_geom": geom_,           # (n_in, 2) - NO batch
            "latent_queries": q_,          # (H, W, 2) - NO batch
            "output_queries": geom_.clone(),  # (n_out, 2) - NO batch
            "x": x_,                       # (B, n_in, features) - HAS batch
            "y": y_,                       # (B, n_out, 3) - HAS batch
        }

    def postprocess(self, out: torch.Tensor, sample: Dict) -> Tuple[torch.Tensor, Dict]:
        r"""
        Postprocess model output.

        Parameters
        ----------
        out : torch.Tensor
            Model output tensor of shape :math:`(B, n_{out}, C_{out})`.
        sample : Dict
            Sample dictionary.

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            Tuple of (postprocessed output, sample).
        """
        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if not isinstance(out, torch.Tensor):
                raise ValueError(
                    f"Expected out to be torch.Tensor, got {type(out)}"
                )
            if out.ndim < 2:
                raise ValueError(
                    f"Expected out to be at least 2D tensor (B, n_out, C_out), "
                    f"got {out.ndim}D tensor with shape {tuple(out.shape)}"
                )
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Expected sample to be dict, got {type(sample)}"
                )
        
        if (not self.training) and self.inverse_test and (self.target_norm is not None):
            out = self.target_norm.inverse_transform(out)
            if sample.get("y") is not None:
                sample["y"] = self.target_norm.inverse_transform(sample["y"])
        return out, sample

    def to(self, device: Union[str, torch.device]):
        r"""
        Move processor to device.

        Parameters
        ----------
        device : str or torch.device
            Target device (string like "cuda" or "cpu", or torch.device object).

        Returns
        -------
        FloodGINODataProcessor
            Self for method chaining.

        Raises
        ------
        TypeError
            If ``device`` is not a string or torch.device.
        """
        # Accept both string and torch.device objects - preserve original type
        if not isinstance(device, (str, torch.device)):
            raise TypeError(f"device must be a string or torch.device, got {type(device)}")
        
        # Update device string reference
        self._device_str = str(device) if isinstance(device, torch.device) else device
        # Move module to device using parent class method (physicsnemo.Module has read-only device property)
        super().to(device)
        if self.target_norm is not None:
            self.target_norm = self.target_norm.to(device)
        return self

    def wrap(self, model: nn.Module):
        r"""
        Wrap model with this processor.

        Parameters
        ----------
        model : nn.Module
            Model to wrap.

        Returns
        -------
        FloodGINODataProcessor
            Self for method chaining.

        Raises
        ------
        ValueError
            If ``model`` is None.
        """
        if model is None:
            raise ValueError("model cannot be None")
        self.model = model
        return self

    def forward(self, **data_dict) -> Tuple[torch.Tensor, Dict]:
        r"""
        Forward pass through processor and model.

        Parameters
        ----------
        **data_dict
            Input data dictionary.

        Returns
        -------
        Tuple[torch.Tensor, Dict]
            Tuple of (output tensor, processed data dictionary).

        Raises
        ------
        RuntimeError
            If no model is attached.
        """
        data_dict = self.preprocess(data_dict)
        if self.model is None:
            raise RuntimeError("No model attached. Call wrap(model).")
        
        model_input = {
            "input_geom": data_dict["input_geom"],
            "latent_queries": data_dict["latent_queries"],
            "output_queries": data_dict["output_queries"],
            "x": data_dict["x"],
        }
        out = self.model(**model_input)
        
        out, data_dict = self.postprocess(out, data_dict)
        return out, data_dict
