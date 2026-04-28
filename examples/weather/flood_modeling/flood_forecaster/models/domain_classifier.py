r"""FloodForecaster domain-classifier layers."""

import json
from typing import Any, Optional

import physicsnemo
import torch
import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from torch.autograd import Function

from physicsnemo.core.module import ModelMetaData

from models.types import ClassifierInputTensor, ClassifierOutputTensor


def _sanitize_args_for_json(obj: Any) -> Any:
    r"""Recursively convert config objects to JSON-serializable structures."""
    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)

    if isinstance(obj, dict):
        return {k: _sanitize_args_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_args_for_json(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


class GradientReversalFunction(Function):
    r"""Autograd primitive that reverses gradients during the backward pass."""

    @staticmethod
    def forward(
        ctx, x: Float[torch.Tensor, "*shape"], lambda_: float
    ) -> Float[torch.Tensor, "*shape"]:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversal(physicsnemo.Module):
    r"""
    Gradient reversal layer with configurable strength.

    Parameters
    ----------
    lambda_ : float, optional, default=1.0
        Gradient scaling factor applied during the backward pass.
    lambda_max : float, optional
        Backward-compatible alias for ``lambda_``.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(d_{1}, \ldots, d_{n})`.

    Outputs
    -------
    torch.Tensor
        Tensor of shape :math:`(d_{1}, \ldots, d_{n})` with identity forward values
        and reversed gradients during backpropagation.
    """

    def __init__(self, lambda_: float = 1.0, lambda_max: Optional[float] = None):
        super().__init__(meta=ModelMetaData())
        if lambda_max is not None:
            lambda_ = lambda_max
        self.lambda_ = lambda_

    def forward(
        self, x: Float[torch.Tensor, "*shape"]
    ) -> Float[torch.Tensor, "*shape"]:
        r"""
        Apply the gradient reversal transform.

        Parameters
        ----------
        x : Float[torch.Tensor, "*shape"]
            Input tensor of shape :math:`(d_{1}, \ldots, d_{n})`.

        Returns
        -------
        Float[torch.Tensor, "*shape"]
            Identity output tensor with reversed gradients in the backward pass.
        """
        if not torch.compiler.is_compiling():
            if not isinstance(x, torch.Tensor):
                raise TypeError("Expected ``x`` to be a torch.Tensor")
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, value: float) -> None:
        r"""
        Update the gradient-reversal strength.

        Parameters
        ----------
        value : float
            New gradient scaling factor.

        Returns
        -------
        None
            Updates the layer in place.
        """
        self.lambda_ = value


class MetaData(ModelMetaData):
    r"""Metadata describing the FloodForecaster domain classifier."""

    name: str = "FloodForecaster Domain Classifier"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    bf16: bool = True
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = True


class CNNDomainClassifier(physicsnemo.Module):
    r"""
    CNN classifier for domain discrimination on latent flood features.

    Parameters
    ----------
    in_channels : int
        Number of latent feature channels expected at the classifier input.
    da_cfg : Any, optional, default=None
        Domain-adaptation configuration containing ``conv_layers`` and ``fc_dim``.
    conv_layers : list, optional, default=None
        Optional explicit convolution-layer configuration used when ``da_cfg`` is not provided.
    fc_dim : int, optional, default=None
        Output width of the classifier head when ``da_cfg`` is not provided.
    lambda_max : float, optional, default=1.0
        Maximum gradient-reversal strength used during adversarial training.
    meta : ModelMetaData, optional, default=None
        PhysicsNeMo metadata attached to the classifier module.

    Forward
    -------
    x : torch.Tensor
        Latent feature tensor of shape :math:`(B, C_{in}, H, W)`.

    Outputs
    -------
    torch.Tensor
        Domain-logit tensor of shape :math:`(B, C_{logit})`.
    """

    def __init__(
        self,
        in_channels: int,
        da_cfg: Optional[Any] = None,
        conv_layers: Optional[list] = None,
        fc_dim: Optional[int] = None,
        lambda_max: float = 1.0,
        meta: Optional[ModelMetaData] = None,
    ):
        super().__init__(meta=meta or MetaData())
        if in_channels <= 0:
            raise ValueError(f"Expected ``in_channels`` > 0, got {in_channels}")
        if lambda_max < 0:
            raise ValueError(f"Expected ``lambda_max`` >= 0, got {lambda_max}")

        if da_cfg is None:
            da_cfg = {"conv_layers": conv_layers or [], "fc_dim": fc_dim or 1}
        elif isinstance(da_cfg, DictConfig):
            da_cfg = OmegaConf.to_container(da_cfg, resolve=True)

        if conv_layers is None:
            conv_layers = da_cfg.get("conv_layers", [])
        if fc_dim is None:
            fc_dim = da_cfg.get("fc_dim", 1)
        if fc_dim <= 0:
            raise ValueError(f"Expected ``fc_dim`` > 0, got {fc_dim}")
        if not isinstance(conv_layers, list):
            raise TypeError("Expected ``conv_layers`` to be a list of layer configs")

        self.in_channels = in_channels
        self.da_cfg = _sanitize_args_for_json(da_cfg)
        self.conv_layers_cfg = _sanitize_args_for_json(conv_layers)
        self.fc_dim = int(fc_dim)
        self.lambda_max = float(lambda_max)

        # physicsnemo.Module captures raw __init__ arguments in __new__, before
        # this constructor can normalize DictConfig inputs. Keep the checkpoint
        # args aligned with the sanitized runtime attributes so .mdlus saves stay
        # JSON-serializable when callers pass Hydra configs.
        checkpoint_args = self._args["__args__"]
        checkpoint_args["da_cfg"] = self.da_cfg
        checkpoint_args["conv_layers"] = self.conv_layers_cfg
        checkpoint_args["fc_dim"] = self.fc_dim
        checkpoint_args["lambda_max"] = self.lambda_max
        checkpoint_args["meta"] = None

        layers = []
        current_channels = in_channels
        for layer_cfg in conv_layers:
            if not isinstance(layer_cfg, dict):
                raise TypeError("Expected each conv-layer config to be a dict")
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            pool_size = layer_cfg["pool_size"]
            if out_channels <= 0:
                raise ValueError(
                    f"Expected ``out_channels`` > 0, got {out_channels}"
                )
            if kernel_size <= 0:
                raise ValueError(
                    f"Expected ``kernel_size`` > 0, got {kernel_size}"
                )
            if pool_size <= 0:
                raise ValueError(f"Expected ``pool_size`` > 0, got {pool_size}")

            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(pool_size))
            current_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_channels, fc_dim)
        self.grl = GradientReversal(lambda_=lambda_max)

    def forward(self, x: ClassifierInputTensor) -> ClassifierOutputTensor:
        r"""
        Predict domain logits from latent features.

        Parameters
        ----------
        x : ClassifierInputTensor
            Latent feature tensor of shape :math:`(B, C_{in}, H, W)`.

        Returns
        -------
        ClassifierOutputTensor
            Domain-logit tensor of shape :math:`(B, C_{logit})`.
        """
        if not torch.compiler.is_compiling():
            if not isinstance(x, torch.Tensor):
                raise TypeError("Expected ``x`` to be a torch.Tensor")
            if x.ndim != 4:
                raise ValueError(
                    "Expected tensor of shape :math:`(B, C_{in}, H, W)` "
                    f"but got tensor of shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    "Expected tensor with "
                    f"{self.in_channels} channels but got tensor with {x.shape[1]} channels"
                )

        # Reverse gradients before the spatial encoder so the backbone learns
        # domain-invariant latent representations.
        x = self.grl(x)
        x = self.conv_layers(x)  # (B, C_hidden, H', W')
        x = self.global_pool(x)  # (B, C_hidden, 1, 1)
        x = torch.flatten(x, 1)  # (B, C_hidden)
        return self.fc(x)  # (B, C_logit)

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> Any:
        r"""
        Load classifier state while preserving compatibility with older fixtures.

        Parameters
        ----------
        state_dict : dict[str, Any]
            Serialized state dictionary for the classifier.
        strict : bool, optional, default=True
            Whether to require an exact state-dict match.

        Returns
        -------
        Any
            Result returned by ``torch.nn.Module.load_state_dict``.
        """
        upgraded_state_dict = dict(state_dict)
        upgraded_state_dict.setdefault(
            "grl.device_buffer",
            self.grl.device_buffer.detach().clone(),
        )
        return super().load_state_dict(upgraded_state_dict, strict=strict)
