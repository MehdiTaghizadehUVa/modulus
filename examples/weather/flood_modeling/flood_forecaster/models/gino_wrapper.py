r"""FloodForecaster GINO wrapper and importable torch adapter."""

from collections import OrderedDict
import importlib
import inspect
import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig, ListConfig, OmegaConf

import physicsnemo
from physicsnemo.core.module import ModelMetaData

from models.types import (
    AnyFloatTensor,
    FeatureTensor,
    GINOForwardReturn,
    GeometryInput,
    LatentQueryInput,
    ModelOutputTensor,
    OutputQueryContainer,
)


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


_IMPORTABLE_MARKER = "__flood_forecaster_importable__"
_TUPLE_MARKER = "__flood_forecaster_tuple__"


def _resolve_importable(module_name: str, qualname: str) -> Any:
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _serialize_init_value(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {_TUPLE_MARKER: [_serialize_init_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_init_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_init_value(item) for key, item in value.items()}

    module_name = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
    if module_name is not None and qualname is not None:
        return {_IMPORTABLE_MARKER: {"module": module_name, "qualname": qualname}}

    if _is_json_serializable(value):
        return value

    raise ValueError(f"Cannot serialize constructor value of type {type(value).__name__}")


def _deserialize_init_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_init_value(item) for item in value]
    if isinstance(value, dict):
        if _IMPORTABLE_MARKER in value:
            payload = value[_IMPORTABLE_MARKER]
            return _resolve_importable(payload["module"], payload["qualname"])
        if _TUPLE_MARKER in value:
            return tuple(_deserialize_init_value(item) for item in value[_TUPLE_MARKER])
        return {key: _deserialize_init_value(item) for key, item in value.items()}
    return value


def _extract_model_init_spec(model: nn.Module) -> tuple[list[Any], Dict[str, Any]]:
    if hasattr(model, "_init_kwargs") and isinstance(model._init_kwargs, dict):
        raw_init_kwargs = dict(model._init_kwargs)
        raw_args = raw_init_kwargs.pop("args", ())
        if not isinstance(raw_args, (list, tuple)):
            raise ValueError(
                f"Cannot adapt {model.__class__.__name__}: _init_kwargs['args'] must be a sequence"
            )
        module_args = [_serialize_init_value(value) for value in raw_args]
        module_kwargs = {
            key: _serialize_init_value(value)
            for key, value in raw_init_kwargs.items()
            if not str(key).startswith("_")
        }
        return module_args, module_kwargs

    signature = inspect.signature(model.__class__.__init__)
    module_kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "meta"}:
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"Cannot adapt {model.__class__.__name__}: variadic constructor args are unsupported"
            )
        if not hasattr(model, name):
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(
                    f"Cannot adapt {model.__class__.__name__}: missing constructor attribute '{name}'"
                )
            continue
        module_kwargs[name] = _serialize_init_value(getattr(model, name))
    return [], module_kwargs


class ImportableTorchModuleAdapter(physicsnemo.Module):
    r"""
    Importable PhysicsNeMo wrapper for nested torch modules.

    Parameters
    ----------
    module_module : str
        Fully qualified Python module path for the wrapped class.
    module_name : str
        Class name for the wrapped module inside ``module_module``.
    module_args : list[Any], optional, default=None
        Positional constructor arguments serialized for PhysicsNeMo checkpoints.
    module_kwargs : Dict[str, Any], optional, default=None
        Keyword constructor arguments serialized for PhysicsNeMo checkpoints.
    meta : ModelMetaData, optional, default=None
        PhysicsNeMo metadata attached to the wrapper module.

    Forward
    -------
    *args : Any
        Positional arguments forwarded unchanged to the wrapped module.
    **kwargs : Any
        Keyword arguments forwarded unchanged to the wrapped module.

    Outputs
    -------
    Any
        The exact return value produced by the wrapped module.
    """

    def __init__(
        self,
        module_module: str,
        module_name: str,
        module_args: Optional[list[Any]] = None,
        module_kwargs: Optional[Dict[str, Any]] = None,
        meta: Optional[ModelMetaData] = None,
    ):
        super().__init__(meta=meta or ModelMetaData())
        self.module_module = module_module
        self.module_name = module_name
        self.module_args = module_args or []
        self.module_kwargs = module_kwargs or {}
        module_cls = getattr(importlib.import_module(module_module), module_name)
        resolved_args = [_deserialize_init_value(value) for value in self.module_args]
        resolved_kwargs = {
            key: _deserialize_init_value(value)
            for key, value in self.module_kwargs.items()
        }
        self.inner_model = module_cls(*resolved_args, **resolved_kwargs)

    @classmethod
    def from_existing(
        cls, model: nn.Module, meta: Optional[ModelMetaData] = None
    ) -> "ImportableTorchModuleAdapter":
        r"""
        Create a serializable adapter from an existing torch module.

        Parameters
        ----------
        model : nn.Module
            Torch module instance to wrap for PhysicsNeMo serialization.
        meta : ModelMetaData, optional, default=None
            PhysicsNeMo metadata to attach to the adapter.

        Returns
        -------
        ImportableTorchModuleAdapter
            Adapter initialized from the wrapped module constructor and state dict.
        """
        module_cls = model.__class__
        module_args, module_kwargs = _extract_model_init_spec(model)

        adapter = cls(
            module_module=module_cls.__module__,
            module_name=module_cls.__name__,
            module_args=module_args,
            module_kwargs=module_kwargs,
            meta=meta,
        )
        adapter.inner_model.load_state_dict(model.state_dict())
        return adapter

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Forward all inputs to the wrapped torch module.

        Parameters
        ----------
        *args : Any
            Positional arguments accepted by the wrapped module.
        **kwargs : Any
            Keyword arguments accepted by the wrapped module.

        Returns
        -------
        Any
            Output returned by the wrapped module.
        """
        return self.inner_model(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            inner_model = self._modules.get("inner_model")
            if inner_model is not None and hasattr(inner_model, name):
                return getattr(inner_model, name)
            raise


class GINOModelMetaData(ModelMetaData):
    r"""Metadata describing FloodForecaster's registered GINO wrapper."""

    name: str = "FloodForecaster GINO Wrapper"
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    bf16: bool = True
    onnx: bool = False
    func_torch: bool = False
    auto_grad: bool = True


class GINOWrapper(physicsnemo.Module, register=True):
    r"""
    PhysicsNeMo wrapper around a GINO-style model.

    Parameters
    ----------
    model : nn.Module
        Configured GINO-style backend to wrap for PhysicsNeMo-native checkpointing.
    autoregressive : bool, optional, default=False
        Whether rollout mode should feed residual state updates into the output.
    residual_output : bool, optional, default=False
        Whether autoregressive mode adds the predicted increment to the last state.
    max_autoregressive_steps : int, optional, default=1
        Maximum number of autoregressive steps supported by the wrapper metadata.
    meta : ModelMetaData, optional, default=None
        PhysicsNeMo metadata attached to the registered wrapper.

    Forward
    -------
    input_geom : torch.Tensor
        Geometry tensor of shape :math:`(N, C_{coord})` or :math:`(1, N, C_{coord})`.
    latent_queries : torch.Tensor
        Latent query grid of shape :math:`(H, W, C_{coord})` or :math:`(1, H, W, C_{coord})`.
    output_queries : torch.Tensor or Dict[str, torch.Tensor]
        Output query tensor of shape :math:`(N_{q}, C_{coord})` or :math:`(1, N_{q}, C_{coord})`,
        or a non-empty ``dict[str, torch.Tensor]`` whose values each follow the same contract.
    x : torch.Tensor, optional
        Input feature tensor of shape :math:`(B, N, C_{in})` when latent features are not precomputed.
    latent_features : torch.Tensor, optional
        Precomputed latent tensor used to skip the encoder path.
    ada_in : torch.Tensor, optional
        Optional domain-adaptation conditioning tensor forwarded to the latent embedding block.
    return_features : bool, optional, default=False
        If ``True``, return both decoded outputs and the channel-first latent
        feature grid used by the domain-adaptation classifier.

    Outputs
    -------
    torch.Tensor or Dict[str, torch.Tensor] or tuple
        Decoded output tensor of shape :math:`(B, N_{q}, C_{out})`, a matching
        ``dict[str, torch.Tensor]`` when query groups are provided, or a tuple of
        ``(outputs, latent_feature_grid)`` when ``return_features=True``. The
        feature grid has shape :math:`(B, C_{hidden}, H, W)`.
    """

    def __init__(
        self,
        model: nn.Module,
        autoregressive: bool = False,
        residual_output: bool = False,
        max_autoregressive_steps: int = 1,
        meta: Optional[ModelMetaData] = None,
    ):
        super().__init__(meta=meta or GINOModelMetaData())
        if isinstance(model, physicsnemo.Module):
            self.model = model
        else:
            self.model = ImportableTorchModuleAdapter.from_existing(model)
        if hasattr(self, "_args") and "__args__" in self._args:
            self._args["__args__"]["model"] = self.model
        self.autoregressive = autoregressive
        self.residual_output = residual_output
        self.max_autoregressive_steps = max_autoregressive_steps

    @staticmethod
    def _validate_query_tensor(
        query_tensor: torch.Tensor,
        coord_dim: int,
        query_name: str,
    ) -> None:
        if not isinstance(query_tensor, torch.Tensor):
            raise TypeError(f"{query_name} must be a torch.Tensor")
        if query_tensor.ndim not in (2, 3):
            raise ValueError(
                f"{query_name} must have rank 2 or 3, got shape {tuple(query_tensor.shape)}"
            )
        if query_tensor.ndim == 3 and query_tensor.shape[0] != 1:
            raise ValueError(
                f"{query_name} may only use a singleton batch dimension, got shape {tuple(query_tensor.shape)}"
            )
        if query_tensor.shape[-1] != coord_dim:
            raise ValueError(
                f"{query_name} expected trailing coordinate dimension {coord_dim}, "
                f"got {query_tensor.shape[-1]}"
            )

    @classmethod
    def _validate_output_queries(
        cls,
        output_queries: OutputQueryContainer,
        input_geom: GeometryInput,
        latent_queries: LatentQueryInput,
    ) -> None:
        if not isinstance(input_geom, torch.Tensor):
            raise TypeError("input_geom must be a torch.Tensor")
        if not isinstance(latent_queries, torch.Tensor):
            raise TypeError("latent_queries must be a torch.Tensor")

        if input_geom.ndim not in (2, 3):
            raise ValueError(
                f"input_geom must have rank 2 or 3, got shape {tuple(input_geom.shape)}"
            )
        if input_geom.ndim == 3 and input_geom.shape[0] != 1:
            raise ValueError(
                f"input_geom may only use a singleton batch dimension, got shape {tuple(input_geom.shape)}"
            )
        if latent_queries.ndim not in (3, 4):
            raise ValueError(
                f"latent_queries must have rank 3 or 4, got shape {tuple(latent_queries.shape)}"
            )
        if latent_queries.ndim == 4 and latent_queries.shape[0] != 1:
            raise ValueError(
                "latent_queries may only use a singleton batch dimension, "
                f"got shape {tuple(latent_queries.shape)}"
            )

        coord_dim = input_geom.shape[-1]
        if latent_queries.shape[-1] != coord_dim:
            raise ValueError(
                "latent_queries trailing coordinate dimension must match input_geom, "
                f"got {latent_queries.shape[-1]} and {coord_dim}"
            )

        if isinstance(output_queries, torch.Tensor):
            cls._validate_query_tensor(output_queries, coord_dim, "output_queries")
            return

        if not isinstance(output_queries, dict):
            raise TypeError(
                "output_queries must be either a torch.Tensor or a non-empty dict[str, torch.Tensor]"
            )
        if not output_queries:
            raise ValueError("output_queries dict must not be empty")

        expected_rank = None
        for key, value in output_queries.items():
            if not isinstance(key, str):
                raise TypeError("output_queries dict keys must be strings")
            cls._validate_query_tensor(value, coord_dim, f"output_queries['{key}']")
            if expected_rank is None:
                expected_rank = value.ndim
            elif value.ndim != expected_rank:
                raise ValueError(
                    "All output_queries dict values must use the same batch style "
                    f"(mixed ranks {expected_rank} and {value.ndim} found)"
                )

    def forward(
        self,
        input_geom: GeometryInput,
        latent_queries: LatentQueryInput,
        output_queries: OutputQueryContainer,
        x: Optional[FeatureTensor] = None,
        latent_features: Optional[AnyFloatTensor] = None,
        ada_in: Optional[AnyFloatTensor] = None,
        return_features: bool = False,
    ) -> GINOForwardReturn:
        r"""
        Run the wrapped GINO encoder-decoder on geometry-conditioned flood states.

        Parameters
        ----------
        input_geom : GeometryInput
            Geometry tensor of shape :math:`(N, C_{coord})` or :math:`(1, N, C_{coord})`.
        latent_queries : LatentQueryInput
            Latent query grid of shape :math:`(H, W, C_{coord})` or :math:`(1, H, W, C_{coord})`.
        output_queries : OutputQueryContainer
            Tensor or grouped tensors describing the output query coordinates.
        x : FeatureTensor, optional, default=None
            Input flood-state features of shape :math:`(B, N, C_{in})`.
        latent_features : AnyFloatTensor, optional, default=None
            Optional precomputed latent representation.
        ada_in : AnyFloatTensor, optional, default=None
            Optional adaptation-conditioning tensor.
        return_features : bool, optional, default=False
            If ``True``, return the 4D channel-first latent feature grid alongside
            decoded outputs.

        Returns
        -------
        GINOForwardReturn
            Decoded output tensor or tensor dictionary, optionally paired with latent features.
        """
        if not torch.compiler.is_compiling():
            if not isinstance(input_geom, torch.Tensor):
                raise TypeError("input_geom must be a torch.Tensor")
            if not isinstance(latent_queries, torch.Tensor):
                raise TypeError("latent_queries must be a torch.Tensor")
            if x is not None and not isinstance(x, torch.Tensor):
                raise TypeError("x must be a torch.Tensor when provided")
            if latent_features is None and x is None:
                raise ValueError("x is required when latent_features is not provided")
            self._validate_output_queries(output_queries, input_geom, latent_queries)

        gino_model = self.model

        if len(input_geom.shape) == 3:
            input_geom = input_geom[0]
        if isinstance(output_queries, torch.Tensor) and len(output_queries.shape) == 3:
            output_queries = output_queries[0]
        if len(latent_queries.shape) == 4:
            latent_queries = latent_queries[0]

        batch_size = x.shape[0] if x is not None else latent_features.shape[0]
        latent_points = latent_queries.reshape(-1, latent_queries.shape[-1])
        using_precomputed_latent = latent_features is not None

        if latent_features is None:
            in_p = gino_model.gno_in(
                y=input_geom,
                x=latent_points,
                f_y=x,
            )
            in_p = in_p.reshape((batch_size, *latent_queries.shape[:-1], -1))
            latent_feature_grid = gino_model.latent_embedding(in_p=in_p, ada_in=ada_in)
        else:
            latent_feature_grid = latent_features

        if latent_feature_grid.ndim == 3:
            expected_points = 1
            for dim in latent_queries.shape[:-1]:
                expected_points *= int(dim)
            if (
                latent_feature_grid.shape[1] != expected_points
                or latent_feature_grid.shape[2] != gino_model.fno_hidden_channels
            ):
                raise ValueError(
                    "Flattened latent_features must have shape "
                    f"(B, {expected_points}, {gino_model.fno_hidden_channels}), "
                    f"got {tuple(latent_feature_grid.shape)}"
                )
            latent_feature_grid = latent_feature_grid.reshape(
                batch_size,
                *latent_queries.shape[:-1],
                gino_model.fno_hidden_channels,
            ).permute(0, 3, 1, 2).contiguous()
        elif latent_feature_grid.ndim == 4:
            expected_spatial_shape = tuple(int(dim) for dim in latent_queries.shape[:-1])
            has_unexpected_grid_shape = (
                latent_feature_grid.shape[0] != batch_size
                or latent_feature_grid.shape[1] != gino_model.fno_hidden_channels
                or tuple(latent_feature_grid.shape[2:]) != expected_spatial_shape
            )
            if (using_precomputed_latent or return_features) and has_unexpected_grid_shape:
                raise ValueError(
                    "4D latent_features must have shape "
                    f"(B, {gino_model.fno_hidden_channels}, *{expected_spatial_shape}), "
                    f"got {tuple(latent_feature_grid.shape)}"
                )
        else:
            raise ValueError(
                "latent_features must be a 4D feature grid (B, C, H, W) or a "
                f"flattened 3D tensor (B, H*W, C), got shape {tuple(latent_feature_grid.shape)}"
            )

        if getattr(gino_model, "out_gno_tanh", None) in ["latent_embed", "both"]:
            latent_feature_grid = torch.tanh(latent_feature_grid)

        latent_embed = latent_feature_grid.permute(
            0, *gino_model.in_coord_dim_reverse_order, 1
        ).reshape(batch_size, -1, gino_model.fno_hidden_channels)

        gno_out_forward = getattr(gino_model.gno_out, "forward", gino_model.gno_out)
        gno_out_parameters = inspect.signature(gno_out_forward).parameters
        uses_legacy_gno_out = "f_x" in gno_out_parameters

        def _decode_queries(query_points: torch.Tensor) -> ModelOutputTensor:
            if len(query_points.shape) == 3:
                query_points = query_points.squeeze(0)

            if uses_legacy_gno_out:
                out = gino_model.gno_out(
                    y=query_points,
                    x=latent_points,
                    f_y=None,
                    f_x=latent_embed,
                    reduction="sum",
                )
            else:
                out = gino_model.gno_out(
                    y=latent_points,
                    x=query_points,
                    f_y=latent_embed,
                )
            out = out.permute(0, 2, 1)
            out = gino_model.projection(out).permute(0, 2, 1)
            if self.autoregressive and self.residual_output and x is not None:
                out = out + x[:, :, -3:]
            return out

        if isinstance(output_queries, dict):
            outputs = {key: _decode_queries(value) for key, value in output_queries.items()}
        else:
            outputs = _decode_queries(output_queries)

        if return_features:
            return outputs, latent_feature_grid
        return outputs

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            if hasattr(self.model, "device"):
                return self.model.device
            return torch.device("cpu")

    def __getattr__(self, name):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            model = self._modules.get("model")
            if model is not None and hasattr(model, name):
                return getattr(model, name)
            raise

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> Any:
        r"""
        Load checkpoints from both native wrapper keys and legacy ``gino.`` keys.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            Serialized state dictionary to load into the wrapper.
        strict : bool, optional, default=True
            Whether PyTorch should require an exact state-dict key match.

        Returns
        -------
        Any
            Result returned by ``torch.nn.Module.load_state_dict``.
        """
        new_state_dict = OrderedDict()
        metadata = getattr(state_dict, "_metadata", None)
        # Preserve compatibility with older checkpoints that stored the wrapped
        # neural operator under the legacy ``gino.`` prefix.
        for key, value in state_dict.items():
            if key == "_metadata":
                continue
            if key.startswith("gino."):
                new_state_dict["model." + key[5:]] = value
            else:
                new_state_dict[key] = value
        if metadata is not None:
            new_state_dict._metadata = metadata
        return super().load_state_dict(new_state_dict, strict=strict)
