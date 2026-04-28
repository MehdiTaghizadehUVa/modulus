r"""Typed tensor contracts for FloodForecaster model-facing APIs."""

from typing import Dict, TypedDict, Union

import torch
from jaxtyping import Float

AnyFloatTensor = Float[torch.Tensor, "*shape"]
GeometryTensor = Float[torch.Tensor, "points coord"]
BatchedGeometryTensor = Float[torch.Tensor, "one points coord"]
QueryGridTensor = Float[torch.Tensor, "height width coord"]
BatchedQueryGridTensor = Float[torch.Tensor, "one height width coord"]
FeatureTensor = Float[torch.Tensor, "batch points channels"]
LatentFeatureGridTensor = Float[torch.Tensor, "batch channels height width"]
ModelOutputTensor = Float[torch.Tensor, "batch points channels"]
ClassifierInputTensor = Float[torch.Tensor, "batch channels height width"]
ClassifierOutputTensor = Float[torch.Tensor, "batch logits"]

GeometryInput = Union[GeometryTensor, BatchedGeometryTensor]
LatentQueryInput = Union[QueryGridTensor, BatchedQueryGridTensor]
OutputQueryTensor = Union[GeometryTensor, BatchedGeometryTensor]
OutputQueryContainer = Union[OutputQueryTensor, Dict[str, OutputQueryTensor]]
ModelOutputContainer = Union[ModelOutputTensor, Dict[str, ModelOutputTensor]]
GINOForwardReturn = Union[
    ModelOutputContainer,
    tuple[ModelOutputContainer, LatentFeatureGridTensor],
]


class _RequiredRawFloodSample(TypedDict):
    geometry: AnyFloatTensor
    static: AnyFloatTensor
    boundary: AnyFloatTensor
    dynamic: AnyFloatTensor
    query_points: AnyFloatTensor


class RawFloodSample(_RequiredRawFloodSample, total=False):
    target: AnyFloatTensor
    cell_area: AnyFloatTensor
    run_id: str
    time_index: int


class _RequiredProcessedGINOBatch(TypedDict):
    input_geom: GeometryTensor
    latent_queries: QueryGridTensor
    output_queries: GeometryTensor
    x: FeatureTensor


class ProcessedGINOBatch(_RequiredProcessedGINOBatch, total=False):
    y: ModelOutputTensor
