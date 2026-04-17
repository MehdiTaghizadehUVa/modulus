r"""FloodForecaster example-local model package."""

from .domain_classifier import CNNDomainClassifier
from .gino_wrapper import GINOWrapper, ImportableTorchModuleAdapter

__all__ = [
    "CNNDomainClassifier",
    "GINOWrapper",
    "ImportableTorchModuleAdapter",
]
