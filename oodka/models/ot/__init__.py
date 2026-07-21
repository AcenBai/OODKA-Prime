"""Dynamic optimal-transport teachers used only during OODKA training."""

from .cost import OTCostBuilder, pool_feature_map
from .losses import WeightedCosineDistillation
from .mass import ResidualMassBuilder, StructureMassBuilder
from .objective import MultiScaleOTDistillation
from .sinkhorn import BalancedSinkhorn, UnbalancedSinkhorn
from .transport import BarycentricProjector

__all__ = [
    "BalancedSinkhorn",
    "BarycentricProjector",
    "OTCostBuilder",
    "MultiScaleOTDistillation",
    "ResidualMassBuilder",
    "StructureMassBuilder",
    "UnbalancedSinkhorn",
    "WeightedCosineDistillation",
    "pool_feature_map",
]
