"""Dynamic optimal-transport teachers used only during OODKA training."""

from .cost import OTCostBuilder, pool_feature_map
from .losses import WeightedCosineDistillation, WeightedLogRMSAlignment
from .mass import ResidualMassBuilder, StructureMassBuilder
from .objective import MultiScaleOTDistillation
from .sinkhorn import (
    BalancedSinkhorn,
    CapacityConstrainedPartialSinkhorn,
    UnbalancedSinkhorn,
)
from .transport import BarycentricProjector

__all__ = [
    "BalancedSinkhorn",
    "BarycentricProjector",
    "CapacityConstrainedPartialSinkhorn",
    "OTCostBuilder",
    "MultiScaleOTDistillation",
    "ResidualMassBuilder",
    "StructureMassBuilder",
    "UnbalancedSinkhorn",
    "WeightedCosineDistillation",
    "WeightedLogRMSAlignment",
    "pool_feature_map",
]
