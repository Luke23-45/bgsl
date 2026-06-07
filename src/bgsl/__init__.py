"""
bgsl/__init__.py
----------------
Public API for the BGSL (Biological Gradient Supervised Learning) package.

Method:
    BGSL supervises not only the risk level at each time step, but also its
    first and second temporal derivatives, using soft event-onset targets
    derived from the known event time t*.

    L_BGSL = L_state + λ_v · L_velocity + λ_a · L_acceleration

Primary application: Sepsis early warning in ICU time series.
General applicability: Any timestamped event-risk prediction task.
"""

from bgsl.core.sepsis.targets import SoftOnsetTarget
from bgsl.core.common.targets import BaseSoftOnsetTarget
from bgsl.models.hypernetwork import HyperNetwork
from bgsl.core.common.losses import (
    BGSLLoss,
    BCELoss,
    TLSLoss,
    FocalLoss,
    SmoothnessLoss,
    TotalVariationLoss,
    WeightedBCELoss,
)
from bgsl.core.sepsis.metrics import SepsisMetrics as BGSLMetrics
from bgsl.core.sepsis.calibration import SepsisCalibration as TrajectoryCalibration

__version__ = "0.1.0"

__all__ = [
    # Core method
    "SoftOnsetTarget",
    "BaseSoftOnsetTarget",
    "HyperNetwork",
    "BGSLLoss",
    # Baselines
    "BCELoss",
    "TLSLoss",
    "FocalLoss",
    "SmoothnessLoss",
    "TotalVariationLoss",
    "WeightedBCELoss",
    # Evaluation
    "BGSLMetrics",
    "TrajectoryCalibration",
]
