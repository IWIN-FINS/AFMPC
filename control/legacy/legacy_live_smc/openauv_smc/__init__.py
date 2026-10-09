"""Minimal two-DOF controller and model for OpenAUV-style experiments."""

from .controller import (
    AxisControlConfig,
    AxisControlOutput,
    CascadeSMCAxis,
    OpenAUV2DOFController,
    TwoDOFControlOutput,
    build_default_controller,
    saturation,
    wrap_angle,
)
from .model import (
    AxisDynamics,
    OpenAUV2DOFModel,
    OpenAUVState,
    build_openauv_model,
)
from .finesub_interface import FineSUBCommandMapper, TelemetryStateEstimator
from .vision import (
    BBoxTargetEstimator,
    VisualControlReference,
    VisualObservation,
    VisionConfig,
)

__all__ = [
    "AxisControlConfig",
    "AxisControlOutput",
    "AxisDynamics",
    "BBoxTargetEstimator",
    "CascadeSMCAxis",
    "FineSUBCommandMapper",
    "OpenAUV2DOFController",
    "OpenAUV2DOFModel",
    "OpenAUVState",
    "TelemetryStateEstimator",
    "TwoDOFControlOutput",
    "VisualControlReference",
    "VisualObservation",
    "VisionConfig",
    "build_default_controller",
    "build_openauv_model",
    "saturation",
    "wrap_angle",
]
