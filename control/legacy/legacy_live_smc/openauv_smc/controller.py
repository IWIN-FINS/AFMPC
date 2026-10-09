"""Cascaded position/velocity saturated sliding-mode controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .model import AxisDynamics, OpenAUV2DOFModel, OpenAUVState


def saturation(value: float) -> float:
    """Continuous unit saturation used instead of sign(value)."""

    return max(-1.0, min(1.0, value))


def wrap_angle(angle: float) -> float:
    """Map an angle to [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class AxisControlConfig:
    """Tuning values for one outer-P / inner-SMC cascade."""

    outer_gain: float
    rate_limit: float
    rate_filter_tau: float
    reaching_gain: float
    robust_gain: float
    boundary_layer: float
    input_limit: float

    def __post_init__(self) -> None:
        positive = {
            "outer_gain": self.outer_gain,
            "rate_limit": self.rate_limit,
            "rate_filter_tau": self.rate_filter_tau,
            "reaching_gain": self.reaching_gain,
            "robust_gain": self.robust_gain,
            "boundary_layer": self.boundary_layer,
            "input_limit": self.input_limit,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"control parameters must be positive: {invalid}")


@dataclass(frozen=True)
class AxisControlOutput:
    control_input: float
    unsaturated_input: float
    position_error: float
    rate_command: float
    rate_reference: float
    rate_reference_dot: float
    sliding_variable: float
    input_saturated: bool


class CascadeSMCAxis:
    """Outer proportional position loop and inner saturated SMC rate loop."""

    def __init__(
        self,
        dynamics: AxisDynamics,
        config: AxisControlConfig,
        *,
        angular_position: bool = False,
    ) -> None:
        self.dynamics = dynamics
        self.config = config
        self.angular_position = angular_position
        self.rate_reference = 0.0

    def reset(self, rate_reference: float = 0.0) -> None:
        self.rate_reference = float(rate_reference)

    def compute(
        self,
        position: float,
        rate: float,
        position_reference: float,
        dt: float,
    ) -> AxisControlOutput:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        error = position_reference - position
        if self.angular_position:
            error = wrap_angle(error)

        rate_command = max(
            -self.config.rate_limit,
            min(self.config.rate_limit, self.config.outer_gain * error),
        )

        previous_reference = self.rate_reference
        alpha = 1.0 - math.exp(-dt / self.config.rate_filter_tau)
        self.rate_reference += alpha * (rate_command - self.rate_reference)
        rate_reference_dot = (self.rate_reference - previous_reference) / dt

        sliding_variable = self.rate_reference - rate
        damping_compensation = self.dynamics.damping_force(rate)
        switching = self.config.robust_gain * saturation(
            sliding_variable / self.config.boundary_layer
        )
        desired_acceleration = (
            rate_reference_dot
            + self.config.reaching_gain * sliding_variable
            + switching
        )
        unsaturated_input = (
            damping_compensation + self.dynamics.inertia * desired_acceleration
        )
        control_input = max(
            -self.config.input_limit,
            min(self.config.input_limit, unsaturated_input),
        )

        return AxisControlOutput(
            control_input=control_input,
            unsaturated_input=unsaturated_input,
            position_error=error,
            rate_command=rate_command,
            rate_reference=self.rate_reference,
            rate_reference_dot=rate_reference_dot,
            sliding_variable=sliding_variable,
            input_saturated=not math.isclose(control_input, unsaturated_input),
        )


@dataclass(frozen=True)
class TwoDOFControlOutput:
    heave_force: float
    yaw_moment: float
    heave: AxisControlOutput
    yaw: AxisControlOutput

    def generalized_force_6dof(self) -> tuple[float, float, float, float, float, float]:
        """Return [Fx, Fy, Fz, K, M, N] in FRD body coordinates."""

        return (0.0, 0.0, self.heave_force, 0.0, 0.0, self.yaw_moment)


class OpenAUV2DOFController:
    def __init__(self, heave: CascadeSMCAxis, yaw: CascadeSMCAxis) -> None:
        self.heave = heave
        self.yaw = yaw

    def reset(self) -> None:
        self.heave.reset()
        self.yaw.reset()

    def compute(
        self,
        state: OpenAUVState,
        depth_reference: float,
        yaw_reference: float,
        dt: float,
    ) -> TwoDOFControlOutput:
        heave = self.heave.compute(
            state.depth,
            state.heave_velocity,
            depth_reference,
            dt,
        )
        yaw = self.yaw.compute(
            state.yaw,
            state.yaw_rate,
            yaw_reference,
            dt,
        )
        return TwoDOFControlOutput(
            heave_force=heave.control_input,
            yaw_moment=yaw.control_input,
            heave=heave,
            yaw=yaw,
        )


def build_default_controller(
    model: OpenAUV2DOFModel,
    *,
    heave_input_limit: float = 80.0,
    yaw_input_limit: float = 12.0,
) -> OpenAUV2DOFController:
    """Create conservative gains for the published nominal model.

    The larger defaults are used by the standalone simulation.  A live caller
    should pass the calibrated FineSUB wrench limits so the SMC saturates at
    the same physical boundary as the lower-controller command adapter.
    """

    heave_config = AxisControlConfig(
        outer_gain=0.80,
        rate_limit=0.35,
        rate_filter_tau=0.35,
        reaching_gain=2.20,
        robust_gain=0.55,
        boundary_layer=0.04,
        input_limit=heave_input_limit,
    )
    yaw_config = AxisControlConfig(
        outer_gain=1.20,
        rate_limit=math.radians(35.0),
        rate_filter_tau=0.25,
        reaching_gain=3.00,
        robust_gain=0.90,
        boundary_layer=math.radians(2.0),
        input_limit=yaw_input_limit,
    )
    return OpenAUV2DOFController(
        heave=CascadeSMCAxis(model.heave, heave_config),
        yaw=CascadeSMCAxis(model.yaw, yaw_config, angular_position=True),
    )
