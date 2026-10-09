"""Two-degree-of-freedom OpenAUV model used by the reproducible demo.

The model intentionally decouples heave and yaw:

    m_eff * velocity_dot + d_linear * velocity
        + d_quadratic * abs(velocity) * velocity = input + disturbance

The published OpenAUV added-mass and damping values are retained. The rigid-body
yaw inertia is only a transparent box approximation because the paper does not
publish the measured inertia tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _wrap_angle(angle: float) -> float:
    """Map an angle to [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class AxisDynamics:
    """Parameters for one decoupled marine-vehicle axis."""

    inertia: float
    linear_damping: float
    quadratic_damping: float

    def __post_init__(self) -> None:
        if self.inertia <= 0.0:
            raise ValueError("inertia must be positive")
        if self.linear_damping < 0.0 or self.quadratic_damping < 0.0:
            raise ValueError("damping coefficients must be non-negative")

    def damping_force(self, velocity: float) -> float:
        return (
            self.linear_damping * velocity
            + self.quadratic_damping * abs(velocity) * velocity
        )

    def acceleration(
        self,
        velocity: float,
        control_input: float,
        disturbance: float = 0.0,
    ) -> float:
        return (
            control_input + disturbance - self.damping_force(velocity)
        ) / self.inertia


@dataclass(frozen=True)
class OpenAUVState:
    """State in FRD convention: depth down and positive yaw to the right."""

    depth: float = 0.0
    heave_velocity: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class OpenAUV2DOFModel:
    heave: AxisDynamics
    yaw: AxisDynamics

    def derivative(
        self,
        state: OpenAUVState,
        heave_force: float,
        yaw_moment: float,
        heave_disturbance: float = 0.0,
        yaw_disturbance: float = 0.0,
    ) -> OpenAUVState:
        return OpenAUVState(
            depth=state.heave_velocity,
            heave_velocity=self.heave.acceleration(
                state.heave_velocity, heave_force, heave_disturbance
            ),
            yaw=state.yaw_rate,
            yaw_rate=self.yaw.acceleration(
                state.yaw_rate, yaw_moment, yaw_disturbance
            ),
        )

    def step_rk4(
        self,
        state: OpenAUVState,
        heave_force: float,
        yaw_moment: float,
        dt: float,
        heave_disturbance: float = 0.0,
        yaw_disturbance: float = 0.0,
    ) -> OpenAUVState:
        """Advance the state with a fixed-input fourth-order Runge-Kutta step."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")

        def deriv(x: OpenAUVState) -> OpenAUVState:
            return self.derivative(
                x,
                heave_force,
                yaw_moment,
                heave_disturbance,
                yaw_disturbance,
            )

        def add_scaled(
            x: OpenAUVState, dx: OpenAUVState, scale: float
        ) -> OpenAUVState:
            return OpenAUVState(
                depth=x.depth + scale * dx.depth,
                heave_velocity=x.heave_velocity + scale * dx.heave_velocity,
                yaw=x.yaw + scale * dx.yaw,
                yaw_rate=x.yaw_rate + scale * dx.yaw_rate,
            )

        k1 = deriv(state)
        k2 = deriv(add_scaled(state, k1, 0.5 * dt))
        k3 = deriv(add_scaled(state, k2, 0.5 * dt))
        k4 = deriv(add_scaled(state, k3, dt))

        next_state = OpenAUVState(
            depth=state.depth
            + dt * (k1.depth + 2.0 * k2.depth + 2.0 * k3.depth + k4.depth) / 6.0,
            heave_velocity=state.heave_velocity
            + dt
            * (
                k1.heave_velocity
                + 2.0 * k2.heave_velocity
                + 2.0 * k3.heave_velocity
                + k4.heave_velocity
            )
            / 6.0,
            yaw=state.yaw
            + dt * (k1.yaw + 2.0 * k2.yaw + 2.0 * k3.yaw + k4.yaw) / 6.0,
            yaw_rate=state.yaw_rate
            + dt
            * (k1.yaw_rate + 2.0 * k2.yaw_rate + 2.0 * k3.yaw_rate + k4.yaw_rate)
            / 6.0,
        )
        return OpenAUVState(
            depth=next_state.depth,
            heave_velocity=next_state.heave_velocity,
            yaw=_wrap_angle(next_state.yaw),
            yaw_rate=next_state.yaw_rate,
        )


def build_openauv_model() -> OpenAUV2DOFModel:
    """Build the nominal model from the 2025 OpenAUV paper.

    Published values:
      mass = 18.5 kg, body length = 0.920 m, body width = 0.360 m
      added mass in heave = 16.4712 kg
      added yaw inertia = 0.8198 kg m^2
      heave damping = (6.2491, 51.0402)
      yaw damping = (0.0698, 1.1343)
    """

    rigid_mass = 18.5
    length = 0.920
    width = 0.360
    rigid_yaw_inertia = rigid_mass * (length**2 + width**2) / 12.0

    return OpenAUV2DOFModel(
        heave=AxisDynamics(
            inertia=rigid_mass + 16.4712,
            linear_damping=6.2491,
            quadratic_damping=51.0402,
        ),
        yaw=AxisDynamics(
            inertia=rigid_yaw_inertia + 0.8198,
            linear_damping=0.0698,
            quadratic_damping=1.1343,
        ),
    )

