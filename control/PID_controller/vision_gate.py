"""Gating and hold-last-valid handling for camera positions used by PID.

The MPC runtime does not feed every finite camera result to its controller: it
checks range, freshness and physically plausible motion first.  The PID path
keeps the same range and telemetry safety checks, but an isolated visual jump
is treated as a rejected measurement: while locked, the last valid camera
position is held and returned to the controller.  This avoids turning one bad
detector frame into an abrupt motor disarm or a direction reversal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


def _finite_vector3(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result.copy()


@dataclass(frozen=True)
class VisionGateConfig:
    """Conservative camera-coordinate limits for a real-vehicle PID test."""

    min_forward_m: float = 0.15
    max_forward_m: float = 2.50
    max_speed_m_s: float = 1.0
    jump_margin_m: float = 0.10
    max_step_m: float = 0.30
    max_inter_sample_gap_s: float = 0.50
    startup_confirmation_samples: int = 3
    reacquire_confirmation_samples: int = 3

    def __post_init__(self) -> None:
        positive = (
            self.min_forward_m,
            self.max_forward_m,
            self.max_speed_m_s,
            self.jump_margin_m,
            self.max_step_m,
            self.max_inter_sample_gap_s,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise ValueError("vision gate limits must be finite and positive")
        if self.max_forward_m <= self.min_forward_m:
            raise ValueError("max_forward_m must exceed min_forward_m")
        if self.max_step_m < self.jump_margin_m:
            raise ValueError("max_step_m must be at least jump_margin_m")
        if int(self.startup_confirmation_samples) < 1:
            raise ValueError("startup_confirmation_samples must be positive")
        if int(self.reacquire_confirmation_samples) < 1:
            raise ValueError("reacquire_confirmation_samples must be positive")


@dataclass(frozen=True)
class VisionGateResult:
    """One gate decision returned to the hardware session."""

    ready: bool
    reason: str
    position_camera_xyz: np.ndarray | None
    confirmation_count: int
    locked: bool
    lost: bool


class PIDVisionGate:
    """Stateful range/motion gate with startup confirmation and jump rejection.

    A large single-frame jump never reaches the PID.  Once a lock exists, the
    previous valid position is returned instead and the lock is retained.  If
    a new position repeats consistently, it is accepted in-place after the
    reacquisition confirmation count without disarming.  Invalid coordinates
    and gradual out-of-range forward distance remain fail-closed.
    """

    def __init__(self, config: VisionGateConfig | None = None) -> None:
        self.config = config or VisionGateConfig()
        self.reset()

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def last_position(self) -> np.ndarray | None:
        return None if self._last_position is None else self._last_position.copy()

    def reset(self, *, preserve_lock_history: bool = False) -> None:
        had_lock = self._has_ever_locked if hasattr(self, "_has_ever_locked") else False
        self._last_position: np.ndarray | None = None
        self._last_time: float | None = None
        self._pending_position: np.ndarray | None = None
        self._pending_time: float | None = None
        self._pending_count = 0
        self._jump_candidate_position: np.ndarray | None = None
        self._jump_candidate_time: float | None = None
        self._jump_candidate_count = 0
        self._locked = False
        self._has_ever_locked = bool(preserve_lock_history and had_lock)

    def _clear_jump_candidate(self) -> None:
        self._jump_candidate_position = None
        self._jump_candidate_time = None
        self._jump_candidate_count = 0

    def _register_jump_candidate(self, position: np.ndarray, now: float) -> int:
        """Record a consistent jump candidate without feeding it to PID.

        The caller may invoke ``update`` more often than the camera publishes.
        Identical repeated values therefore do not count as extra confirmation
        samples; only a new, nearby measurement advances the candidate.
        """
        if (
            self._jump_candidate_position is None
            or self._jump_candidate_time is None
        ):
            self._jump_candidate_count = 1
        else:
            dt = now - self._jump_candidate_time
            distance = float(np.linalg.norm(position - self._jump_candidate_position))
            limit = min(
                self.config.max_step_m,
                self.config.jump_margin_m + self.config.max_speed_m_s * max(dt, 0.0),
            )
            if (
                not math.isfinite(dt)
                or dt <= 0.0
                or dt > self.config.max_inter_sample_gap_s
                or distance > limit
            ):
                self._jump_candidate_count = 1
            elif distance > 1e-6:
                self._jump_candidate_count += 1
        self._jump_candidate_position = position.copy()
        self._jump_candidate_time = now
        return self._jump_candidate_count

    def _hold_or_accept_jump_candidate(
        self,
        position: np.ndarray,
        now: float,
    ) -> VisionGateResult:
        """Hold the previous sample, or accept a stable new target in-place."""
        count = self._register_jump_candidate(position, now)
        # One-sample reacquisition after a genuine gap must not make the
        # first implausible jump acceptable while a lock still exists.
        required = max(2, int(self.config.reacquire_confirmation_samples))
        in_range = self.config.min_forward_m <= float(position[2]) <= self.config.max_forward_m
        if count >= required and in_range:
            self._last_position = position.copy()
            self._last_time = now
            self._locked = True
            self._clear_jump_candidate()
            return self._result(True, "vision_reacquired", position)

        held_position = self._last_position
        if held_position is None:
            # This helper is only used after a lock, but keep it fail-closed
            # for custom callers and future refactors.
            return self._result(False, "vision_pending", None)
        return self._result(True, "vision_jump_ignored", held_position)

    def _result(
        self,
        ready: bool,
        reason: str,
        position: np.ndarray | None,
        *,
        lost: bool = False,
    ) -> VisionGateResult:
        return VisionGateResult(
            ready=bool(ready),
            reason=str(reason),
            position_camera_xyz=None if position is None else position.copy(),
            confirmation_count=int(self._pending_count),
            locked=bool(self._locked),
            lost=bool(lost),
        )

    def hold_without_sample(self) -> VisionGateResult:
        """Keep a locked position between camera frames without confirming it twice.

        The hardware session separately enforces a maximum age for the last
        genuinely new frame. This method only preserves the current gate
        state for the faster 20 Hz command loop.
        """
        if self._locked and self._last_position is not None:
            return self._result(True, "vision_held_between_frames", self._last_position)
        return self._result(False, "vision_wait_new_frame", None)

    def _plausible_step(self, position: np.ndarray, now: float) -> bool:
        if self._last_position is None or self._last_time is None:
            return True
        dt = now - self._last_time
        if not math.isfinite(dt) or dt <= 0.0 or dt > self.config.max_inter_sample_gap_s:
            return False
        distance = float(np.linalg.norm(position - self._last_position))
        limit = min(
            self.config.max_step_m,
            self.config.jump_margin_m + self.config.max_speed_m_s * dt,
        )
        return distance <= limit

    def update(
        self,
        position_camera_xyz: object,
        *,
        now: float | None = None,
    ) -> VisionGateResult:
        """Validate one camera position and report whether it is control-ready."""

        try:
            position = _finite_vector3(position_camera_xyz, "position_camera_xyz")
        except (TypeError, ValueError):
            was_locked = self._locked
            self.reset()
            return self._result(False, "invalid_position", None, lost=was_locked)

        current_time = time.monotonic() if now is None else float(now)
        if not math.isfinite(current_time):
            raise ValueError("now must be finite")

        # A genuine camera gap ends the old lock. The next valid measurement
        # follows the configured reacquisition count, rather than being
        # misclassified as a jump from an obsolete position.
        if (
            self._last_time is not None
            and current_time - self._last_time > self.config.max_inter_sample_gap_s
        ):
            self.reset(preserve_lock_history=True)

        forward = float(position[2])
        plausible = self._plausible_step(position, current_time)

        # A one-frame range excursion is often the same detector jump seen in
        # the horizontal/depth coordinates.  When it is also physically
        # implausible relative to the last lock, hold the last valid sample;
        # a gradual, in-range-to-out-of-range departure remains fail-closed.
        if not self.config.min_forward_m <= forward <= self.config.max_forward_m:
            if self._locked and self._last_position is not None and not plausible:
                self._last_time = current_time
                self._pending_position = None
                self._pending_time = None
                self._pending_count = 0
                return self._hold_or_accept_jump_candidate(position, current_time)
            was_locked = self._locked
            self.reset()
            return self._result(False, "forward_range", None, lost=was_locked)

        if not plausible:
            # A locked controller must not turn one detector outlier into an
            # armed-session disarm.  Keep the last accepted sample and let the
            # hardware session continue the PID update against that held
            # position.  The new sample is deliberately not copied into any
            # pending/reacquisition state, so it cannot contaminate the next
            # valid measurement.
            if self._locked and self._last_position is not None:
                self._last_time = current_time
                self._pending_position = None
                self._pending_time = None
                self._pending_count = 0
                return self._hold_or_accept_jump_candidate(position, current_time)

            was_locked = self._locked
            self._pending_position = position.copy()
            self._pending_time = current_time
            self._pending_count = 1
            self._locked = False
            self._last_position = None
            self._last_time = None
            return self._result(
                False,
                "vision_jump" if was_locked else "vision_pending",
                None,
                lost=was_locked,
            )

        # A lock is retained across an ignored jump.  The first plausible
        # sample after it should resume tracking immediately; do not impose
        # the startup/reacquisition confirmation delay on an already locked
        # armed loop.
        if self._locked:
            self._clear_jump_candidate()
            self._last_position = position.copy()
            self._last_time = current_time
            self._pending_position = None
            self._pending_time = None
            self._pending_count = 0
            return self._result(True, "vision_tracking", position)

        if self._pending_position is None or self._pending_time is None:
            self._pending_position = position.copy()
            self._pending_time = current_time
            self._pending_count = 1
        else:
            pending_dt = current_time - self._pending_time
            pending_distance = float(np.linalg.norm(position - self._pending_position))
            pending_limit = min(
                self.config.max_step_m,
                self.config.jump_margin_m + self.config.max_speed_m_s * max(pending_dt, 0.0),
            )
            if (
                pending_dt <= 0.0
                or pending_dt > self.config.max_inter_sample_gap_s
                or pending_distance > pending_limit
            ):
                self._pending_position = position.copy()
                self._pending_time = current_time
                self._pending_count = 1
            else:
                self._pending_position = position.copy()
                self._pending_time = current_time
                self._pending_count += 1

        required = (
            self.config.reacquire_confirmation_samples
            if self._has_ever_locked
            else self.config.startup_confirmation_samples
        )
        if self._pending_count < required:
            return self._result(False, "vision_pending", None)

        first_lock = not self._has_ever_locked
        self._last_position = position.copy()
        self._last_time = current_time
        self._locked = True
        self._has_ever_locked = True
        self._pending_count = required
        return self._result(True, "vision_locked" if first_lock else "vision_tracking", position)
