from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceFusionConfig:
    enabled: bool = True
    buffer_size: int = 6
    fallback_dt_s: float = 0.1
    process_var_z: float = 0.02
    process_var_v: float = 0.5
    init_var_z: float = 0.04
    init_var_v: float = 1.0
    r_base: float = 0.04
    r_min: float = 0.05
    measurement_variance_mode: str = "sample_variance"
    effective_sample_divisor: float = 1.0
    measurement_var_min: float = 1.0e-6
    measurement_var_max: float = 4.0e-2
    n_ref: float = 48.0
    sigma_iqr: float = 0.04
    sep_ref: float = 2.0
    sigma_res: float = 0.25
    z_iqr_hard: float = 0.08
    r_hard_bad: float = 0.05
    update_gain_threshold: float = 0.65
    weak_gain_threshold: float = 0.15
    confidence_decay: float = 0.8
    nis_gate_enabled: bool = True
    nis_gate_threshold: float = 6.63
    hard_gate_enabled: bool = True
    hard_gate_history_size: int = 4
    hard_gate_min_history: int = 3
    hard_gate_pred_margin_m: float = 0.28
    hard_gate_hist_margin_m: float = 0.28
    hard_gate_rel_ratio: float = 0.35

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "SequenceFusionConfig":
        cfg = dict(cfg or {})
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            buffer_size=max(int(cfg.get("buffer_size", 6)), 2),
            fallback_dt_s=float(cfg.get("fallback_dt_s", 0.1)),
            process_var_z=float(cfg.get("process_var_z", 0.02)),
            process_var_v=float(cfg.get("process_var_v", cfg.get("process_var_z_dot", 0.5))),
            init_var_z=float(cfg.get("init_var_z", cfg.get("r_base", cfg.get("R_base", 0.04)))),
            init_var_v=float(cfg.get("init_var_v", 1.0)),
            r_base=float(cfg.get("R_base", cfg.get("r_base", 0.04))),
            r_min=float(cfg.get("r_min", 0.05)),
            measurement_variance_mode=str(
                cfg.get("measurement_variance_mode", "sample_variance")
            ).strip().lower(),
            effective_sample_divisor=max(
                float(cfg.get("effective_sample_divisor", 1.0)), 1.0
            ),
            measurement_var_min=max(
                float(cfg.get("measurement_var_min", 1.0e-6)), 1.0e-9
            ),
            measurement_var_max=max(
                float(cfg.get("measurement_var_max", 4.0e-2)), 1.0e-9
            ),
            n_ref=float(cfg.get("N_ref", cfg.get("n_ref", 48.0))),
            sigma_iqr=max(float(cfg.get("sigma_iqr", 0.04)), 1e-6),
            sep_ref=max(float(cfg.get("sep_ref", 2.0)), 1e-6),
            sigma_res=max(float(cfg.get("sigma_res", 0.25)), 1e-6),
            z_iqr_hard=max(float(cfg.get("z_iqr_hard", 0.08)), 1e-6),
            r_hard_bad=float(np.clip(cfg.get("r_hard_bad", 0.05), 0.0, 1.0)),
            update_gain_threshold=float(np.clip(cfg.get("update_gain_threshold", 0.65), 0.0, 1.0)),
            weak_gain_threshold=float(np.clip(cfg.get("weak_gain_threshold", 0.15), 0.0, 1.0)),
            confidence_decay=float(np.clip(cfg.get("confidence_decay", 0.8), 0.0, 1.0)),
            nis_gate_enabled=bool(cfg.get("nis_gate_enabled", True)),
            nis_gate_threshold=max(float(cfg.get("nis_gate_threshold", 6.63)), 0.0),
            hard_gate_enabled=bool(cfg.get("hard_gate_enabled", True)),
            hard_gate_history_size=max(int(cfg.get("hard_gate_history_size", 4)), 1),
            hard_gate_min_history=max(int(cfg.get("hard_gate_min_history", 3)), 1),
            hard_gate_pred_margin_m=max(float(cfg.get("hard_gate_pred_margin_m", 0.28)), 0.0),
            hard_gate_hist_margin_m=max(float(cfg.get("hard_gate_hist_margin_m", 0.28)), 0.0),
            hard_gate_rel_ratio=max(float(cfg.get("hard_gate_rel_ratio", 0.35)), 0.0),
        )


class SequenceConfidenceKalmanDepthFilter:
    """Constant-velocity Kalman filter with confidence-weighted depth updates."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = SequenceFusionConfig.from_dict(cfg)
        self.enabled = bool(self.cfg.enabled)
        self.initialized = False
        self.x = np.zeros(2, dtype=np.float64)
        self.P = np.diag([self.cfg.init_var_z, self.cfg.init_var_v]).astype(np.float64)
        self.confidence = 0.0
        self.last_valid = False
        self.last_rejected = False
        self.last_gain = 0.0
        self.last_quality = 0.0
        self.last_mode = "predict_only"
        self.last_r_t = 0.0
        self.last_R_t = float(self.cfg.r_base)
        self.last_z_pred = None
        self.last_z_ref = None
        self.last_v_ref = 0.0
        self.last_nis = None
        self.last_innovation = None
        self.last_S_t = None
        self.last_P_zz_pred = None
        self.last_dt_s = float(self.cfg.fallback_dt_s)
        self._time = 0.0
        self._history = deque(maxlen=max(int(self.cfg.buffer_size), 2))

    def predict_only(self, dt_s: float | None = None, **_: object):
        if not self.enabled:
            self.last_valid = False
            self.last_rejected = False
            self.last_mode = "predict_only"
            self.last_gain = 0.0
            self.last_quality = 0.0
            return

        dt = self._resolve_dt(dt_s)
        self.last_dt_s = float(dt)
        self._time += dt
        if not self.initialized:
            self.confidence *= self.cfg.confidence_decay
            self.last_valid = False
            self.last_rejected = False
            self.last_gain = 0.0
            self.last_quality = 0.0
            self.last_mode = "predict_only"
            self.last_r_t = 0.0
            self.last_R_t = float(self.cfg.r_base)
            return

        x_pred, p_pred = self._predict_state(dt)
        self.x = x_pred
        self.P = p_pred
        self.confidence *= self.cfg.confidence_decay
        self.last_valid = False
        self.last_rejected = False
        self.last_gain = 0.0
        self.last_quality = 0.0
        self.last_mode = "predict_only"
        self.last_r_t = 0.0
        self.last_R_t = float(self.cfg.r_base)
        self.last_z_pred = float(self.x[0])
        self.last_z_ref = float(self.x[0])
        self.last_v_ref = float(self.x[1])
        self._append_history(
            z_meas=None,
            z_out=float(self.x[0]),
            core_px=0,
            z_iqr=np.nan,
            sep_score=np.nan,
            r_t=0.0,
            gain=0.0,
            has_measurement=False,
        )

    def update(self, z_raw: float | None, depth_stats=None,
               depth_confidence: float | None = None,
               dt_s: float | None = None,
               **_: object) -> dict:
        if not self.enabled:
            return self._disabled_state(z_raw, depth_confidence)

        dt = self._resolve_dt(dt_s)
        if z_raw is None or not np.isfinite(z_raw):
            self.predict_only(dt)
            return self.state()

        self._time += dt
        z_raw = float(z_raw)
        if not self.initialized:
            r_t, diag = self._quality_from_stats(
                z_raw,
                depth_stats=depth_stats,
                z_pred=z_raw,
                depth_confidence=depth_confidence,
            )
            r_t_meas = self._measurement_variance_from_stats(
                depth_stats,
                fallback_confidence=depth_confidence,
                quality=r_t,
            )
            self.x[:] = [z_raw, 0.0]
            self.P = np.diag([
                max(self.cfg.init_var_z, r_t_meas),
                self.cfg.init_var_v,
            ]).astype(np.float64)
            self.initialized = True
            self.confidence = float(r_t)
            self.last_valid = True
            self.last_rejected = False
            self.last_gain = 1.0
            self.last_quality = float(r_t)
            self.last_mode = "update"
            self.last_r_t = float(r_t)
            self.last_R_t = float(r_t_meas)
            self.last_z_pred = z_raw
            self.last_z_ref = z_raw
            self.last_v_ref = 0.0
            self.last_nis = 0.0
            self.last_innovation = 0.0
            self.last_S_t = float(r_t_meas)
            self.last_P_zz_pred = float(self.P[0, 0])
            self._append_history(
                z_meas=z_raw,
                z_out=z_raw,
                core_px=int(diag["core_px"]),
                z_iqr=float(diag["z_iqr"]),
                sep_score=float(diag["sep_score"]),
                r_t=float(r_t),
                gain=1.0,
                has_measurement=True,
            )
            return self.state()

        x_pred, p_pred = self._predict_state(dt)
        return self._correct_from_prediction(
            x_pred,
            p_pred,
            z_raw=z_raw,
            depth_stats=depth_stats,
            depth_confidence=depth_confidence,
            overwrite_last=False,
        )

    def correct_only(self, z_raw: float | None, depth_stats=None,
                     depth_confidence: float | None = None,
                     dt_s: float | None = None,
                     **_: object) -> dict:
        if not self.enabled:
            return self._disabled_state(z_raw, depth_confidence)
        if dt_s is not None and np.isfinite(dt_s) and dt_s > 0:
            self.last_dt_s = float(dt_s)
        if z_raw is None or not np.isfinite(z_raw):
            return self.state()
        if not self.initialized:
            return self.update(
                z_raw,
                depth_stats=depth_stats,
                depth_confidence=depth_confidence,
                dt_s=dt_s,
            )
        return self._correct_from_prediction(
            self.x.copy(),
            self.P.copy(),
            z_raw=float(z_raw),
            depth_stats=depth_stats,
            depth_confidence=depth_confidence,
            overwrite_last=True,
        )

    def state(self) -> dict:
        return {
            "z": float(self.x[0]) if self.initialized else None,
            "z_dot": float(self.x[1]) if self.initialized else 0.0,
            "confidence": float(self.confidence),
            "valid": bool(self.last_valid),
            "rejected": bool(self.last_rejected),
            "gain": float(self.last_gain),
            "quality": float(self.last_quality),
            "mode": str(self.last_mode),
            "r_t": float(self.last_r_t),
            "R_t": float(self.last_R_t),
            "z_pred": float(self.last_z_pred) if self.last_z_pred is not None else None,
            "z_ref": float(self.last_z_ref) if self.last_z_ref is not None else None,
            "v_ref": float(self.last_v_ref),
            "nis": None if self.last_nis is None else float(self.last_nis),
            "innovation": (
                None if self.last_innovation is None else float(self.last_innovation)
            ),
            "S_t": None if self.last_S_t is None else float(self.last_S_t),
            "P_zz_pred": (
                None if self.last_P_zz_pred is None else float(self.last_P_zz_pred)
            ),
            "dt_s": float(self.last_dt_s),
        }

    def seed_state(self,
                   *,
                   z: float | None,
                   z_dot: float = 0.0,
                   confidence: float = 0.0,
                   p_zz: float | None = None,
                   p_vv: float | None = None) -> None:
        if z is None or not np.isfinite(z):
            return
        if not np.isfinite(z_dot):
            z_dot = 0.0
        self.initialized = True
        self.x[:] = [float(z), float(z_dot)]
        self.P = np.diag([
            float(self.cfg.init_var_z if p_zz is None or not np.isfinite(p_zz) else p_zz),
            float(self.cfg.init_var_v if p_vv is None or not np.isfinite(p_vv) else p_vv),
        ]).astype(np.float64)
        self.confidence = float(np.clip(confidence, 0.0, 1.0))
        self.last_valid = False
        self.last_rejected = False
        self.last_gain = 0.0
        self.last_quality = self.confidence

        self.last_mode = "seeded"
        self.last_r_t = self.confidence
        self.last_R_t = float(self.P[0, 0])
        self.last_z_pred = float(self.x[0])
        self.last_z_ref = float(self.x[0])
        self.last_v_ref = float(self.x[1])
        self.last_nis = None
        self.last_innovation = 0.0
        self.last_S_t = float(self.P[0, 0])
        self.last_P_zz_pred = float(self.P[0, 0])

    def _disabled_state(self, z_raw: float | None, depth_confidence: float | None) -> dict:
        return {
            "z": z_raw,
            "z_dot": 0.0,
            "confidence": float(depth_confidence or 0.0),
            "valid": z_raw is not None and np.isfinite(z_raw),
            "rejected": False,
            "gain": 1.0,
            "quality": float(depth_confidence or 0.0),
            "mode": "disabled",
            "r_t": float(depth_confidence or 0.0),
            "R_t": 0.0,
            "z_pred": z_raw,
            "z_ref": z_raw,
            "v_ref": 0.0,
            "nis": None,
            "innovation": 0.0,
            "S_t": 0.0,
            "P_zz_pred": 0.0,
            "dt_s": float(self.last_dt_s),
        }

    def _predict_state(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), 0.0)
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        qz = max(float(self.cfg.process_var_z), 1e-9)
        qv = max(float(self.cfg.process_var_v), 1e-9)
        q = np.array([
            [qz * max(dt, self.cfg.fallback_dt_s) + 0.25 * qv * dt4, 0.5 * qv * dt3],
            [0.5 * qv * dt3, qv * dt2 + 1e-9],
        ], dtype=np.float64)
        x_pred = f @ self.x
        p_pred = f @ self.P @ f.T + q
        return x_pred, p_pred

    def _correct_from_prediction(self,
                                 x_pred: np.ndarray,
                                 p_pred: np.ndarray,
                                 *,
                                 z_raw: float,
                                 depth_stats,
                                 depth_confidence: float | None,
                                 overwrite_last: bool) -> dict:
        z_pred = float(x_pred[0])
        p_zz_pred = float(p_pred[0, 0])
        r_t, diag = self._quality_from_stats(
            z_raw,
            depth_stats=depth_stats,
            z_pred=z_pred,
            depth_confidence=depth_confidence,
        )
        r_t_meas = self._measurement_variance_from_stats(
            depth_stats,
            fallback_confidence=depth_confidence,
            quality=r_t,
        )
        innovation = float(z_raw - z_pred)
        s = float(p_zz_pred + r_t_meas)
        nis = float((innovation * innovation) / max(s, 1e-9))
        reject_measurement, reject_reason, history_ref = self._reject_measurement(
            z_raw,
            z_pred=z_pred,
            diag=diag,
            nis=nis,
            innovation_var=s,
        )
        if reject_measurement:
            self.x = x_pred
            self.P = p_pred
            self.confidence *= self.cfg.confidence_decay
            self.last_valid = False
            self.last_rejected = True
            self.last_gain = 0.0
            self.last_quality = float(r_t)
            self.last_mode = str(reject_reason)
            self.last_r_t = float(r_t)
            self.last_R_t = float(r_t_meas)
            self.last_z_pred = float(z_pred)
            self.last_z_ref = float(history_ref if history_ref is not None else z_pred)
            self.last_v_ref = float(x_pred[1])
            self.last_nis = float(nis)
            self.last_innovation = float(innovation)
            self.last_S_t = float(s)
            self.last_P_zz_pred = float(p_zz_pred)
            self._append_history(
                z_meas=float(z_raw),
                z_out=float(self.x[0]),
                core_px=int(diag["core_px"]),
                z_iqr=float(diag["z_iqr"]),
                sep_score=float(diag["sep_score"]),
                r_t=float(r_t),
                gain=0.0,
                has_measurement=False,
                overwrite_last=overwrite_last,
            )
            return self.state()

        h = np.array([[1.0, 0.0]], dtype=np.float64)
        s = float((h @ p_pred @ h.T)[0, 0] + r_t_meas)
        k = (p_pred @ h.T)[:, 0] / max(s, 1e-9)
        x_new = x_pred + k * innovation
        kh = np.outer(k, h[0])
        i = np.eye(2, dtype=np.float64)
        p_new = (i - kh) @ p_pred @ (i - kh).T + np.outer(k, k) * r_t_meas

        self.x = x_new
        self.P = p_new
        self.confidence = float(r_t)
        self.last_valid = True
        self.last_rejected = bool(k[0] <= self.cfg.weak_gain_threshold)
        self.last_gain = float(k[0])
        self.last_quality = float(r_t)
        self.last_mode = self._mode_from_gain(float(k[0]))
        self.last_r_t = float(r_t)
        self.last_R_t = float(r_t_meas)
        self.last_z_pred = float(z_pred)
        self.last_z_ref = float(z_pred)
        self.last_v_ref = float(x_pred[1])
        self.last_nis = float(nis)
        self.last_innovation = float(innovation)
        self.last_S_t = float(s)
        self.last_P_zz_pred = float(p_zz_pred)
        self._append_history(
            z_meas=float(z_raw),
            z_out=float(self.x[0]),
            core_px=int(diag["core_px"]),
            z_iqr=float(diag["z_iqr"]),
            sep_score=float(diag["sep_score"]),
            r_t=float(r_t),
            gain=float(k[0]),
            has_measurement=True,
            overwrite_last=overwrite_last,
        )
        return self.state()

    def _reject_measurement(self,
                            z_raw: float,
                            *,
                            z_pred: float,
                            diag: dict,
                            nis: float,
                            innovation_var: float) -> tuple[bool, str | None, float | None]:
        if bool(self.cfg.nis_gate_enabled):
            if np.isfinite(nis) and np.isfinite(innovation_var):
                if nis > float(self.cfg.nis_gate_threshold):
                    history_ref = self._history_reference_depth()
                    return True, "depth_reject_nis", history_ref

        if not bool(self.cfg.hard_gate_enabled):
            return False, None, None

        history = self._history_depth_values()
        if len(history) < int(self.cfg.hard_gate_min_history):
            return False, None, None

        history_tail = history[-int(self.cfg.hard_gate_history_size):]
        z_hist = float(np.median(np.asarray(history_tail, dtype=np.float64)))
        pred_scale = max(abs(float(z_pred)), abs(z_hist), 1e-6)
        hist_scale = max(abs(z_hist), 1e-6)
        pred_limit = max(float(self.cfg.hard_gate_pred_margin_m),
                         float(self.cfg.hard_gate_rel_ratio) * pred_scale)
        hist_limit = max(float(self.cfg.hard_gate_hist_margin_m),
                         float(self.cfg.hard_gate_rel_ratio) * hist_scale)

        pred_delta = abs(float(z_raw) - float(z_pred))
        hist_delta = abs(float(z_raw) - float(z_hist))
        reject = pred_delta > pred_limit and hist_delta > hist_limit
        return bool(reject), ("depth_reject_hard" if reject else None), z_hist

    def _history_depth_values(self) -> list[float]:
        return [
            float(rec["z_out"])
            for rec in self._history
            if np.isfinite(rec.get("z_out", np.nan))
        ]

    def _history_reference_depth(self) -> float | None:
        history = self._history_depth_values()
        if not history:
            return None
        history_tail = history[-int(self.cfg.hard_gate_history_size):]
        return float(np.median(np.asarray(history_tail, dtype=np.float64)))

    def _measurement_variance_from_stats(self,
                                         depth_stats,
                                         *,
                                         fallback_confidence: float | None,
                                         quality: float | None = None) -> float:
        if self.cfg.measurement_variance_mode in {
            "quality_scaled",
            "legacy_quality",
        }:
            measurement_quality = float(np.clip(
                quality if quality is not None else (fallback_confidence or 0.0),
                0.0,
                1.0,
            ))
            return float(self.cfg.r_base / max(measurement_quality, self.cfg.r_min))

        core_px = int(max(getattr(depth_stats, "core_px", 0) or 0, 0))
        z_iqr = float(getattr(depth_stats, "z_iqr", np.inf))
        if np.isfinite(z_iqr) and core_px > 0:
            sigma_depth_sample = 0.7413 * max(z_iqr, 0.0)
            n_eff = max(core_px / max(self.cfg.effective_sample_divisor, 1.0), 1.0)
            r_raw = (np.pi / (2.0 * float(n_eff))) * (sigma_depth_sample ** 2)
            r_lo = min(self.cfg.measurement_var_min, self.cfg.measurement_var_max)
            r_hi = max(self.cfg.measurement_var_min, self.cfg.measurement_var_max)
            return float(np.clip(r_raw, r_lo, r_hi))

        fallback_conf = float(np.clip(fallback_confidence or 0.0, 0.0, 1.0))
        fallback_conf = max(fallback_conf, self.cfg.r_min)
        return float(self.cfg.r_base / fallback_conf)

    def _quality_from_stats(self,
                            z_raw: float,
                            *,
                            depth_stats,
                            z_pred: float,
                            depth_confidence: float | None) -> tuple[float, dict]:
        core_px = int(max(getattr(depth_stats, "core_px", 0) or 0, 0))
        z_iqr = float(getattr(depth_stats, "z_iqr", np.inf))
        if not np.isfinite(z_iqr):
            z_iqr = float("inf")
        sep_score = float(getattr(depth_stats, "sep_score", np.nan))
        z_residual = abs(float(z_raw) - float(z_pred))

        if core_px <= 0:
            fallback_conf = float(np.clip(depth_confidence or 0.0, 0.0, 1.0))
            return fallback_conf, {
                "core_px": 0,
                "z_iqr": z_iqr,
                "sep_score": sep_score,
                "z_residual": z_residual,
            }

        s_px = float(np.clip(core_px / max(self.cfg.n_ref, 1.0), 0.0, 1.0))
        s_iqr = float(np.exp(-max(z_iqr, 0.0) / self.cfg.sigma_iqr)) if np.isfinite(z_iqr) else 0.0
        s_sep = float(np.clip(sep_score / self.cfg.sep_ref, 0.0, 1.0)) if np.isfinite(sep_score) else 0.0
        s_res = float(np.exp(-max(z_residual, 0.0) / self.cfg.sigma_res))

        r_t = float(max(s_px * s_iqr * s_sep * s_res, 0.0) ** 0.25)
        if np.isfinite(sep_score) and sep_score < 0.0 and np.isfinite(z_iqr) and z_iqr > self.cfg.z_iqr_hard:
            r_t = min(r_t, self.cfg.r_hard_bad)

        return float(np.clip(r_t, 0.0, 1.0)), {
            "core_px": core_px,
            "z_iqr": z_iqr,
            "sep_score": sep_score,
            "z_residual": z_residual,
        }

    def _append_history(self, *, z_meas, z_out: float, core_px: int,
                        z_iqr: float, sep_score: float,
                        r_t: float, gain: float,
                        has_measurement: bool,
                        overwrite_last: bool = False) -> None:
        rec = {
            "t": float(self._time),
            "z_meas": None if z_meas is None else float(z_meas),
            "z_out": float(z_out),
            "core_px": int(core_px),
            "z_iqr": float(z_iqr),
            "sep_score": float(sep_score),
            "r_t": float(r_t),
            "gain": float(gain),
            "has_measurement": bool(has_measurement),
        }
        if overwrite_last and self._history:
            self._history[-1] = rec
        else:
            self._history.append(rec)

    def _mode_from_gain(self, gain: float) -> str:
        if gain > self.cfg.update_gain_threshold:
            return "update"
        if gain > self.cfg.weak_gain_threshold:
            return "weak_update"
        return "predict_only_like"

    def _resolve_dt(self, dt_s: float | None) -> float:
        if dt_s is None or not np.isfinite(dt_s) or dt_s <= 0:
            return float(self.cfg.fallback_dt_s)
        return float(dt_s)
