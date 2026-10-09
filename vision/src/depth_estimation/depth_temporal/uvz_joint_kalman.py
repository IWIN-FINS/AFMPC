from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class UVZJointKalmanConfig:
    enabled: bool = True
    fallback_dt_s: float = 0.1
    sigma_a_u_px: float = 20.0
    sigma_a_v_px: float = 14.0
    sigma_a_z_m: float = 0.40
    sigma_u_meas_px: float = 4.0
    sigma_v_meas_px: float = 3.0
    init_sigma_u_px: float = 8.0
    init_sigma_v_px: float = 8.0
    init_sigma_z_m: float = 0.35
    init_sigma_udot_px_s: float = 40.0
    init_sigma_vdot_px_s: float = 40.0
    init_sigma_zdot_m_s: float = 0.8
    effective_sample_divisor: float = 1.0
    r_z_min_m2: float = 2.5e-4
    r_z_max_m2: float = 0.16
    nis_gate_enabled: bool = False
    partial_update_enabled: bool = True
    gate_chi2_df1: float = 6.63
    gate_chi2_df3: float = 11.34
    gate_chi2_df2: float = 9.21
    hard_gate_enabled: bool = True
    hard_gate_history_size: int = 4
    hard_gate_min_history: int = 3
    hard_gate_pred_margin_m: float = 0.28
    hard_gate_hist_margin_m: float = 0.28
    hard_gate_rel_ratio: float = 0.35
    confidence_decay: float = 0.8

    @classmethod
    def from_dict(cls, cfg: dict | None) -> "UVZJointKalmanConfig":
        cfg = dict(cfg or {})
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            fallback_dt_s=float(cfg.get("fallback_dt_s", 0.1)),
            sigma_a_u_px=max(float(cfg.get("sigma_a_u_px", 20.0)), 1e-6),
            sigma_a_v_px=max(float(cfg.get("sigma_a_v_px", 14.0)), 1e-6),
            sigma_a_z_m=max(float(cfg.get("sigma_a_z_m", 0.40)), 1e-6),
            sigma_u_meas_px=max(float(cfg.get("sigma_u_meas_px", 4.0)), 1e-6),
            sigma_v_meas_px=max(float(cfg.get("sigma_v_meas_px", 3.0)), 1e-6),
            init_sigma_u_px=max(float(cfg.get("init_sigma_u_px", 8.0)), 1e-6),
            init_sigma_v_px=max(float(cfg.get("init_sigma_v_px", 8.0)), 1e-6),
            init_sigma_z_m=max(float(cfg.get("init_sigma_z_m", 0.35)), 1e-6),
            init_sigma_udot_px_s=max(float(cfg.get("init_sigma_udot_px_s", 40.0)), 1e-6),
            init_sigma_vdot_px_s=max(float(cfg.get("init_sigma_vdot_px_s", 40.0)), 1e-6),
            init_sigma_zdot_m_s=max(float(cfg.get("init_sigma_zdot_m_s", 0.8)), 1e-6),
            effective_sample_divisor=max(
                float(cfg.get("effective_sample_divisor", 1.0)), 1.0
            ),
            r_z_min_m2=max(float(cfg.get("r_z_min_m2", 2.5e-4)), 1e-12),
            r_z_max_m2=max(float(cfg.get("r_z_max_m2", 0.16)), 1e-12),
            nis_gate_enabled=bool(cfg.get("nis_gate_enabled", False)),
            partial_update_enabled=bool(cfg.get("partial_update_enabled", True)),
            gate_chi2_df1=max(float(cfg.get("gate_chi2_df1", 6.63)), 1e-6),
            gate_chi2_df3=max(float(cfg.get("gate_chi2_df3", 11.34)), 1e-6),
            gate_chi2_df2=max(float(cfg.get("gate_chi2_df2", 9.21)), 1e-6),
            hard_gate_enabled=bool(cfg.get("hard_gate_enabled", True)),
            hard_gate_history_size=max(int(cfg.get("hard_gate_history_size", 4)), 1),
            hard_gate_min_history=max(int(cfg.get("hard_gate_min_history", 3)), 1),
            hard_gate_pred_margin_m=max(
                float(cfg.get("hard_gate_pred_margin_m", 0.28)), 0.0
            ),
            hard_gate_hist_margin_m=max(
                float(cfg.get("hard_gate_hist_margin_m", 0.28)), 0.0
            ),
            hard_gate_rel_ratio=max(float(cfg.get("hard_gate_rel_ratio", 0.35)), 0.0),
            confidence_decay=float(np.clip(cfg.get("confidence_decay", 0.8), 0.0, 1.0)),
        )


class UVZJointKalmanFilter:
    """Joint constant-velocity Kalman filter over image center and depth."""

    def __init__(self, cfg: dict | None = None):
        raw_cfg = dict(cfg or {})
        self.cfg = UVZJointKalmanConfig.from_dict(raw_cfg)
        self.enabled = bool(self.cfg.enabled)
        self.camera = dict(raw_cfg.get("camera", {}))

        self.initialized = False
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.diag([
            self.cfg.init_sigma_u_px ** 2,
            self.cfg.init_sigma_v_px ** 2,
            self.cfg.init_sigma_z_m ** 2,
            self.cfg.init_sigma_udot_px_s ** 2,
            self.cfg.init_sigma_vdot_px_s ** 2,
            self.cfg.init_sigma_zdot_m_s ** 2,
        ]).astype(np.float64)

        self.confidence = 0.0
        self.last_valid = False
        self.last_rejected = False
        self.last_gain = 0.0
        self.last_quality = 0.0
        self.last_mode = "predict_only"
        self.last_R_t = 0.0
        self.last_nis = np.nan
        self.last_uv_nis = np.nan
        self._z_history = deque(maxlen=max(
            self.cfg.hard_gate_history_size,
            self.cfg.hard_gate_min_history,
            2,
        ))

    def predict_only(self, dt_s: float | None = None, **_: object) -> None:
        if not self.enabled or not self.initialized:
            self.confidence *= self.cfg.confidence_decay
            self.last_valid = False
            self.last_rejected = False
            self.last_gain = 0.0
            self.last_quality = 0.0
            self.last_mode = "predict_only"
            self.last_nis = np.nan
            return
        x_pred, p_pred = self._predict_state(self._resolve_dt(dt_s))
        self.x = x_pred
        self.P = p_pred
        self.confidence *= self.cfg.confidence_decay
        self.last_valid = False
        self.last_rejected = False
        self.last_gain = 0.0
        self.last_quality = 0.0
        self.last_mode = "predict_only"
        self.last_nis = np.nan
        self._append_z_history(float(self.x[2]))

    def update(self,
               z_raw: float | None,
               depth_stats=None,
               depth_confidence: float | None = None,
               dt_s: float | None = None,
               center_uv: tuple[float, float] | None = None,
               _overwrite_history: bool = False,
               **_: object) -> dict:
        depth_confidence = float(depth_confidence or 0.0)
        if not self.enabled:
            return self._disabled_state(z_raw, center_uv, depth_confidence)

        center_valid = self._center_valid(center_uv)
        z_valid = z_raw is not None and np.isfinite(z_raw)
        if not self.initialized:
            if not (center_valid and z_valid):
                self.predict_only(dt_s)
                return self.state()
            u_meas, v_meas = float(center_uv[0]), float(center_uv[1])
            z_meas = float(z_raw)
            self.x[:] = [u_meas, v_meas, z_meas, 0.0, 0.0, 0.0]
            self.P = np.diag([
                self.cfg.init_sigma_u_px ** 2,
                self.cfg.init_sigma_v_px ** 2,
                self._compute_r_z(depth_stats),
                self.cfg.init_sigma_udot_px_s ** 2,
                self.cfg.init_sigma_vdot_px_s ** 2,
                self.cfg.init_sigma_zdot_m_s ** 2,
            ]).astype(np.float64)
            self.initialized = True
            self.confidence = float(np.clip(depth_confidence, 0.0, 1.0))
            self.last_valid = True
            self.last_rejected = False
            self.last_gain = 1.0
            self.last_quality = self.confidence
            self.last_mode = "init_uvz"
            self.last_R_t = float(self.P[2, 2])
            self.last_nis = 0.0
            self._append_z_history(z_meas, overwrite_last=_overwrite_history)
            return self.state()

        x_pred, p_pred = self._predict_state(self._resolve_dt(dt_s))
        if not center_valid and not z_valid:
            self.x = x_pred
            self.P = p_pred
            self.confidence *= self.cfg.confidence_decay
            self.last_valid = False
            self.last_rejected = False
            self.last_gain = 0.0
            self.last_quality = 0.0
            self.last_mode = "predict_only"
            self.last_nis = np.nan
            self._append_z_history(
                float(self.x[2]), overwrite_last=_overwrite_history
            )
            return self.state()

        if center_valid and z_valid:
            return self._update_uvz_blockwise(
                x_pred,
                p_pred,
                center_uv=(float(center_uv[0]), float(center_uv[1])),
                z_raw=float(z_raw),
                depth_stats=depth_stats,
                overwrite_history=_overwrite_history,
            )

        if center_valid and not self.cfg.partial_update_enabled:
            self.x = x_pred
            self.P = p_pred
            self.confidence *= self.cfg.confidence_decay
            self.last_valid = False
            self.last_rejected = False
            self.last_gain = 0.0
            self.last_quality = 0.0
            self.last_mode = "depth_missing_predict_all"
            self.last_nis = np.nan
            self._append_z_history(
                float(self.x[2]), overwrite_last=_overwrite_history
            )
            return self.state()

        if center_valid:
            state = self._try_update(
                x_pred,
                p_pred,
                y=np.array([float(center_uv[0]), float(center_uv[1])], dtype=np.float64),
                h=np.array([
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ], dtype=np.float64),
                r=np.diag([
                    self.cfg.sigma_u_meas_px ** 2,
                    self.cfg.sigma_v_meas_px ** 2,
                ]).astype(np.float64),
                gate=(
                    float(self.cfg.gate_chi2_df2)
                    if self.cfg.nis_gate_enabled else None
                ),
                mode_accept="update_uv_only",
                quality=self._compute_quality(depth_stats, z_valid=False),
            )
            if state is not None:
                self._append_z_history(
                    float(self.x[2]), overwrite_last=_overwrite_history
                )
                return state

        self.x = x_pred
        self.P = p_pred
        self.confidence *= self.cfg.confidence_decay
        self.last_valid = False
        self.last_rejected = True
        self.last_gain = 0.0
        self.last_quality = 0.0
        self.last_mode = "uv_reject"
        self._append_z_history(float(self.x[2]), overwrite_last=_overwrite_history)
        return self.state()

    def correct_only(self,
                     z_raw: float | None,
                     depth_stats=None,
                     depth_confidence: float | None = None,
                     dt_s: float | None = None,
                     center_uv: tuple[float, float] | None = None,
                     **kwargs: object) -> dict:
        if not self.initialized:
            return self.update(
                z_raw,
                depth_stats=depth_stats,
                depth_confidence=depth_confidence,
                dt_s=dt_s,
                center_uv=center_uv,
                **kwargs,
            )
        if not self._center_valid(center_uv) and (
            z_raw is None or not np.isfinite(z_raw)
        ):
            return self.state()
        return self.update(
            z_raw,
            depth_stats=depth_stats,
            depth_confidence=depth_confidence,
            dt_s=0.0,
            center_uv=center_uv,
            _overwrite_history=True,
            **kwargs,
        )

    def state(self) -> dict:
        center_uv = None
        pos_3d = None
        if self.initialized:
            center_uv = (float(self.x[0]), float(self.x[1]))
            pos_3d = self._uvz_to_xyz(center_uv, float(self.x[2]))
        return {
            "z": float(self.x[2]) if self.initialized else None,
            "z_dot": float(self.x[5]) if self.initialized else 0.0,
            "u_dot": float(self.x[3]) if self.initialized else 0.0,
            "v_dot": float(self.x[4]) if self.initialized else 0.0,
            "center_uv": center_uv,
            "position": pos_3d,
            "confidence": float(self.confidence),
            "valid": bool(self.last_valid),
            "rejected": bool(self.last_rejected),
            "gain": float(self.last_gain),
            "quality": float(self.last_quality),
            "mode": str(self.last_mode),
            "r_t": float(self.confidence),
            "R_t": float(self.last_R_t),
            "nis": float(self.last_nis) if np.isfinite(self.last_nis) else None,
            "uv_nis": float(self.last_uv_nis) if np.isfinite(self.last_uv_nis) else None,
            "z_pred": float(self.x[2]) if self.initialized else None,
            "z_ref": float(self.x[2]) if self.initialized else None,
            "v_ref": float(self.x[5]) if self.initialized else 0.0,
        }

    def seed_state(self,
                   *,
                   z: float | None,
                   center_uv: tuple[float, float] | None = None,
                   z_dot: float = 0.0,
                   u_dot: float = 0.0,
                   v_dot: float = 0.0,
                   confidence: float = 0.0,
                   p_diag: list[float] | None = None,
                   p_zz: float | None = None,
                   p_vv: float | None = None,
                   **_: object) -> None:
        if z is None or not np.isfinite(z) or not self._center_valid(center_uv):
            return
        self.initialized = True
        self.x[:] = [
            float(center_uv[0]),
            float(center_uv[1]),
            float(z),
            float(u_dot) if np.isfinite(u_dot) else 0.0,
            float(v_dot) if np.isfinite(v_dot) else 0.0,
            float(z_dot) if np.isfinite(z_dot) else 0.0,
        ]
        if p_diag is not None and len(p_diag) >= 6:
            diag = [max(float(v), 1e-9) if np.isfinite(v) else 1.0 for v in p_diag[:6]]
            self.P = np.diag(diag).astype(np.float64)
        else:
            self.P = np.diag([
                self.cfg.init_sigma_u_px ** 2,
                self.cfg.init_sigma_v_px ** 2,
                float(self.cfg.init_sigma_z_m ** 2 if p_zz is None or not np.isfinite(p_zz) else p_zz),
                self.cfg.init_sigma_udot_px_s ** 2,
                float(self.cfg.init_sigma_vdot_px_s ** 2 if p_vv is None or not np.isfinite(p_vv) else p_vv),
                self.cfg.init_sigma_zdot_m_s ** 2,
            ]).astype(np.float64)
        self.confidence = float(np.clip(confidence, 0.0, 1.0))
        self.last_valid = False
        self.last_rejected = False
        self.last_gain = 0.0
        self.last_quality = self.confidence
        self.last_mode = "seeded"
        self.last_R_t = float(self.P[2, 2])
        self.last_nis = np.nan
        self.last_uv_nis = np.nan
        self._z_history.clear()
        self._append_z_history(float(self.x[2]))

    def _update_uvz_blockwise(self,
                              x_pred: np.ndarray,
                              p_pred: np.ndarray,
                              *,
                              center_uv: tuple[float, float],
                              z_raw: float,
                              depth_stats,
                              overwrite_history: bool = False) -> dict:
        """Gate image center and depth independently within one 6D state."""
        h_uv = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64)
        r_uv = np.diag([
            self.cfg.sigma_u_meas_px ** 2,
            self.cfg.sigma_v_meas_px ** 2,
        ]).astype(np.float64)
        uv_ok, x_after_uv, p_after_uv, uv_nis, uv_gain = self._block_update(
            x_pred,
            p_pred,
            y=np.asarray(center_uv, dtype=np.float64),
            h=h_uv,
            r=r_uv,
            gate=(
                float(self.cfg.gate_chi2_df2)
                if self.cfg.nis_gate_enabled else None
            ),
        )

        h_z = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        r_z_value = self._compute_r_z(depth_stats)
        hard_reject_z = self._hard_gate_reject(z_raw, z_pred=float(x_pred[2]))
        z_ok, x_new, p_new, z_nis, z_gain = self._block_update(
            x_after_uv if uv_ok else x_pred,
            p_after_uv if uv_ok else p_pred,
            y=np.asarray([z_raw], dtype=np.float64),
            h=h_z,
            r=np.asarray([[r_z_value]], dtype=np.float64),
            gate=(
                float(self.cfg.gate_chi2_df1)
                if self.cfg.nis_gate_enabled else None
            ),
        )
        z_ok = bool(z_ok and not hard_reject_z)

        if not self.cfg.partial_update_enabled and not (uv_ok and z_ok):
            self.x = x_pred
            self.P = p_pred
            self.confidence *= self.cfg.confidence_decay
            self.last_gain = 0.0
            self.last_quality = 0.0
            self.last_valid = False
            self.last_rejected = bool(not z_ok)
            if hard_reject_z:
                self.last_mode = "depth_reject_hard_predict_all"
            elif not z_ok:
                self.last_mode = "depth_reject_predict_all"
            else:
                self.last_mode = "center_reject_predict_all"
            self.last_R_t = float(r_z_value)
            self.last_nis = float(z_nis)
            self.last_uv_nis = float(uv_nis)
            self._append_z_history(
                float(self.x[2]), overwrite_last=overwrite_history
            )
            return self.state()

        if uv_ok or z_ok:
            self.x = x_new if z_ok else x_after_uv
            self.P = p_new if z_ok else p_after_uv
            self.confidence = self._compute_quality(depth_stats, z_valid=z_ok)
            gains = [gain for ok, gain in ((uv_ok, uv_gain), (z_ok, z_gain)) if ok]
            self.last_gain = float(np.mean(gains)) if gains else 0.0
            self.last_quality = float(self.confidence)
            self.last_valid = True
        else:
            self.x = x_pred
            self.P = p_pred
            self.confidence *= self.cfg.confidence_decay
            self.last_gain = 0.0
            self.last_quality = 0.0
            self.last_valid = False

        self.last_rejected = not (uv_ok and z_ok)
        if uv_ok and z_ok:
            self.last_mode = "update_uvz"
        elif uv_ok:
            self.last_mode = (
                "depth_reject_hard_uv_only"
                if hard_reject_z else "depth_reject_uv_only"
            )
        elif z_ok:
            self.last_mode = "center_reject_z_only"
        else:
            self.last_mode = "uvz_reject"
        self.last_R_t = float(r_z_value)
        self.last_nis = float(z_nis)
        self.last_uv_nis = float(uv_nis)
        self._append_z_history(
            float(self.x[2]), overwrite_last=overwrite_history
        )
        return self.state()

    def _hard_gate_reject(self, z_raw: float, *, z_pred: float) -> bool:
        if not self.cfg.hard_gate_enabled:
            return False
        history = [float(value) for value in self._z_history if np.isfinite(value)]
        if len(history) < self.cfg.hard_gate_min_history:
            return False
        history_tail = history[-self.cfg.hard_gate_history_size:]
        z_hist = float(np.median(np.asarray(history_tail, dtype=np.float64)))
        pred_scale = max(abs(float(z_pred)), abs(z_hist), 1e-6)
        hist_scale = max(abs(z_hist), 1e-6)
        pred_limit = max(
            self.cfg.hard_gate_pred_margin_m,
            self.cfg.hard_gate_rel_ratio * pred_scale,
        )
        hist_limit = max(
            self.cfg.hard_gate_hist_margin_m,
            self.cfg.hard_gate_rel_ratio * hist_scale,
        )
        return bool(
            abs(float(z_raw) - float(z_pred)) > pred_limit
            and abs(float(z_raw) - z_hist) > hist_limit
        )

    def _append_z_history(self, z_out: float, *, overwrite_last: bool = False) -> None:
        if not np.isfinite(z_out):
            return
        if overwrite_last and self._z_history:
            self._z_history[-1] = float(z_out)
        else:
            self._z_history.append(float(z_out))

    @staticmethod
    def _block_update(x_pred: np.ndarray,
                      p_pred: np.ndarray,
                      *,
                      y: np.ndarray,
                      h: np.ndarray,
                      r: np.ndarray,
                      gate: float | None) -> tuple[bool, np.ndarray, np.ndarray, float, float]:
        innovation = y - h @ x_pred
        s = h @ p_pred @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s)
        nis = float(innovation.T @ s_inv @ innovation)
        if not np.isfinite(nis) or (gate is not None and nis > gate):
            return False, x_pred, p_pred, nis, 0.0
        k = p_pred @ h.T @ s_inv
        x_new = x_pred + k @ innovation
        identity = np.eye(p_pred.shape[0], dtype=np.float64)
        p_new = (
            (identity - k @ h) @ p_pred @ (identity - k @ h).T
            + k @ r @ k.T
        )
        observed_state_indices = np.argmax(np.abs(h), axis=1)
        gains = [float(k[index, column]) for column, index in enumerate(observed_state_indices)]
        return True, x_new, p_new, nis, float(np.mean(gains))

    def _try_update(self,
                    x_pred: np.ndarray,
                    p_pred: np.ndarray,
                    *,
                    y: np.ndarray,
                    h: np.ndarray,
                    r: np.ndarray,
                    gate: float | None,
                    mode_accept: str,
                    quality: float) -> dict | None:
        nu = y - h @ x_pred
        s = h @ p_pred @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s)
        nis = float(nu.T @ s_inv @ nu)
        if not np.isfinite(nis) or (gate is not None and nis > gate):
            self.last_nis = nis
            return None
        k = p_pred @ h.T @ s_inv
        x_new = x_pred + k @ nu
        i = np.eye(6, dtype=np.float64)
        p_new = (i - k @ h) @ p_pred @ (i - k @ h).T + k @ r @ k.T
        self.x = x_new
        self.P = p_new
        self.confidence = float(np.clip(quality, 0.0, 1.0))
        self.last_valid = True
        self.last_rejected = False
        self.last_gain = float(np.mean(np.diag(k[: h.shape[0], :]) if h.shape[0] <= k.shape[0] else k[:, :h.shape[0]]))
        if not np.isfinite(self.last_gain):
            pos_gains = []
            for idx in range(min(h.shape[0], 3)):
                if idx < k.shape[0] and idx < k.shape[1]:
                    pos_gains.append(float(k[idx, idx]))
            self.last_gain = float(np.mean(pos_gains)) if pos_gains else 0.0
        self.last_quality = float(np.clip(quality, 0.0, 1.0))
        self.last_mode = mode_accept
        self.last_R_t = float(r[-1, -1]) if r.size else 0.0
        self.last_nis = nis
        return self.state()

    def _predict_state(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt_s), 0.0)
        f = np.array([
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, dt],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        g = np.array([
            [0.5 * dt * dt, 0.0, 0.0],
            [0.0, 0.5 * dt * dt, 0.0],
            [0.0, 0.0, 0.5 * dt * dt],
            [dt, 0.0, 0.0],
            [0.0, dt, 0.0],
            [0.0, 0.0, dt],
        ], dtype=np.float64)
        qa = np.diag([
            self.cfg.sigma_a_u_px ** 2,
            self.cfg.sigma_a_v_px ** 2,
            self.cfg.sigma_a_z_m ** 2,
        ]).astype(np.float64)
        q = g @ qa @ g.T
        x_pred = f @ self.x
        p_pred = f @ self.P @ f.T + q
        return x_pred, p_pred

    def _compute_r_z(self, depth_stats) -> float:
        if depth_stats is None:
            return float(self.cfg.r_z_max_m2)
        z_iqr = float(getattr(depth_stats, "z_iqr", np.nan))
        core_px = int(max(getattr(depth_stats, "core_px", 0) or 0, 0))
        if not np.isfinite(z_iqr) or core_px <= 0:
            return float(self.cfg.r_z_max_m2)
        sigma_depth_sample = 0.7413 * max(z_iqr, 0.0)
        n_eff = max(
            core_px / max(self.cfg.effective_sample_divisor, 1.0),
            1.0,
        )
        r_raw = (np.pi / (2.0 * float(n_eff))) * (sigma_depth_sample ** 2)
        return float(np.clip(r_raw, self.cfg.r_z_min_m2, self.cfg.r_z_max_m2))

    def _compute_quality(self, depth_stats, *, z_valid: bool) -> float:
        q_u = float(np.clip(4.0 / max(self.cfg.sigma_u_meas_px, 4.0), 0.0, 1.0))
        q_v = float(np.clip(4.0 / max(self.cfg.sigma_v_meas_px, 4.0), 0.0, 1.0))
        q = min(q_u, q_v)
        if z_valid:
            r_z = self._compute_r_z(depth_stats)
            q_z = float(np.sqrt(self.cfg.r_z_min_m2 / max(r_z, self.cfg.r_z_min_m2)))
            q = min(q, q_z)
        return float(np.clip(q, 0.0, 1.0))

    def _disabled_state(self,
                        z_raw: float | None,
                        center_uv: tuple[float, float] | None,
                        depth_confidence: float) -> dict:
        pos = None
        if self._center_valid(center_uv) and z_raw is not None and np.isfinite(z_raw):
            pos = self._uvz_to_xyz((float(center_uv[0]), float(center_uv[1])), float(z_raw))
        return {
            "z": float(z_raw) if z_raw is not None and np.isfinite(z_raw) else None,
            "z_dot": 0.0,
            "u_dot": 0.0,
            "v_dot": 0.0,
            "center_uv": (
                (float(center_uv[0]), float(center_uv[1]))
                if self._center_valid(center_uv) else None
            ),
            "position": pos,
            "confidence": float(depth_confidence),
            "valid": bool(self._center_valid(center_uv) and z_raw is not None and np.isfinite(z_raw)),
            "rejected": False,
            "gain": 1.0,
            "quality": float(depth_confidence),
            "mode": "disabled",
            "r_t": float(depth_confidence),
            "R_t": 0.0,
            "nis": None,
            "z_pred": float(z_raw) if z_raw is not None and np.isfinite(z_raw) else None,
            "z_ref": float(z_raw) if z_raw is not None and np.isfinite(z_raw) else None,
            "v_ref": 0.0,
        }

    def _uvz_to_xyz(self,
                    center_uv: tuple[float, float],
                    z_m: float) -> np.ndarray | None:
        fx = float(self.camera.get("fx", np.nan))
        fy = float(self.camera.get("fy", fx))
        cx = float(self.camera.get("cx", np.nan))
        cy = float(self.camera.get("cy", np.nan))
        if not all(np.isfinite(v) for v in (fx, fy, cx, cy, z_m)):
            return None
        u, v = float(center_uv[0]), float(center_uv[1])
        x = (u - cx) * z_m / fx
        y = (v - cy) * z_m / fy
        return np.array([x, y, z_m], dtype=np.float32)

    @staticmethod
    def _center_valid(center_uv: tuple[float, float] | None) -> bool:
        return (
            center_uv is not None
            and len(center_uv) >= 2
            and np.isfinite(center_uv[0])
            and np.isfinite(center_uv[1])
        )

    def _resolve_dt(self, dt_s: float | None) -> float:
        if dt_s is None or not np.isfinite(dt_s) or dt_s <= 0:
            return float(self.cfg.fallback_dt_s)
        return float(dt_s)
