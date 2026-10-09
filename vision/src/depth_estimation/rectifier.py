"""Stereo image rectification utilities."""

from __future__ import annotations

import cv2
import numpy as np


class StereoRectifier:
    """Lazy OpenCV stereo rectifier initialized from the first frame size."""

    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.alpha = float(cfg.get("alpha", 0.0))
        self.flags = cv2.CALIB_ZERO_DISPARITY if cfg.get("zero_disparity", True) else 0
        self.calibration_size = self._read_calibration_size(cfg)

        self.K_left = self._matrix(cfg.get("K_left"), (3, 3))
        self.K_right = self._matrix(cfg.get("K_right"), (3, 3))
        self.D_left = self._vector(cfg.get("D_left"))
        self.D_right = self._vector(cfg.get("D_right"))
        self.R = self._matrix(cfg.get("R"), (3, 3))
        self.T = self._matrix(cfg.get("T"), (3, 1))
        if self.enabled and any(
            value is None
            for value in (self.K_left, self.K_right, self.D_left, self.D_right,
                          self.R, self.T)
        ):
            raise ValueError(
                "rectification.enabled is true, but K_left/K_right/D_left/"
                "D_right/R/T are not fully configured."
            )

        self._image_size = None
        self._maps = None
        self._camera_params = None

    def rectify(self, left, right):
        """Return rectified left/right images, or originals when disabled."""
        if not self.enabled:
            return left, right
        image_size = (int(left.shape[1]), int(left.shape[0]))
        if self._image_size != image_size:
            self._init_maps(image_size)
        map_l1, map_l2, map_r1, map_r2 = self._maps
        left_rect = cv2.remap(left, map_l1, map_l2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, map_r1, map_r2, cv2.INTER_LINEAR)
        return left_rect, right_rect

    @property
    def camera_params(self) -> dict | None:
        return None if self._camera_params is None else dict(self._camera_params)

    def _init_maps(self, image_size: tuple[int, int]):
        K_left, K_right = self._scaled_intrinsics(image_size)
        R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
            K_left,
            self.D_left,
            K_right,
            self.D_right,
            image_size,
            self.R,
            self.T,
            flags=self.flags,
            alpha=self.alpha,
        )

        map_l1, map_l2 = cv2.initUndistortRectifyMap(
            K_left, self.D_left, R1, P1, image_size, cv2.CV_32FC1)
        map_r1, map_r2 = cv2.initUndistortRectifyMap(
            K_right, self.D_right, R2, P2, image_size, cv2.CV_32FC1)

        baseline = abs(float(P2[0, 3] / P2[0, 0]))
        self._camera_params = {
            "fx": float(P1[0, 0]),
            "fy": float(P1[1, 1]),
            "cx": float(P1[0, 2]),
            "cy": float(P1[1, 2]),
            "baseline_m": baseline,
        }
        self._maps = (map_l1, map_l2, map_r1, map_r2)
        self._image_size = image_size

    def _scaled_intrinsics(self, image_size: tuple[int, int]):
        if self.calibration_size is None or self.calibration_size == image_size:
            return self.K_left.copy(), self.K_right.copy()

        src_w, src_h = self.calibration_size
        dst_w, dst_h = image_size
        sx = dst_w / src_w
        sy = dst_h / src_h
        K_left = self.K_left.copy()
        K_right = self.K_right.copy()
        for K in (K_left, K_right):
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
        return K_left, K_right

    @staticmethod
    def _read_calibration_size(cfg: dict):
        width = cfg.get("calibration_image_width")
        height = cfg.get("calibration_image_height")
        if width in (None, "") or height in (None, ""):
            return None
        return (int(width), int(height))

    @staticmethod
    def _matrix(value, shape):
        if value in (None, ""):
            return None
        return np.array(value, dtype=np.float64).reshape(shape)

    @staticmethod
    def _vector(value):
        if value in (None, ""):
            return None
        return np.array(value, dtype=np.float64).reshape(-1, 1)
