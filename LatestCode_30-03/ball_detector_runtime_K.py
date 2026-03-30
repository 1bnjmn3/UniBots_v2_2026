from __future__ import annotations

import collections
import math
import queue as _queue
import threading
import time
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

try:
    from .ball_detection_krishiv_K import CAMERA_MATRIX, DIST_COEFFS
    from .config_K import (
        ATTRACTION_WEIGHT,
        BALL_DIAMETER_CM,
        BALL_REPULSION_FILTER_ENABLED,
        BALL_REPULSION_MAX_THRESHOLD,
        BALL_REPULSION_MIN_KEEP,
        BALL_TYPE_PRIORITY,
        CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG,
        CAPTURE_Y_THRESHOLD,
        CLOSE_RANGE_CREEP_DISTANCE_CM,
        CLOSE_RANGE_CREEP_MAGNITUDE,
        DEBUG_STREAM_PORT,
        DEBUG_TRAJECTORY_LENGTH,
        DEFAULT_PING_PONG_PROFILE,
        FRAME_GAUSSIAN_BLUR,
        GRAB_DISTANCE_CM,
        GRAB_MAX_ABS_ANGLE_DEG,
        HSVRange,
        KALMAN_MAX_LOST_FRAMES,
        KALMAN_ROI_EXPAND_PX,
        KALMAN_ROI_HALF_SIZE,
        MASK_CLOSE_KERNEL,
        MASK_OPEN_KERNEL,
        MIN_TRACKING_MAGNITUDE,
        OBSTACLE_MAX_AREA,
        OBSTACLE_MAX_VALUE,
        OBSTACLE_BEHIND_TARGET_REPULSION_SCALE,
        OBSTACLE_BLOCKING_BOTTOM_RATIO,
        OBSTACLE_BLOCKING_CENTRAL_DEG,
        OBSTACLE_BLOCKING_MIN_AREA,
        OBSTACLE_LARGE_BLOB_AREA,
        OBSTACLE_MIN_AREA,
        OBSTACLE_MIN_SATURATION,
        OBSTACLE_MIN_VALUE,
        OBSTACLE_SIZE_FULL_SCALE,
        OBSTACLE_TOP_IGNORE_RATIO,
        OPPONENT_LIKELY_REPULSION_BOOST,
        PING_PONG_MAX_AREA,
        PING_PONG_MIN_AREA,
        PING_PONG_MIN_CIRCULARITY,
        PING_PONG_PRESETS,
        PING_PONG_RADIUS_RANGE_PX,
        REPULSION_WEIGHT,
        SPECULAR_MAX_RATIO,
        SPECULAR_MAX_SATURATION,
        SPECULAR_MIN_RATIO,
        SPECULAR_MIN_VALUE,
        STEEL_CLAHE_CLIP_LIMIT,
        STEEL_CLAHE_TILE_GRID_SIZE,
        STEEL_MAX_AREA,
        STEEL_MAX_MEAN_VALUE,
        STEEL_MAX_SATURATION,
        STEEL_MAX_VALUE,
        STEEL_MIN_AREA,
        STEEL_MIN_CIRCULARITY,
        STEEL_MIN_CIRCULARITY_STRICT,
        STEEL_MIN_CONVEXITY,
        STEEL_MIN_VALUE,
        STEEL_MORPH_CLOSE_BIG_KERNEL,
        STEEL_RADIUS_RANGE_PX,
        TARGET_DISTANCE_FAR_CM,
        TARGET_DISTANCE_NEAR_CM,
    )
except ImportError:
    from ball_detection_krishiv_K import CAMERA_MATRIX, DIST_COEFFS
    from config_K import (
        ATTRACTION_WEIGHT,
        BALL_DIAMETER_CM,
        BALL_REPULSION_FILTER_ENABLED,
        BALL_REPULSION_MAX_THRESHOLD,
        BALL_REPULSION_MIN_KEEP,
        BALL_TYPE_PRIORITY,
        CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG,
        CAPTURE_Y_THRESHOLD,
        CLOSE_RANGE_CREEP_DISTANCE_CM,
        CLOSE_RANGE_CREEP_MAGNITUDE,
        DEBUG_STREAM_PORT,
        DEBUG_TRAJECTORY_LENGTH,
        DEFAULT_PING_PONG_PROFILE,
        FRAME_GAUSSIAN_BLUR,
        GRAB_DISTANCE_CM,
        GRAB_MAX_ABS_ANGLE_DEG,
        HSVRange,
        KALMAN_MAX_LOST_FRAMES,
        KALMAN_ROI_EXPAND_PX,
        KALMAN_ROI_HALF_SIZE,
        MASK_CLOSE_KERNEL,
        MASK_OPEN_KERNEL,
        MIN_TRACKING_MAGNITUDE,
        OBSTACLE_MAX_AREA,
        OBSTACLE_MAX_VALUE,
        OBSTACLE_BEHIND_TARGET_REPULSION_SCALE,
        OBSTACLE_BLOCKING_BOTTOM_RATIO,
        OBSTACLE_BLOCKING_CENTRAL_DEG,
        OBSTACLE_BLOCKING_MIN_AREA,
        OBSTACLE_LARGE_BLOB_AREA,
        OBSTACLE_MIN_AREA,
        OBSTACLE_MIN_SATURATION,
        OBSTACLE_MIN_VALUE,
        OBSTACLE_SIZE_FULL_SCALE,
        OBSTACLE_TOP_IGNORE_RATIO,
        OPPONENT_LIKELY_REPULSION_BOOST,
        PING_PONG_MAX_AREA,
        PING_PONG_MIN_AREA,
        PING_PONG_MIN_CIRCULARITY,
        PING_PONG_PRESETS,
        PING_PONG_RADIUS_RANGE_PX,
        REPULSION_WEIGHT,
        SPECULAR_MAX_RATIO,
        SPECULAR_MAX_SATURATION,
        SPECULAR_MIN_RATIO,
        SPECULAR_MIN_VALUE,
        STEEL_CLAHE_CLIP_LIMIT,
        STEEL_CLAHE_TILE_GRID_SIZE,
        STEEL_MAX_AREA,
        STEEL_MAX_MEAN_VALUE,
        STEEL_MAX_SATURATION,
        STEEL_MAX_VALUE,
        STEEL_MIN_AREA,
        STEEL_MIN_CIRCULARITY,
        STEEL_MIN_CIRCULARITY_STRICT,
        STEEL_MIN_CONVEXITY,
        STEEL_MIN_VALUE,
        STEEL_MORPH_CLOSE_BIG_KERNEL,
        STEEL_RADIUS_RANGE_PX,
        TARGET_DISTANCE_FAR_CM,
        TARGET_DISTANCE_NEAR_CM,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Data classes
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    optimal_matrix: np.ndarray
    map1: np.ndarray
    map2: np.ndarray
    fx: float
    cx: float
    frame_size: tuple[int, int]


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Module-level cached objects
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CALIBRATION_CACHE: dict[tuple[int, int], CameraCalibration] = {}
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MASK_OPEN_KERNEL)
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MASK_CLOSE_KERNEL)

# Steel detection derived constants
# Hybrid Hough + HSV validation constants.
_STEEL_CENTER_MAX_SAT = 70
_STEEL_CENTER_MIN_VAL = 20
_STEEL_HOUGH_DP = 1.2
_STEEL_HOUGH_MIN_DIST = 30
_STEEL_HOUGH_PARAM1 = 50
_STEEL_HOUGH_PARAM2 = 25
_STEEL_HOUGH_MIN_RADIUS = 5
_STEEL_HOUGH_MAX_RADIUS = 50

# Steel CLAHE and large morphological-close kernel (fills specular highlight voids)
_STEEL_CLAHE = cv2.createCLAHE(
    clipLimit=STEEL_CLAHE_CLIP_LIMIT,
    tileGridSize=STEEL_CLAHE_TILE_GRID_SIZE,
)

_REPULSION_GLOBAL_SCALE = 0.25
_RUNTIME_LOG_HEARTBEAT = 45
log = logging.getLogger(__name__)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Debug stream module-level state
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Set to False to disable all annotation and MJPEG serving overhead.
STREAM_DEBUG_FEED: bool = False
_debug_frame_queue: _queue.Queue = _queue.Queue(maxsize=1)
_mjpeg_server_started: bool = False
_mjpeg_server_lock = threading.Lock()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Kalman filter tracker for steel balls
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SteelBallKalmanTracker:
    """2D constant-velocity Kalman filter for a single steel ball.

    State vector  : [x, y, dx, dy]
    Measurement   : [x, y]

    Usage
    -----
    1.  predict()  â€“ advance one frame; returns predicted (x, y) or None.
    2.  update(x, y) â€“ correct with a real measurement.
    3.  get_roi(frame_shape) â€“ returns (x1, y1, x2, y2) clamped to the frame.
    4.  .is_initialized â€“ True once at least one measurement has been received.
    5.  .lost_frames â€“ frames since last successful measurement.
    6.  .trajectory â€“ list of recent (x, y) for debug drawing.
    """

    def __init__(self) -> None:
        # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self._kf = cv2.KalmanFilter(4, 2)
        dt = 1.0  # one frame step

        # Constant-velocity transition matrix
        self._kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0,  dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]], dtype=np.float32,
        )
        # Observe only (x, y)
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )
        # Process noise covariance Q (tune for expected inter-frame ball movement)
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        # Measurement noise covariance R (tune for pixel-level detection jitter)
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        # Initial estimate error covariance P
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)

        self._initialized: bool = False
        self.lost_frames: int = 0
        self._trajectory: collections.deque = collections.deque(
            maxlen=DEBUG_TRAJECTORY_LENGTH
        )

        # Motion-gated Kalman state (3-frame confirmation gate + stationary override)
        self._frame_count: int = 0
        self._last_position: tuple[float, float] | None = None
        self.moving: bool = False
        self.moving_streak: int = 0
        self.stationary_streak: int = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def trajectory(self) -> list[tuple[int, int]]:
        return list(self._trajectory)

    @property
    def frames_tracked(self) -> int:
        """Number of frames since this tracker was initialized."""
        return self._frame_count

    def compute_motion_state(self, x: float, y: float) -> tuple[bool, float]:
        """Compute whether the new position indicates motion.

        Returns (is_moving, displacement_px) where:
        - is_moving: True if displacement >= 8px
        - displacement_px: Euclidean distance from last position
        """
        if self._last_position is None:
            return False, 0.0
        lx, ly = self._last_position
        displacement = math.hypot(x - lx, y - ly)
        is_moving = displacement >= 8.0
        return is_moving, displacement

    def predict(self, coast_limit: int = 2) -> tuple[int, int] | None:
        """Step the filter forward and return the predicted (x, y).

        Motion-gated: only predicts if moving mode is active. If coasting
        beyond coast_limit frames while moving, resets tracker and returns None.
        Always returns None if in stationary mode (no coasting).
        """
        if not self._initialized:
            return None
        if not self.moving:
            # Stationary: do not predict, use raw detection instead
            return None
        # Moving mode: check coast limit
        if self.lost_frames > coast_limit:
            # Coasting too long while moving; reset tracker
            self.reset()
            return None
        # Predict and append to trajectory
        predicted = self._kf.predict()
        px = int(predicted[0, 0])
        py = int(predicted[1, 0])
        self._trajectory.append((px, py))
        self.lost_frames += 1
        return px, py

    def update(self, x: float, y: float) -> None:
        """Correct the filter with an observed measurement.

        Updates motion state: after 3 consecutive moving frames, enables moving mode.
        If stationary (<8px movement), disables moving mode after 2 consecutive frames.
        """
        # Increment frame counter
        self._frame_count += 1

        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._initialized:
            # Warm-start: initialise state with the first measurement (zero velocity)
            self._kf.statePost = np.array(
                [[np.float32(x)], [np.float32(y)], [0.0], [0.0]],
                dtype=np.float32,
            )
            self._initialized = True
            self._last_position = (x, y)
            self.lost_frames = 0
            self._trajectory.append((int(x), int(y)))
            return

        # Compute motion state relative to previous measurement
        _, displacement = self.compute_motion_state(x, y)
        if displacement >= 8.0:
            # Movement detected
            self.moving_streak += 1
            self.stationary_streak = 0
            if self.moving_streak >= 3:
                self.moving = True
        else:
            # Stationary (or very small movement)
            self.stationary_streak += 1
            self.moving_streak = 0
            if self.moving and self.stationary_streak >= 2:
                self.moving = False  # Override: switch to stationary mode

        # Always correct Kalman with measurement
        self._kf.correct(measurement)
        self.lost_frames = 0
        self._trajectory.append((int(x), int(y)))
        self._last_position = (x, y)

    def get_roi(self, frame_shape: tuple) -> tuple[int, int, int, int]:
        """Return an (x1, y1, x2, y2) ROI clamped to frame boundaries.

        The half-size grows by KALMAN_ROI_EXPAND_PX for every consecutive
        lost frame, widening the search area when the ball disappears.
        """
        h, w = frame_shape[:2]
        if not self._initialized:
            return 0, 0, w, h  # use the whole frame before first measurement
        state = self._kf.statePost
        px = int(state[0, 0])
        py = int(state[1, 0])
        half = KALMAN_ROI_HALF_SIZE + self.lost_frames * KALMAN_ROI_EXPAND_PX
        x1 = max(px - half, 0)
        y1 = max(py - half, 0)
        x2 = min(px + half, w)
        y2 = min(py + half, h)
        return x1, y1, x2, y2

    def reset(self) -> None:
        """Reset the tracker to an uninitialised state."""
        self._initialized = False
        self.lost_frames = 0
        self._trajectory.clear()
        self._frame_count = 0
        self._last_position = None
        self.moving = False
        self.moving_streak = 0
        self.stationary_streak = 0


# Module-level singleton tracker (one per Python process)
_STEEL_KALMAN = SteelBallKalmanTracker()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Camera calibration helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_calibration(frame_shape: tuple[int, int, int]) -> CameraCalibration:
    height, width = frame_shape[:2]
    frame_size = (width, height)
    cached = _CALIBRATION_CACHE.get(frame_size)
    if cached is not None:
        return cached

    optimal_matrix, _ = cv2.getOptimalNewCameraMatrix(
        CAMERA_MATRIX,
        DIST_COEFFS,
        frame_size,
        0.0,
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        CAMERA_MATRIX,
        DIST_COEFFS,
        None,
        optimal_matrix,
        frame_size,
        cv2.CV_16SC2,
    )
    calibration = CameraCalibration(
        camera_matrix=CAMERA_MATRIX,
        dist_coeffs=DIST_COEFFS,
        optimal_matrix=optimal_matrix,
        map1=map1,
        map2=map2,
        fx=float(optimal_matrix[0, 0]),
        cx=float(optimal_matrix[0, 2]),
        frame_size=frame_size,
    )
    _CALIBRATION_CACHE[frame_size] = calibration
    return calibration


def set_ping_pong_profile(profile: str) -> str:
    if profile not in PING_PONG_PRESETS:
        valid = ", ".join(sorted(PING_PONG_PRESETS))
        raise ValueError(f"Unknown ping-pong profile '{profile}'. Expected one of: {valid}")
    return profile


def _undistort_frame(frame: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    return cv2.remap(frame, calibration.map1, calibration.map2, cv2.INTER_LINEAR)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Shared low-level vision helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_mask(hsv: np.ndarray, ranges: tuple[HSVRange, ...]) -> np.ndarray:
    mask = cv2.inRange(hsv, ranges[0].lower, ranges[0].upper)
    for hsv_range in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, hsv_range.lower, hsv_range.upper))
    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _CLOSE_KERNEL, iterations=1)


def _contour_circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    area = cv2.contourArea(contour)
    return 4.0 * math.pi * area / (perimeter * perimeter)


def _estimate_distance_cm(radius_px: float, ball_type: str, calibration: CameraCalibration) -> float:
    apparent_diameter_px = max(radius_px * 2.0, 1e-6)
    return (BALL_DIAMETER_CM[ball_type] * calibration.fx) / apparent_diameter_px


def _bearing_deg_from_x(x_px: float, calibration: CameraCalibration) -> float:
    return math.degrees(math.atan2(x_px - calibration.cx, calibration.fx))


def _draw_ball_mask(mask: np.ndarray, ball: dict) -> None:
    cv2.circle(mask, (int(ball["x"]), int(ball["y"])), int(ball["radius"] * 1.2), 255, thickness=-1)


def _overlaps_existing(ball: dict, accepted: list[dict]) -> bool:
    for existing in accepted:
        centre_distance = math.hypot(ball["x"] - existing["x"], ball["y"] - existing["y"])
        if centre_distance <= max(ball["radius"], existing["radius"]):
            return True
    return False


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ping-pong ball detection  (UNCHANGED â€“ do not modify)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_ping_pong(
    hsv: np.ndarray,
    calibration: CameraCalibration,
    profile: str,
    accepted_ball_mask: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    profile_ranges = PING_PONG_PRESETS[profile]
    mask = _clean_mask(_build_mask(hsv, profile_ranges))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[dict] = []
    min_radius, max_radius = PING_PONG_RADIUS_RANGE_PX
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < PING_PONG_MIN_AREA or area > PING_PONG_MAX_AREA:
            continue
        circularity = _contour_circularity(contour)
        if circularity < PING_PONG_MIN_CIRCULARITY:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        ball = {
            "type": "PingPong",
            "x": int(cx),
            "y": int(cy),
            "radius": float(radius),
            "distance": float(_estimate_distance_cm(radius, "PingPong", calibration)),
            "angle": float(_bearing_deg_from_x(cx, calibration)),
            "confidence": float(min(1.0, 0.55 + 0.45 * circularity)),
            "predicted": False,
        }
        detections.append(ball)
        _draw_ball_mask(accepted_ball_mask, ball)

    return detections, mask


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Steel ball detection  (CLAHE + morph-close + contour + Kalman tracker)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_steel(
    hsv: np.ndarray,
    gray: np.ndarray,
    calibration: CameraCalibration,
    accepted_ball_mask: np.ndarray,
) -> tuple[list[dict], np.ndarray, list[tuple[int, int, int]], tuple[int, int] | None]:
    """Hybrid Hough + HSV steel detection with motion-gated Kalman tracking.

    Detection scans the full frame every time (no Kalman ROI cropping).
    Kalman is used only for smoothing a primary steel target after 3 frames
    of confirmed motion. Stationary checks always active.
    """
    frame_h, frame_w = hsv.shape[:2]

    # 1) Global Hough circles on blurred grayscale.
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)
    raw_circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=_STEEL_HOUGH_DP,
        minDist=_STEEL_HOUGH_MIN_DIST,
        param1=_STEEL_HOUGH_PARAM1,
        param2=_STEEL_HOUGH_PARAM2,
        minRadius=_STEEL_HOUGH_MIN_RADIUS,
        maxRadius=_STEEL_HOUGH_MAX_RADIUS,
    )

    hough_circles: list[tuple[int, int, int]] = []
    if raw_circles is not None:
        for x, y, r in raw_circles[0]:
            hough_circles.append((int(round(x)), int(round(y)), int(round(r))))

    # 2) HSV center-pixel validation for steel candidates.
    candidates: list[tuple[int, int, int, float]] = []
    steel_debug_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for cx, cy, radius in hough_circles:
        if cx < 0 or cy < 0 or cx >= frame_w or cy >= frame_h:
            continue

        # Reject circles centered on already-accepted ping-pong detections.
        if accepted_ball_mask[cy, cx] > 0:
            continue

        sat = int(hsv[cy, cx, 1])
        val = int(hsv[cy, cx, 2])
        if sat < _STEEL_CENTER_MAX_SAT and val > _STEEL_CENTER_MIN_VAL:
            candidates.append((cx, cy, radius, float(val)))
            cv2.circle(steel_debug_mask, (cx, cy), max(radius, 2), 255, thickness=2)

    # De-duplicate overlapping circle proposals while preserving multi-ball output.
    unique_candidates: list[tuple[int, int, int, float]] = []
    for cx, cy, radius, val in sorted(candidates, key=lambda c: c[2], reverse=True):
        duplicate = False
        for ux, uy, ur, _ in unique_candidates:
            if math.hypot(cx - ux, cy - uy) <= max(5.0, min(radius, ur) * 0.6):
                duplicate = True
                break
        if not duplicate:
            unique_candidates.append((cx, cy, radius, val))

    # 3) Kalman update phase and primary-target selection
    accepted: list[dict] = []
    kalman_smoothed_point: tuple[int, int] | None = None

    if unique_candidates:
        # Take the first (best) candidate for Kalman update
        cx, cy, radius, _ = unique_candidates[0]
        _STEEL_KALMAN.update(float(cx), float(cy))

        # Include all candidates in output (only primary gets Kalman smoothing)
        for i, (cx, cy, radius, _) in enumerate(unique_candidates):
            ball = {
                "type": "steel",
                "x": int(cx),
                "y": int(cy),
                "radius": float(radius),
                "distance": float(_estimate_distance_cm(float(radius), "steel", calibration)),
                "angle": float(_bearing_deg_from_x(float(cx), calibration)),
                "confidence": 0.75,
                "predicted": False,
            }
            accepted.append(ball)

        # Store primary target's smoothed centroid if moving mode enabled and confident
        if _STEEL_KALMAN.moving and _STEEL_KALMAN.frames_tracked >= 3:
            kalman_smoothed_point = (int(cx), int(cy))

    elif _STEEL_KALMAN.is_initialized:
        # No detection this frame: attempt Kalman prediction (coast)
        predicted = _STEEL_KALMAN.predict(coast_limit=2)
        if predicted is not None:
            px, py = predicted
            ball = {
                "type": "steel",
                "x": px,
                "y": py,
                "radius": 10.0,  # placeholder
                "distance": float(_estimate_distance_cm(10.0, "steel", calibration)),
                "angle": float(_bearing_deg_from_x(float(px), calibration)),
                "confidence": 0.5,
                "predicted": True,
            }
            accepted.append(ball)
            kalman_smoothed_point = (px, py)

    for ball in accepted:
        _draw_ball_mask(accepted_ball_mask, ball)

    return accepted, steel_debug_mask, hough_circles, kalman_smoothed_point


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Detection orchestration
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _prioritize_balls(detections: list[dict]) -> list[dict]:
    return sorted(
        detections,
        key=lambda item: (BALL_TYPE_PRIORITY[item["type"]], item["distance"], abs(item["angle"])),
    )


def _ball_repulsion_magnitude(
    ball: dict,
    obstacles: list[dict],
    frame_shape: tuple[int, int, int],
    calibration: CameraCalibration,
) -> float:
    frame_height = float(frame_shape[0])
    target_angle = float(ball["angle"])
    target_y = float(ball["y"])

    resultant = np.zeros(2, dtype=np.float32)
    for obstacle in obstacles:
        obstacle_angle = _bearing_deg_from_x(obstacle["x"], calibration)
        size_scale = min(obstacle["size"] / OBSTACLE_SIZE_FULL_SCALE, 1.0)
        bottom_bias = min(max(obstacle["y"] / frame_height, 0.0), 1.0)
        centrality = max(0.0, 1.0 - min(abs(obstacle_angle) / 90.0, 1.0))
        strength = (
            REPULSION_WEIGHT
            * _REPULSION_GLOBAL_SCALE
            * size_scale
            * (0.15 + 0.25 * bottom_bias)
            * (0.20 + 0.30 * centrality)
        )

        if float(obstacle["y"]) < target_y:
            strength *= OBSTACLE_BEHIND_TARGET_REPULSION_SCALE

        if obstacle["area"] >= OBSTACLE_LARGE_BLOB_AREA:
            strength *= 1.20

        if obstacle["area"] >= OBSTACLE_LARGE_BLOB_AREA and centrality >= 0.8 and bottom_bias >= 0.5:
            strength *= OPPONENT_LIKELY_REPULSION_BOOST

        if _angle_delta_deg(obstacle_angle, target_angle) <= CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG * 1.8:
            strength *= 0.65

        resultant += _vector_from_angle(obstacle_angle, strength)

    return float(np.clip(np.linalg.norm(resultant), 0.0, 1.0))


def _filter_repulsion_blocked_balls(
    detections: list[dict],
    obstacles: list[dict],
    frame_shape: tuple[int, int, int],
    calibration: CameraCalibration,
) -> list[dict]:
    if not BALL_REPULSION_FILTER_ENABLED or not detections or not obstacles:
        return detections

    scored: list[tuple[float, dict]] = []
    for ball in detections:
        scored.append((
            _ball_repulsion_magnitude(ball, obstacles, frame_shape, calibration),
            ball,
        ))

    kept: list[dict] = []
    for repulsion_mag, ball in scored:
        if repulsion_mag <= BALL_REPULSION_MAX_THRESHOLD:
            kept.append(ball)

    if len(kept) < BALL_REPULSION_MIN_KEEP:
        scored_sorted = sorted(scored, key=lambda item: item[0])
        for _, ball in scored_sorted:
            if ball not in kept:
                kept.append(ball)
            if len(kept) >= BALL_REPULSION_MIN_KEEP:
                break

    if len(kept) != len(detections):
        log.info(
            f"Repulsion filter kept {len(kept)}/{len(detections)} balls "
            f"(threshold={BALL_REPULSION_MAX_THRESHOLD:.2f})."
        )

    return _prioritize_balls(kept)


def _rebuild_ball_mask(frame_shape: tuple[int, int, int], detections: list[dict]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    for ball in detections:
        _draw_ball_mask(mask, ball)
    return mask


def detect_balls(frame: np.ndarray, ping_pong_profile: str = DEFAULT_PING_PONG_PROFILE) -> dict:
    try:
        from .vision_pipeline_K import detect_balls_pipeline
    except ImportError:
        from vision_pipeline_K import detect_balls_pipeline

    result = detect_balls_pipeline(frame, ping_pong_profile)

    try:
        detected_balls: list[dict] = result.get("detected_balls", [])
        if detected_balls:
            obstacles = detect_obstacles(result["frame"], result["ball_mask"])
            filtered_balls = _filter_repulsion_blocked_balls(
                detected_balls,
                obstacles,
                result["frame"].shape,
                result["calibration"],
            )
            if len(filtered_balls) != len(detected_balls):
                result["detected_balls"] = filtered_balls
                result["ball_mask"] = _rebuild_ball_mask(result["frame"].shape, filtered_balls)
    except Exception as exc:
        log.warning(f"Repulsion filtering skipped due to error: {exc}")

    if STREAM_DEBUG_FEED:
        start_debug_mjpeg_server()  # idempotent; no-op after first call
        push_debug_frame(annotate_debug_frame(result["frame"].copy(), result))

    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Debug stream: annotation, encoding, and MJPEG HTTP server
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def annotate_debug_frame(frame: np.ndarray, detection_result: dict) -> np.ndarray:
    """Draw Hough circles, Kalman-smoothed centroid, and ball labels.

    Parameters
    ----------
    frame            : BGR frame to annotate (will be copied internally).
    detection_result : dict returned by detect_balls().

    Returns
    -------
    Annotated BGR copy of the input frame.
    """
    out = frame.copy()

    # Raw Hough circles (full-frame scan) for verification.
    for cx, cy, radius in detection_result.get("steel_hough_circles", []):
        cv2.circle(out, (int(cx), int(cy)), max(int(radius), 2), (255, 0, 200), 1, cv2.LINE_AA)

    # Kalman trajectory (yellow polyline) only when moving confirmed
    trajectory = detection_result.get("steel_trajectory", [])
    if trajectory and len(trajectory) >= 2:
        traj_array = np.array(trajectory, dtype=np.int32)
        cv2.polylines(out, [traj_array], False, (0, 255, 255), 2, cv2.LINE_AA)

    # Kalman ROI (cyan box) for search area visualization
    roi = detection_result.get("steel_kalman_roi")
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1, cv2.LINE_AA)

    # Per-ball bounding circles and text labels
    for ball in detection_result.get("detected_balls", []):
        bx, by = int(ball["x"]), int(ball["y"])
        br = max(int(ball["radius"]), 4)
        is_predicted = ball.get("predicted", False)

        if ball["type"] == "steel":
            colour = (100, 100, 100) if is_predicted else (200, 200, 200)
        else:
            colour = (0, 200, 80)

        cv2.circle(out, (bx, by), br, colour, 2)

        if is_predicted:
            # Dashed inner circle indicates Kalman coast
            cv2.circle(out, (bx, by), br + 3, (0, 180, 255), 1)

        pred_tag = " [PRED]" if is_predicted else ""
        label = (
            f"{ball['type']}{pred_tag} | "
            f"{ball['angle']:.1f}deg | "
            f"{ball['distance']:.1f}cm"
        )
        cv2.putText(
            out, label, (bx - br, max(by - br - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA,
        )

    return out


def push_debug_frame(annotated: np.ndarray) -> None:
    """JPEG-encode an annotated frame and make it available to the MJPEG server.

    Non-blocking: drops the previous stale frame so the queue never grows
    beyond one entry.  The CV loop is never stalled by this call.
    """
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return
    jpeg_bytes: bytes = buf.tobytes()
    # Keep queue at maxsize=1: drop stale frame and push newest without blocking.
    try:
        if _debug_frame_queue.full():
            _debug_frame_queue.get_nowait()
        _debug_frame_queue.put_nowait(jpeg_bytes)
    except _queue.Full:
        pass
    except _queue.Empty:
        # A concurrent reader can race between full() and get_nowait().
        try:
            _debug_frame_queue.put_nowait(jpeg_bytes)
        except Exception:
            pass
    except Exception:
        pass


def start_debug_mjpeg_server(port: int = DEBUG_STREAM_PORT) -> None:
    """Start a daemon thread hosting a lightweight MJPEG HTTP server.

    Idempotent â€“ subsequent calls are no-ops.

    Endpoints
    ---------
    GET /         HTML page with embedded <img> tag pointing at /stream.
    GET /stream   multipart/x-mixed-replace MJPEG stream.

    The server reads JPEG frames from ``_debug_frame_queue`` with a 0.5 s
    timeout so the client connection is never blocked by the CV loop.
    """
    global _mjpeg_server_started
    with _mjpeg_server_lock:
        if _mjpeg_server_started:
            return
        _mjpeg_server_started = True

    def _server_loop() -> None:
        class MJPEGHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                # Silence the default per-request access log to keep stderr clean
                pass

            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=--jpgboundary",
                    )
                    self.end_headers()
                    while True:
                        try:
                            jpeg = _debug_frame_queue.get(timeout=0.5)
                        except Exception:
                            # No new frame yet â€“ write an empty chunk to keep TCP alive
                            try:
                                self.wfile.write(b"")
                            except Exception:
                                break
                            continue
                        try:
                            self.wfile.write(
                                b"--jpgboundary\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: "
                                + str(len(jpeg)).encode()
                                + b"\r\n\r\n"
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                else:
                    # Serve a minimal HTML viewer page
                    html = (
                        b"<html><body style='background:#111;margin:0'>"
                        b"<img src='/stream' style='max-width:100%;display:block;margin:auto'>"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)

        try:
            server = HTTPServer(("", port), MJPEGHandler)
            log.info(f"Debug MJPEG stream â†’ http://<pi-ip>:{port}")
            server.serve_forever()
        except Exception as exc:
            log.error(f"MJPEG server error on port {port}: {exc}")

    t = threading.Thread(target=_server_loop, name="MJPEGStream", daemon=True)
    t.start()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Obstacle detection
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def detect_obstacles(frame: np.ndarray, ball_mask: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    obstacle_mask = np.zeros_like(val)
    obstacle_mask[(val >= OBSTACLE_MIN_VALUE) & (val <= OBSTACLE_MAX_VALUE)] = 255
    obstacle_mask[sat < OBSTACLE_MIN_SATURATION] = 0
    obstacle_mask[ball_mask > 0] = 0

    top_ignore = int(frame.shape[0] * OBSTACLE_TOP_IGNORE_RATIO)
    obstacle_mask[:top_ignore, :] = 0
    obstacle_mask = _clean_mask(obstacle_mask)

    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < OBSTACLE_MIN_AREA or area > OBSTACLE_MAX_AREA:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        moment = cv2.moments(contour)
        if moment["m00"] == 0:
            centre_x = x + width // 2
            centre_y = y + height // 2
        else:
            centre_x = int(moment["m10"] / moment["m00"])
            centre_y = int(moment["m01"] / moment["m00"])
        obstacles.append(
            {
                "x": centre_x,
                "y": centre_y,
                "size": float(math.sqrt(area)),
                "area": float(area),
                "bbox": (x, y, width, height),
            }
        )
    return obstacles


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Navigation helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def is_ball_grabbable(ball: dict, calibration: CameraCalibration) -> bool:
    max_distance = GRAB_DISTANCE_CM[ball["type"]]
    return abs(ball["angle"]) <= GRAB_MAX_ABS_ANGLE_DEG and ball["distance"] <= max_distance


def _distance_to_strength(distance_cm: float) -> float:
    clipped = min(max(distance_cm, TARGET_DISTANCE_NEAR_CM), TARGET_DISTANCE_FAR_CM)
    span = max(TARGET_DISTANCE_FAR_CM - TARGET_DISTANCE_NEAR_CM, 1.0)
    strength = 1.0 - ((clipped - TARGET_DISTANCE_NEAR_CM) / span)
    return float(np.clip(strength, 0.0, 1.0))


def _vector_from_angle(angle_deg: float, strength: float) -> np.ndarray:
    radians = math.radians(angle_deg)
    return np.array([math.sin(radians), math.cos(radians)], dtype=np.float32) * strength


def _angle_delta_deg(a: float, b: float) -> float:
    """Return the smallest absolute angular difference in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def is_capture_corridor_blocked(
    target_ball: dict | None,
    obstacles: list[dict],
    frame_shape: tuple[int, int, int],
    calibration: CameraCalibration,
) -> bool:
    """True when a large central obstacle blocks near-capture approach lane."""
    if target_ball is None:
        return False

    frame_height = float(frame_shape[0])
    target_angle = float(target_ball["angle"])
    for obstacle in obstacles:
        if obstacle["area"] < OBSTACLE_BLOCKING_MIN_AREA:
            continue
        obstacle_angle = _bearing_deg_from_x(obstacle["x"], calibration)
        if _angle_delta_deg(obstacle_angle, target_angle) > OBSTACLE_BLOCKING_CENTRAL_DEG:
            continue
        bottom_bias = min(max(obstacle["y"] / frame_height, 0.0), 1.0)
        if bottom_bias < OBSTACLE_BLOCKING_BOTTOM_RATIO:
            continue
        return True
    return False


def compute_navigation_vector(
    target_ball: dict | None,
    obstacles: list[dict],
    frame_shape: tuple[int, int, int],
    calibration: CameraCalibration,
) -> dict | None:
    if target_ball is None:
        return None

    attraction_strength = max(MIN_TRACKING_MAGNITUDE, _distance_to_strength(target_ball["distance"]))
    resultant = _vector_from_angle(target_ball["angle"], attraction_strength * ATTRACTION_WEIGHT)

    frame_height = float(frame_shape[0])
    target_angle = float(target_ball["angle"])
    target_y = float(target_ball["y"])

    for obstacle in obstacles:
        obstacle_angle = _bearing_deg_from_x(obstacle["x"], calibration)
        size_scale = min(obstacle["size"] / OBSTACLE_SIZE_FULL_SCALE, 1.0)
        bottom_bias = min(max(obstacle["y"] / frame_height, 0.0), 1.0)
        centrality = max(0.0, 1.0 - min(abs(obstacle_angle) / 90.0, 1.0))
        strength = (
            REPULSION_WEIGHT
            * _REPULSION_GLOBAL_SCALE
            * size_scale
            * (0.15 + 0.25 * bottom_bias)
            * (0.20 + 0.30 * centrality)
        )

        # Obstacles behind target should repel less to allow collection near walls.
        if float(obstacle["y"]) < target_y:
            strength *= OBSTACLE_BEHIND_TARGET_REPULSION_SCALE

        # Large blobs are high-risk blockers and should repel more.
        if obstacle["area"] >= OBSTACLE_LARGE_BLOB_AREA:
            strength *= 1.20

        # Stronger penalty for likely dynamic/central threats.
        if obstacle["area"] >= OBSTACLE_LARGE_BLOB_AREA and centrality >= 0.8 and bottom_bias >= 0.5:
            strength *= OPPONENT_LIKELY_REPULSION_BOOST

        # Obstacles directly on target ray should mostly push sideways, not backward.
        if _angle_delta_deg(obstacle_angle, target_angle) <= CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG * 1.8:
            strength *= 0.65

        resultant -= _vector_from_angle(obstacle_angle, strength)

    lateral = float(resultant[0])
    forward = float(resultant[1])
    angle = math.degrees(math.atan2(lateral, forward))
    magnitude = float(np.clip(np.linalg.norm(resultant), 0.0, 1.0))

    # Gentle creep only near the ball and only when capture corridor is not blocked.
    if (
        float(target_ball["distance"]) <= CLOSE_RANGE_CREEP_DISTANCE_CM
        and not is_capture_corridor_blocked(target_ball, obstacles, frame_shape, calibration)
    ):
        magnitude = max(magnitude, CLOSE_RANGE_CREEP_MAGNITUDE)

    return {"angle": angle, "magnitude": magnitude}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# High-level frame processing (used by process_ball_frame and tests)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_ball_frame(
    frame: np.ndarray,
    ping_pong_profile: str = DEFAULT_PING_PONG_PROFILE,
    skip_anchor: tuple[float, float, float] | None = None,
) -> dict:
    if not hasattr(process_ball_frame, "_log_state"):
        process_ball_frame._log_state = None
        process_ball_frame._log_counter = 0
        process_ball_frame._last_log_counter = -10_000

    detection = detect_balls(frame, ping_pong_profile=ping_pong_profile)
    calibration: CameraCalibration = detection["calibration"]
    detected_balls: list[dict] = detection["detected_balls"]
    candidates = detected_balls
    if skip_anchor is not None and detected_balls:
        sx, sy, sr = skip_anchor
        filtered = []
        for ball in detected_balls:
            dx = float(ball["x"]) - sx
            dy = float(ball["y"]) - sy
            if (dx * dx + dy * dy) ** 0.5 > sr:
                filtered.append(ball)
        if filtered:
            candidates = filtered

    target_ball = candidates[0] if candidates else None
    obstacles = detect_obstacles(detection["frame"], detection["ball_mask"])

    locked_ball = ""
    move_target = None
    grab_ready = False
    is_blocked_in_zone = False

    if target_ball is not None:
        # 1) Y-axis pixel position check.
        ball_bottom_y = target_ball["y"] + target_ball["radius"]
        in_capture_zone = ball_bottom_y >= frame.shape[0] * CAPTURE_Y_THRESHOLD

        # 2) Alignment check.
        well_aligned = abs(float(target_ball["angle"])) <= CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG

        # Grab trigger only needs Y-zone + alignment.
        log.info(f"Ball '{target_ball['type']}': in_zone={in_capture_zone} aligned={well_aligned} angle={target_ball['angle']:.1f} distance={target_ball['distance']:.1f}cm")
        
        grab_ready = in_capture_zone and well_aligned
        is_blocked_in_zone = in_capture_zone and not grab_ready

        if grab_ready:
            locked_ball = target_ball["type"]
        else:
            move_target = compute_navigation_vector(target_ball, obstacles, detection["frame"].shape, calibration)

    process_ball_frame._log_counter += 1
    if target_ball is None:
        runtime_state = "SEARCHING"
        state_detail = "no targets"
    elif grab_ready:
        runtime_state = f"CAPTURING {target_ball['type']}"
        state_detail = f"dist={target_ball['distance']:.1f}cm"
    elif move_target is not None:
        runtime_state = f"TRACKING {target_ball['type']}"
        state_detail = f"angle={move_target['angle']:.1f}deg mag={move_target['magnitude']:.2f}"
    else:
        runtime_state = "TRACKING"
        state_detail = "nav=none"

    if (
        runtime_state != process_ball_frame._log_state
        or (process_ball_frame._log_counter - process_ball_frame._last_log_counter) >= _RUNTIME_LOG_HEARTBEAT
    ):
        log.info(f"CV runtime state: {runtime_state} | {state_detail}")
        process_ball_frame._log_state = runtime_state
        process_ball_frame._last_log_counter = process_ball_frame._log_counter

    return {
        "detected_balls": detected_balls,
        "target_ball": target_ball,
        "obstacles": obstacles,
        "move_target": move_target,
        "locked_ball": locked_ball,
        "grab_ready": grab_ready,
        "is_blocked_in_zone": is_blocked_in_zone,
        "camera_fx": calibration.fx,
        "camera_cx": calibration.cx,
    }


def benchmark_pipeline(frame: np.ndarray, runs: int = 30) -> dict:
    if runs <= 0:
        raise ValueError("runs must be positive")
    durations_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        process_ball_frame(frame)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    samples = np.array(durations_ms, dtype=np.float32)
    return {
        "runs": runs,
        "mean_ms": float(samples.mean()),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
    }

