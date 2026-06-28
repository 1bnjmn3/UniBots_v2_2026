from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HSVRange:
    lower: np.ndarray
    upper: np.ndarray


PING_PONG_PRESETS: dict[str, tuple[HSVRange, ...]] = {
    "orange": (
        HSVRange(np.array([5, 120, 120], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
    ),
    "white": (
        HSVRange(np.array([0, 0, 200], dtype=np.uint8), np.array([179, 50, 255], dtype=np.uint8)),
    ),
}

DEFAULT_PING_PONG_PROFILE = "orange"

PING_PONG_MIN_AREA = 100
PING_PONG_MAX_AREA = 80000
PING_PONG_MIN_CIRCULARITY = 0.55
PING_PONG_RADIUS_RANGE_PX = (7.0, 160.0)
PING_PONG_DIAMETER_CM: float = 4.0

STEEL_BALL_DIAMETER_CM: float = 2.0

HOUGH_DP          = 1.2
HOUGH_MIN_DIST    = 50
HOUGH_PARAM1      = 100
HOUGH_PARAM2      = 45
HOUGH_MIN_RADIUS  = 12
HOUGH_MAX_RADIUS  = 180
HOUGH_MATCH_DIST  = 50

STEEL_HOUGH_DP         = 1.2
STEEL_HOUGH_MIN_DIST   = 30
STEEL_HOUGH_PARAM1     = 100
STEEL_HOUGH_PARAM2     = 25
STEEL_HOUGH_MIN_RADIUS = 5
STEEL_HOUGH_MAX_RADIUS = 40
STEEL_OVERLAP_DIST     = 40

STEEL_MAX_SATURATION = 70
STEEL_MIN_VALUE = 20
STEEL_MAX_VALUE = 185
STEEL_MIN_AREA = 30
STEEL_MAX_AREA = 6000
STEEL_MIN_CIRCULARITY = 0.62
STEEL_RADIUS_RANGE_PX = (4.0, 42.0)
STEEL_MAX_MEAN_VALUE = 170.0

SPECULAR_MAX_SATURATION = 70
SPECULAR_MIN_VALUE = 190
SPECULAR_MIN_RATIO = 0.002
SPECULAR_MAX_RATIO = 0.38

OBSTACLE_MIN_AREA = 1200
OBSTACLE_MAX_AREA = 120000
OBSTACLE_MIN_VALUE = 15
OBSTACLE_MAX_VALUE = 210
OBSTACLE_MIN_SATURATION = 35
OBSTACLE_TOP_IGNORE_RATIO = 0.12
OBSTACLE_SIZE_FULL_SCALE = 180.0

FRAME_GAUSSIAN_BLUR = (5, 5)
MASK_OPEN_KERNEL = (3, 3)
MASK_CLOSE_KERNEL = (5, 5)

BALL_TYPE_PRIORITY = {"steel": 1, "PingPong": 0}
BALL_DIAMETER_CM = {"steel": 2.0, "PingPong": 4.0}

GRAB_MAX_ABS_ANGLE_DEG = 4.5
GRAB_DISTANCE_CM = {"steel": 14.0, "PingPong": 18.0}

TARGET_DISTANCE_NEAR_CM = 15.0
TARGET_DISTANCE_FAR_CM = 130.0
MIN_TRACKING_MAGNITUDE = 0.18

ATTRACTION_WEIGHT = 1.0
REPULSION_WEIGHT = 0.75

# â”€â”€ Obstacle-aware capture and APF tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CAPTURE_ALIGN_MAX_ABS_ANGLE_DEG = 8.5
CAPTURE_Y_THRESHOLD = 0.88
CAPTURE_BLOCKED_WAIT_S = 3.0
CAPTURE_SKIP_COOLDOWN_S = 1.0

CLOSE_RANGE_CREEP_DISTANCE_CM = 15.0
CLOSE_RANGE_CREEP_MAGNITUDE = 0.10

OBSTACLE_BLOCKING_MIN_AREA = 3500
OBSTACLE_BLOCKING_CENTRAL_DEG = 18.0
OBSTACLE_BLOCKING_BOTTOM_RATIO = 0.45
OBSTACLE_BEHIND_TARGET_REPULSION_SCALE = 0.60
OBSTACLE_LARGE_BLOB_AREA = 3500
OPPONENT_LIKELY_REPULSION_BOOST = 1.35

# Filter out candidate balls that are strongly repelled by nearby obstacles.
BALL_REPULSION_FILTER_ENABLED = True
BALL_REPULSION_MAX_THRESHOLD = 0.45
BALL_REPULSION_MIN_KEEP = 1

# â”€â”€ Steel ball CLAHE preprocessing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STEEL_CLAHE_CLIP_LIMIT: float = 2.0
STEEL_CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

# â”€â”€ Steel ball morphological close (fills specular highlight voids) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STEEL_MORPH_CLOSE_BIG_KERNEL: tuple = (11, 11)

# â”€â”€ Steel ball strict contour filters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STEEL_MIN_CIRCULARITY_STRICT: float = 0.75   # replaces relaxed STEEL_MIN_CIRCULARITY for new pipeline
STEEL_MIN_CONVEXITY: float = 0.82            # convexity = contourArea / convexHullArea

# â”€â”€ Steel specular-shadow pairing (new classical pipeline) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STEEL_SHADOW_MIN_VALUE: int = 8
STEEL_SHADOW_MAX_VALUE: int = 125
STEEL_PAIR_GLARE_ADJACENCY_PX: int = 7
STEEL_RELAXED_MIN_CONVEXITY: float = 0.72
STEEL_RELAXED_MIN_INERTIA: float = 0.18
STEEL_RELAXED_MIN_CIRCULARITY: float = 0.22

# â”€â”€ Kalman filter tracker constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
KALMAN_ROI_HALF_SIZE: int = 100         # initial ROI is 200Ã—200 px centred on prediction
KALMAN_MAX_LOST_FRAMES: int = 5         # frames to coast on Kalman prediction without a measurement
KALMAN_ROI_EXPAND_PX: int = 20          # expand ROI by this many px per consecutive lost frame

# â”€â”€ Debug stream â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEBUG_STREAM_PORT: int = 5000
DEBUG_TRAJECTORY_LENGTH: int = 8        # number of past predicted positions drawn as trajectory

