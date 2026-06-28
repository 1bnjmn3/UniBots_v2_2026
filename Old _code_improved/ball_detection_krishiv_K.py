"""
Ball Detection Module â€” UniBots Competition
============================================
Detects ALL visible ping-pong balls and steel ball bearings from
640Ã—480 BGR webcam frames using HSV colour filtering, contour
analysis and Hough circle detection.

Pipeline
--------
    Ping-pong : HSV colour mask  â†’  contour  â†’  Hough circle (confirm)
    Steel     : grey-region mask  â†’  Hough circle (primary)

Public API
----------
    detect_balls(frame, target="all") -> list[dict]
    BallTracker                       -> class  (persistent IDs across frames)
    detection_worker(in_q, out_q)     -> None   (multiprocessing target)

Output dict per ball
--------------------
    id        : int    â€” persistent tracking ID (assigned by BallTracker)
    type      : str    â€” "ping_pong" | "steel"
    x         : int    â€” pixel x centre in frame
    y         : int    â€” pixel y centre in frame
    radius    : float  â€” apparent pixel radius
    distance  : float  â€” estimated distance from lens (cm), pinhole model
    confirmed : bool   â€” True if Hough circle backs up the HSV detection
    method    : str    â€” "hsv+circle" | "hsv" | "grey+hough"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ===================================================================
# Configuration dataclasses
# ===================================================================

@dataclass(frozen=True, slots=True)
class HSVRange:
    """A single lower/upper HSV bound pair."""
    lower: np.ndarray
    upper: np.ndarray

@dataclass(frozen=True, slots=True)
class ContourLimits:
    """Area and circularity thresholds for contour filtering."""
    min_area: int      = 80
    max_area: int      = 80_000
    min_circularity: float = 0.55

# ===================================================================
# Camera / lens calibration  (update after checkerboard calibration)
# ===================================================================

CAMERA_MATRIX = np.array(
    [[564.98, 0.0, 303.59],
     [0.0, 565.87, 282.69],
     [0.0, 0.0, 1.0]], dtype=np.float64,
)
DIST_COEFFS = np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000], dtype=np.float64)


_NEW_CAM, _ROI = cv2.getOptimalNewCameraMatrix(
    CAMERA_MATRIX, DIST_COEFFS, (640, 480), alpha=0.0,
)
UNDISTORT_MAP1, UNDISTORT_MAP2 = cv2.initUndistortRectifyMap(
    CAMERA_MATRIX, DIST_COEFFS, None, _NEW_CAM, (640, 480), cv2.CV_16SC2,
)

FOCAL_PX: float = _NEW_CAM[0, 0]

# ===================================================================
# Ball physical properties
# ===================================================================

PING_PONG_DIAMETER_CM: float = 4.0   # 40 mm
STEEL_BALL_DIAMETER_CM: float = 2.0  # 20 mm

# ===================================================================
# HSV colour presets
# ===================================================================

PING_PONG_PRESETS: dict[str, list[HSVRange]] = {
    "orange": [HSVRange(np.array([5, 120, 120]),  np.array([25, 255, 255]))],
    "white":  [HSVRange(np.array([0,   0, 200]),  np.array([179, 50, 255]))],
}

# >>> Change this before the match <<<
PING_PONG_COLOUR: str = "orange"
PING_PONG_RANGES: list[HSVRange] = PING_PONG_PRESETS[PING_PONG_COLOUR]

# Steel grey-region thresholds (mask first, then shape).
STEEL_MAX_SATURATION: int = 70
STEEL_MIN_VALUE: int = 20
STEEL_MAX_VALUE: int = 255

# Contour limits (ping-pong only; steel uses Hough directly)
PING_PONG_CONTOUR = ContourLimits(min_area=100,  max_area=80_000, min_circularity=0.55)

# ===================================================================
# Hough Circle detection params â€” ping-pong (confirmation only)
# ===================================================================

HOUGH_DP          = 1.2
HOUGH_MIN_DIST    = 50
HOUGH_PARAM1      = 100
HOUGH_PARAM2      = 45
HOUGH_MIN_RADIUS  = 12
HOUGH_MAX_RADIUS  = 180
HOUGH_MATCH_DIST  = 50

# ===================================================================
# Hough Circle detection params â€” steel (primary detection)
# Tuned from Raef's working values (param2=18) but made stricter.
# ===================================================================

STEEL_HOUGH_DP         = 1.2
STEEL_HOUGH_MIN_DIST   = 30    # steel balls can be close together
STEEL_HOUGH_PARAM1     = 100
STEEL_HOUGH_PARAM2     = 25    # lower than ping-pong (steel has strong edges)
STEEL_HOUGH_MIN_RADIUS = 5     # steel balls are small
STEEL_HOUGH_MAX_RADIUS = 40    # 20mm ball won't be huge in frame
STEEL_OVERLAP_DIST     = 40    # reject if this close to a ping-pong detection

# ===================================================================
# Reusable kernels
# ===================================================================

_BLUR_KSIZE = (7, 7)
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ===================================================================
# Internal helpers
# ===================================================================

def undistort(frame: np.ndarray) -> np.ndarray:
    return cv2.remap(frame, UNDISTORT_MAP1, UNDISTORT_MAP2, cv2.INTER_LINEAR)


def build_mask(hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
    mask = cv2.inRange(hsv, ranges[0].lower, ranges[0].upper)
    for r in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r.lower, r.upper))
    return mask


def morph_clean(mask: np.ndarray) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=1)
    return mask


def all_contours(
    mask: np.ndarray,
    limits: ContourLimits,
) -> list[tuple[np.ndarray, float, tuple[int, int], float]]:
    """Return all sufficiently-circular blobs sorted largest-first.
    Each element: (contour, area, (cx, cy), radius).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[tuple[np.ndarray, float, tuple[int, int], float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < limits.min_area or area > limits.max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = 4.0 * math.pi * area / (perim * perim)
        if circ < limits.min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        results.append((cnt, area, (int(cx), int(cy)), radius))
    results.sort(key=lambda c: c[1], reverse=True)
    return results


def estimate_distance(apparent_diam_px: float, real_diam_cm: float) -> float:
    """Pinhole-model distance: z = (D_real * f) / D_apparent."""
    if apparent_diam_px <= 0:
        return float("inf")
    return (real_diam_cm * FOCAL_PX) / apparent_diam_px


# ===================================================================
# Hough Circle detection
# ===================================================================

def find_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )
    if circles is None:
        return []
    raw = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    out: list[tuple[int, int, int]] = []
    edges = cv2.Canny(gray, HOUGH_PARAM1 // 2, HOUGH_PARAM1) if raw else None
    h, w = gray.shape[:2]
    for cx, cy, cr in raw:
        hits = 0
        samples = 24
        for i in range(samples):
            angle = 2.0 * math.pi * i / samples
            px = int(cx + cr * math.cos(angle))
            py = int(cy + cr * math.sin(angle))
            if 0 <= px < w and 0 <= py < h and edges[py, px] > 0:
                hits += 1
        if hits / samples >= 0.35:
            out.append((cx, cy, cr))
    out.sort(key=lambda c: c[2], reverse=True)
    return out


def _match_circle_to_contour(
    contour_centre: tuple[int, int],
    circles: list[tuple[int, int, int]],
) -> Optional[tuple[int, int, int]]:
    """Find the Hough circle closest to a contour centre (within threshold)."""
    cx, cy = contour_centre
    best, best_d = None, HOUGH_MATCH_DIST
    for hx, hy, hr in circles:
        d = math.hypot(hx - cx, hy - cy)
        if d < best_d:
            best_d = d
            best = (hx, hy, hr)
    return best


# ===================================================================
# Steel ball Hough detection
# ===================================================================

def find_steel_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Hough circles tuned specifically for small steel ball bearings."""
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=STEEL_HOUGH_DP,
        minDist=STEEL_HOUGH_MIN_DIST,
        param1=STEEL_HOUGH_PARAM1,
        param2=STEEL_HOUGH_PARAM2,
        minRadius=STEEL_HOUGH_MIN_RADIUS,
        maxRadius=STEEL_HOUGH_MAX_RADIUS,
    )
    if circles is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in circles[0]]


# ===================================================================
# Steel validation helpers
# ===================================================================

def _is_grey_region(hsv: np.ndarray, cx: int, cy: int, r: int) -> bool:
    """Check that the circular region has low saturation (achromatic/grey).
    Steel balls are metallic grey â€” reject coloured regions.
    """
    h, w = hsv.shape[:2]
    y1 = max(cy - r, 0)
    y2 = min(cy + r, h)
    x1 = max(cx - r, 0)
    x2 = min(cx + r, w)
    if y2 <= y1 or x2 <= x1:
        return False
    roi = hsv[y1:y2, x1:x2]
    mean_s = float(np.mean(roi[:, :, 1]))
    mean_v = float(np.mean(roi[:, :, 2]))
    return mean_s < STEEL_MAX_SATURATION and STEEL_MIN_VALUE < mean_v < STEEL_MAX_VALUE


# ===================================================================
# Core detection â€” returns ALL detected balls
# ===================================================================

def detect_balls(
    frame: np.ndarray,
    target: str = "all",
) -> list[dict]:
    """Detect all balls in a 640x480 BGR frame.

    Pipeline:
        Ping-pong : HSV mask â†’ contour â†’ Hough circle (confirm)
        Steel     : grey-region mask â†’ Hough circle (primary)

    Parameters
    ----------
    frame  : BGR uint8 ndarray, 640x480.
    target : "all" | "ping_pong" | "steel"

    Returns
    -------
    list[dict], each with keys:
        type      : str    â€” "ping_pong" | "steel"
        x         : int    â€” pixel x centre
        y         : int    â€” pixel y centre
        radius    : float  â€” apparent pixel radius
        distance  : float  â€” estimated cm from camera
        confirmed : bool   â€” Hough circle backed
        method    : str    â€” detection method description
    Note: 'id' is NOT set here â€” use BallTracker.update() to assign IDs.
    """
    frame = undistort(frame)
    blurred = cv2.GaussianBlur(frame, _BLUR_KSIZE, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    circles = find_circles(gray)
    detections: list[dict] = []
    pp_centres: list[tuple[int, int]] = []  # used to reject steel overlaps

    # --- Ping-pong: HSV â†’ contour â†’ Hough circle (confirm) ---
    if target in ("all", "ping_pong"):
        pp_mask = morph_clean(build_mask(hsv, PING_PONG_RANGES))
        for cnt, area, (cx, cy), radius in all_contours(pp_mask, PING_PONG_CONTOUR):
            confirmed = _match_circle_to_contour((cx, cy), circles) is not None
            method = "hsv+circle" if confirmed else "hsv"
            detections.append({
                "type": "ping_pong",
                "x": cx,
                "y": cy,
                "radius": round(radius, 1),
                "distance": round(estimate_distance(radius * 2, PING_PONG_DIAMETER_CM), 1),
                "confirmed": confirmed,
                "method": method,
            })
            pp_centres.append((cx, cy))

    # --- Steel: grey-region mask (primary) â†’ Hough circle ---
    # Steel balls have weak colour cues and strong reflections; mask achromatic
    # regions first, then run shape detection on the masked grayscale image.
    if target in ("all", "steel"):
        steel_mask = np.zeros_like(gray)
        steel_mask[(hsv[:, :, 1] < STEEL_MAX_SATURATION) & (hsv[:, :, 2] > STEEL_MIN_VALUE) & (hsv[:, :, 2] < STEEL_MAX_VALUE)] = 255
        steel_mask = morph_clean(steel_mask)

        steel_gray = cv2.bitwise_and(gray, gray, mask=steel_mask)
        steel_blur = cv2.GaussianBlur(steel_gray, (9, 9), 2)
        steel_circles = find_steel_circles(steel_blur)
        for scx, scy, sr in steel_circles:
            # Reject if too close to a ping-pong detection
            if any(math.hypot(scx - px, scy - py) < STEEL_OVERLAP_DIST
                   for px, py in pp_centres):
                continue

            # Final center-pixel validation in HSV to suppress false positives.
            if not (0 <= scx < hsv.shape[1] and 0 <= scy < hsv.shape[0]):
                continue
            sat = int(hsv[scy, scx, 1])
            val = int(hsv[scy, scx, 2])
            if not (sat < STEEL_MAX_SATURATION and val > STEEL_MIN_VALUE):
                continue

            detections.append({
                "type": "steel",
                "x": scx,
                "y": scy,
                "radius": round(float(sr), 1),
                "distance": round(estimate_distance(sr * 2, STEEL_BALL_DIAMETER_CM), 1),
                "confirmed": True,
                "method": "grey+hough",
            })

    return detections


# ===================================================================
# BallTracker â€” persistent IDs across frames via nearest-neighbour
# ===================================================================

class BallTracker:
    """Assigns stable integer IDs to balls across consecutive frames.

    Uses greedy nearest-neighbour matching on (x, y) pixel position.
    A ball that disappears for more than `max_lost` frames loses its ID.
    """

    def __init__(self, max_lost: int = 15, match_radius: int = 80):
        self._next_id: int = 1
        self._max_lost: int = max_lost
        self._match_radius: int = match_radius
        # tracked[id] = {
        #   "type", "x", "y", "radius", "distance", "confirmed", "method", "id", "lost",
        #   "kalman", "pred_x", "pred_y"
        # }
        self._tracked: dict[int, dict] = {}

    def _create_kalman(self, x: float, y: float):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        # CRITICAL: initialize both statePre and statePost to first detection.
        kf.statePre[0, 0] = np.float32(x)
        kf.statePre[1, 0] = np.float32(y)
        kf.statePost[0, 0] = np.float32(x)
        kf.statePost[1, 0] = np.float32(y)
        kf.statePre[2, 0] = np.float32(0.0)
        kf.statePre[3, 0] = np.float32(0.0)
        kf.statePost[2, 0] = np.float32(0.0)
        kf.statePost[3, 0] = np.float32(0.0)
        return kf

    def reset(self):
        self._tracked.clear()
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """Match new detections against tracked balls, assign IDs.

        Returns a new list of dicts â€” same fields as detect_balls() but
        with an 'id' key added.
        """
        used_track_ids: set[int] = set()
        used_det_indices: set[int] = set()
        result: list[dict] = []

        # Predict each existing track once per frame for robust matching.
        for tid, tb in self._tracked.items():
            pred = tb["kalman"].predict()
            tb["pred_x"] = float(pred[0, 0])
            tb["pred_y"] = float(pred[1, 0])

        # Build distance pairs: (dist, track_id, det_index)
        pairs: list[tuple[float, int, int]] = []
        for tid, tb in self._tracked.items():
            for di, det in enumerate(detections):
                d = math.hypot(det["x"] - tb["pred_x"], det["y"] - tb["pred_y"])
                if d < self._match_radius and det["type"] == tb["type"]:
                    pairs.append((d, tid, di))

        pairs.sort(key=lambda p: p[0])

        # Greedy match
        for d, tid, di in pairs:
            if tid in used_track_ids or di in used_det_indices:
                continue
            used_track_ids.add(tid)
            used_det_indices.add(di)
            det = detections[di]
            measurement = np.array([[np.float32(det["x"])], [np.float32(det["y"])]])
            corrected = self._tracked[tid]["kalman"].correct(measurement)
            sx = int(round(float(corrected[0, 0])))
            sy = int(round(float(corrected[1, 0])))
            entry = {**det, "id": tid, "x": sx, "y": sy}
            self._tracked[tid] = {
                **entry,
                "lost": 0,
                "kalman": self._tracked[tid]["kalman"],
                "pred_x": float(sx),
                "pred_y": float(sy),
            }
            result.append(entry)

        # New balls â€” unmatched detections get fresh IDs
        for di, det in enumerate(detections):
            if di in used_det_indices:
                continue
            tid = self._next_id
            self._next_id += 1
            kf = self._create_kalman(float(det["x"]), float(det["y"]))
            entry = {**det, "id": tid}
            self._tracked[tid] = {
                **entry,
                "lost": 0,
                "kalman": kf,
                "pred_x": float(det["x"]),
                "pred_y": float(det["y"]),
            }
            # Prevent new tracks from being aged in the same update cycle.
            used_track_ids.add(tid)
            result.append(entry)

        # Age out unmatched tracked balls
        lost_ids = []
        for tid in self._tracked:
            if tid not in used_track_ids:
                self._tracked[tid]["lost"] += 1
                if self._tracked[tid]["lost"] > self._max_lost:
                    lost_ids.append(tid)
        for tid in lost_ids:
            del self._tracked[tid]

        return result


# ===================================================================
# Multiprocessing worker
# ===================================================================

def detection_worker(frame_queue, result_queue) -> None:
    """Pull frames â†’ detect â†’ push results.  Send None to stop."""
    tracker = BallTracker()
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        raw = detect_balls(frame)
        result = tracker.update(raw)
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        result_queue.put(result)


# ===================================================================
# Self-test / benchmark
# ===================================================================

if __name__ == "__main__":
    from multiprocessing import Process, Queue
    import time

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    fq: Queue = Queue(maxsize=2)
    rq: Queue = Queue(maxsize=2)

    proc = Process(target=detection_worker, args=(fq, rq))
    proc.start()

    fq.put(dummy)
    rq.get(timeout=5)

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        fq.put(dummy)
        rq.get(timeout=5)
    ms = (time.perf_counter() - t0) / N * 1000

    print(f"Latency : {ms:.1f} ms/frame")
    print(f"FPS     : {1000 / ms:.0f}")

    fq.put(None)
    proc.join(timeout=5)
    print("OK" if not proc.is_alive() else "Worker hung")

