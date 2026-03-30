from __future__ import annotations

import cv2
import numpy as np
import math
from collections import deque

try:
    from .config_K import (
        STEEL_HOUGH_DP,
        STEEL_HOUGH_MIN_DIST,
        STEEL_HOUGH_PARAM1,
        STEEL_HOUGH_PARAM2,
        STEEL_HOUGH_MIN_RADIUS,
        STEEL_HOUGH_MAX_RADIUS,
        STEEL_OVERLAP_DIST,
        STEEL_BALL_DIAMETER_CM,
    )
except ImportError:
    from config_K import (
        STEEL_HOUGH_DP,
        STEEL_HOUGH_MIN_DIST,
        STEEL_HOUGH_PARAM1,
        STEEL_HOUGH_PARAM2,
        STEEL_HOUGH_MIN_RADIUS,
        STEEL_HOUGH_MAX_RADIUS,
        STEEL_OVERLAP_DIST,
        STEEL_BALL_DIAMETER_CM,
    )


class FloorColorTracker:
    """Track dominant floor color across frames and handle variations."""
    
    def __init__(self, history_frames: int = 3):
        self.history = deque(maxlen=history_frames)
    
    def _get_dominant_color_hsv(self, hsv: np.ndarray, roi_ratio: float = 0.6) -> np.ndarray | None:
        """Get dominant color from lower portion of frame (floor region)."""
        h, w = hsv.shape[:2]
        start_row = int(h * (1.0 - roi_ratio))
        roi = hsv[start_row:, :]
        
        if roi.size == 0:
            return None
        
        # Get median HSV from floor region
        h_med = int(np.median(roi[:, :, 0]))
        s_med = int(np.median(roi[:, :, 1]))
        v_med = int(np.median(roi[:, :, 2]))
        
        return np.array([h_med, s_med, v_med], dtype=np.uint8)
    
    def update(self, hsv: np.ndarray) -> np.ndarray:
        """Update floor color estimate from current frame."""
        dominant = self._get_dominant_color_hsv(hsv)
        if dominant is not None:
            self.history.append(dominant)
        
        if len(self.history) == 0:
            return np.array([100, 50, 100], dtype=np.uint8)  # fallback neutral floor
        
        # Return median of history (consensus floor color)
        history_array = np.array(list(self.history), dtype=np.float32)
        consensus = np.median(history_array, axis=0)
        return np.array(consensus, dtype=np.uint8)
    
    def get_floor_mask(self, hsv: np.ndarray, floor_color: np.ndarray, strict: bool = True) -> np.ndarray:
        """Create binary mask of pixels matching floor color (strict: small tolerance)."""
        if strict:
            h_tol, s_tol, v_tol = 15, 40, 35  # strict tolerances
        else:
            h_tol, s_tol, v_tol = 25, 60, 50  # lenient tolerances
        
        lower = np.array([
                (int(floor_color[0]) - h_tol) % 180,
                max(0, int(floor_color[1]) - s_tol),
                max(0, int(floor_color[2]) - v_tol)
        ], dtype=np.uint8)
        
        upper = np.array([
                (int(floor_color[0]) + h_tol) % 180,
                min(255, int(floor_color[1]) + s_tol),
                min(255, int(floor_color[2]) + v_tol)
        ], dtype=np.uint8)
        
        # Handle hue wraparound at 0/180
        if lower[0] <= upper[0]:
            mask = cv2.inRange(hsv, lower, upper)
        else:
            mask_a = cv2.inRange(hsv, np.array([0, lower[1], lower[2]], dtype=np.uint8),
                                np.array([upper[0], upper[1], upper[2]], dtype=np.uint8))
            mask_b = cv2.inRange(hsv, np.array([lower[0], lower[1], lower[2]], dtype=np.uint8),
                                np.array([179, upper[1], upper[2]], dtype=np.uint8))
            mask = cv2.bitwise_or(mask_a, mask_b)
        
        return mask


def find_steel_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Hough circles tuned for small steel ball bearings."""
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


def extract_steel_floor_disturbance(
    *,
    frame: np.ndarray,
    hsv: np.ndarray,
    gray: np.ndarray,
    pp_mask: np.ndarray,
    pp_centres: list[tuple[int, int]],
    floor_tracker: FloorColorTracker,
    estimate_distance_fn,
    bearing_fn,
    calibration,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Extract steel balls using floor-disturbance method:
    1. Determine floor color (with history tracking)
    2. Find disturbances (interruptions to floor color)
    3. Apply Hough circles on disturbance mask
    4. Filter by diameter < 2cm
    5. Reject if overlaps with ping-pong
    """
    
    # Update floor color estimate and get disturbance mask
    floor_color = floor_tracker.update(hsv)
    floor_mask = floor_tracker.get_floor_mask(hsv, floor_color, strict=True)
    
    # Disturbances = places that DON'T match floor color
    disturbance_mask = cv2.bitwise_not(floor_mask)
    
    # Remove ping-pong balls from disturbance mask (they're already handled)
    pp_shield = cv2.dilate(pp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    disturbance_mask = cv2.bitwise_and(disturbance_mask, cv2.bitwise_not(pp_shield))
    
    # Clean the mask (morph operations)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disturbance_mask = cv2.morphologyEx(disturbance_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    disturbance_mask = cv2.morphologyEx(disturbance_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Hough circle detection on disturbance mask
    steel_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    steel_circles = find_steel_circles(steel_blur)
    
    detections = []
    steel_binary_mask = np.zeros_like(disturbance_mask)
    
    # Plausible distance bounds for a steel ball on the playing field.
    # Derived from the pinhole model: distance_cm = (real_radius_cm * fx) / radius_px
    # With fx ≈ 565 px and STEEL_BALL_DIAMETER_CM = 2.0 cm (real_radius = 1.0 cm):
    #   STEEL_HOUGH_MAX_RADIUS = 40 px  →  distance ≈ 565/40 = 14 cm  (ball nearly at claw)
    #   STEEL_HOUGH_MIN_RADIUS =  5 px  →  distance ≈ 565/ 5 = 113 cm (far end of field)
    # Anything below 8 cm would be physically inside the robot chassis (false positive).
    # Anything above 150 cm is beyond reliable detection range (noise rejection).
    _STEEL_MIN_DISTANCE_CM = 8.0
    _STEEL_MAX_DISTANCE_CM = 150.0

    for scx, scy, sr in steel_circles:
        # Filter 1: Reject circles inconsistent with a real 2 cm steel ball.
        # estimate_distance_fn uses the pinhole model with the actual camera calibration,
        # so this is unit-correct and accounts for the real focal length.
        estimated_distance_cm = estimate_distance_fn(float(sr), "steel", calibration)
        if not (_STEEL_MIN_DISTANCE_CM <= estimated_distance_cm <= _STEEL_MAX_DISTANCE_CM):
            continue
        
        # Filter 2: Must have support in disturbance mask (circle center in disturbance region)
        if disturbance_mask[scy, scx] == 0:
            continue
        
        # Filter 3: Reject if too close to a ping-pong detection
        if any(math.hypot(scx - px, scy - py) < STEEL_OVERLAP_DIST for px, py in pp_centres):
            continue
        
        detections.append({
            "type": "steel",
            "x": scx,
            "y": scy,
            "radius": round(float(sr), 1),
            "distance": round(estimate_distance_fn(float(sr), "steel", calibration), 1),
            "angle": float(bearing_fn(float(scx), calibration)),
            "confidence": 0.75,
            "predicted": False,
            "confirmed": True,  # All pass the filters
            "method": "floor_disturbance+hough",
        })
        
        # Mark detected steel regions
        cv2.circle(steel_binary_mask, (scx, scy), sr, 255, thickness=-1)
    
    return detections, steel_binary_mask, disturbance_mask