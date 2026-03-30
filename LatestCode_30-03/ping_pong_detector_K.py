from __future__ import annotations

import cv2
import numpy as np
import math

try:
    from .config_K import (
        PING_PONG_MAX_AREA,
        PING_PONG_MIN_AREA,
        PING_PONG_MIN_CIRCULARITY,
        PING_PONG_PRESETS,
        PING_PONG_DIAMETER_CM,
        HOUGH_DP,
        HOUGH_MIN_DIST,
        HOUGH_PARAM1,
        HOUGH_PARAM2,
        HOUGH_MIN_RADIUS,
        HOUGH_MAX_RADIUS,
        HOUGH_MATCH_DIST,
    )
except ImportError:
    from config_K import (
        PING_PONG_MAX_AREA,
        PING_PONG_MIN_AREA,
        PING_PONG_MIN_CIRCULARITY,
        PING_PONG_PRESETS,
        PING_PONG_DIAMETER_CM,
        HOUGH_DP,
        HOUGH_MIN_DIST,
        HOUGH_PARAM1,
        HOUGH_PARAM2,
        HOUGH_MIN_RADIUS,
        HOUGH_MAX_RADIUS,
        HOUGH_MATCH_DIST,
    )


def build_mask(hsv: np.ndarray, ranges: list) -> np.ndarray:
    """Build HSV mask from one or more ranges."""
    mask = cv2.inRange(hsv, ranges[0].lower, ranges[0].upper)
    for r in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r.lower, r.upper))
    return mask


def morph_clean(mask: np.ndarray) -> np.ndarray:
    """Apply morphological open+close to clean mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def all_contours(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_circularity: float,
) -> list[tuple[np.ndarray, float, tuple[int, int], float]]:
    """Return all sufficiently-circular blobs sorted largest-first.
    Each element: (contour, area, (cx, cy), radius).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        perim = float(cv2.arcLength(cnt, True))
        if perim == 0:
            continue
        circ = 4.0 * math.pi * area / (perim * perim)
        if circ < min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        results.append((cnt, area, (int(cx), int(cy)), float(radius)))
    results.sort(key=lambda c: c[1], reverse=True)
    return results


def find_circles(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Find circles using Hough gradient method (confirmation only)."""
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
    out = []
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
) -> tuple[int, int, int] | None:
    """Find the Hough circle closest to a contour centre."""
    cx, cy = contour_centre
    best, best_d = None, HOUGH_MATCH_DIST
    for hx, hy, hr in circles:
        d = math.hypot(hx - cx, hy - cy)
        if d < best_d:
            best_d = d
            best = (hx, hy, hr)
    return best


def extract_ping_pong(
    *,
    frame: np.ndarray,
    hsv: np.ndarray,
    gray: np.ndarray,
    profile: str,
    estimate_distance_fn,
    bearing_fn,
    calibration,
) -> tuple[list[dict], list[tuple[int, int]], np.ndarray]:
    """Extract ping-pong balls using HSV mask + contour + Hough confirmation (exact GT1 logic)."""

    ranges = PING_PONG_PRESETS[profile]
    
    circles = find_circles(gray)
    detections = []
    pp_centres = []
    
    pp_mask = morph_clean(build_mask(hsv, ranges))
    for cnt, area, (cx, cy), radius in all_contours(
        pp_mask,
        PING_PONG_MIN_AREA,
        PING_PONG_MAX_AREA,
        PING_PONG_MIN_CIRCULARITY,
    ):
        confirmed = _match_circle_to_contour((cx, cy), circles) is not None
        method = "hsv+circle" if confirmed else "hsv"
        detections.append({
            "type": "PingPong",
            "x": cx,
            "y": cy,
            "radius": round(radius, 1),
            "distance": round(estimate_distance_fn(radius, "PingPong", calibration), 1),
            "angle": float(bearing_fn(cx, calibration)),
            "confidence": 0.85 if confirmed else 0.70,
            "predicted": False,
            "confirmed": confirmed,
            "method": method,
        })
        pp_centres.append((cx, cy))
    
    return detections, pp_centres, pp_mask

