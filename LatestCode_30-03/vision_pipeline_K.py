from __future__ import annotations

import cv2
import numpy as np

try:
    from . import ball_detector_runtime_K as bdr
    from .AprilTagNavigator_M import AprilTagNavigator
    from .ping_pong_detector_K import extract_ping_pong
    from .steel_ball_detector_K import extract_steel_floor_disturbance, FloorColorTracker
    from .configTag_M import (
        APRILTAG_CAMERA_PARAMS,
        APRILTAG_FAMILY,
        APRILTAG_FRAME_SIZE,
        APRILTAG_TAG_SIZE_M,
        APRILTAG_TARGET_IDS,
    )
except ImportError:
    import ball_detector_runtime_K as bdr
    from AprilTagNavigator_M import AprilTagNavigator
    from ping_pong_detector_K import extract_ping_pong
    from steel_ball_detector_K import extract_steel_floor_disturbance, FloorColorTracker
    from configTag_M import (
        APRILTAG_CAMERA_PARAMS,
        APRILTAG_FAMILY,
        APRILTAG_FRAME_SIZE,
        APRILTAG_TAG_SIZE_M,
        APRILTAG_TARGET_IDS,
    )


# Module-level floor tracker (persistent across frames)
_FLOOR_TRACKER = FloorColorTracker(history_frames=3)
_APRILTAG_NAVIGATOR = None


def _get_apriltag_navigator() -> AprilTagNavigator:
    global _APRILTAG_NAVIGATOR
    if _APRILTAG_NAVIGATOR is None:
        _APRILTAG_NAVIGATOR = AprilTagNavigator(
            target_tag_ids=APRILTAG_TARGET_IDS,
            camera_params=APRILTAG_CAMERA_PARAMS,
            tag_size_m=APRILTAG_TAG_SIZE_M,
            frame_size=APRILTAG_FRAME_SIZE,
            tag_families=APRILTAG_FAMILY,
        )
    return _APRILTAG_NAVIGATOR


def _build_apriltag_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Use AprilTagNavigator results to build a filled shield mask."""
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    try:
        nav = _get_apriltag_navigator()
        nav_out = nav.process_frame(frame_bgr)
    except Exception:
        return mask

    all_tags = nav_out.get("all_tags", [])
    if not all_tags:
        return mask

    fx = float(APRILTAG_CAMERA_PARAMS[0]) if APRILTAG_CAMERA_PARAMS else 560.0
    h, w = frame_bgr.shape[:2]

    for tag in all_tags:
        cx = int(tag.get("centre_x", 0))
        cy = int(tag.get("centre_y", 0))
        z_cm = float(tag.get("z_cm", 0.0))
        z_m = max(z_cm / 100.0, 0.05)

        # Estimate tag side in pixels from depth and draw a filled square shield.
        side_px = int(max(16.0, min(220.0, (APRILTAG_TAG_SIZE_M * fx) / z_m)))
        half = side_px // 2
        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w - 1)
        y2 = min(cy + half, h - 1)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Expand shield slightly so borders/highlights do not leak into steel stage.
    if np.any(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_balls_pipeline(frame: np.ndarray, ping_pong_profile: str = "orange") -> dict:
    """
    Complete ball detection pipeline:
    1. Preprocess frame (undistort, blur, convert to HSV/grayscale)
    2. Detect ping-pong balls (HSV mask + contour + Hough confirmation)
    3. Shield ping-pong regions to prevent false steel detections
    4. Detect steel balls (floor-disturbance method with Hough circles)
    5. Merge and return all detections
    """
    
    if frame is None or frame.ndim != 3:
        raise ValueError("detect_balls expects a BGR frame with shape (H, W, 3)")

    calibration = bdr._get_calibration(frame.shape)
    undistorted = bdr._undistort_frame(frame, calibration)

    blurred = cv2.GaussianBlur(undistorted, bdr.FRAME_GAUSSIAN_BLUR, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Stage 1: Detect ping-pong balls
    ping_pong, pp_centres, pp_mask = extract_ping_pong(
        frame=undistorted,
        hsv=hsv,
        gray=gray,
        profile=ping_pong_profile,
        estimate_distance_fn=bdr._estimate_distance_cm,
        bearing_fn=bdr._bearing_deg_from_x,
        calibration=calibration,
    )

    # Stage 2: Detect AprilTags and add them to the steel shield mask.
    apriltag_mask = _build_apriltag_mask(undistorted)

    # Keep ping-pong shielding based on raw visual detections, independent of any
    # downstream reachability filtering. This prevents ping-pong blobs from leaking
    # into steel detection even if they are later excluded from target selection.
    pre_steel_shield_mask = cv2.bitwise_or(pp_mask, apriltag_mask)

    # Stage 3: Detect steel balls (using floor-disturbance method)
    steel, steel_mask, disturbance_mask = extract_steel_floor_disturbance(
        frame=undistorted,
        hsv=hsv,
        gray=gray,
        pp_mask=pre_steel_shield_mask,
        pp_centres=pp_centres,
        floor_tracker=_FLOOR_TRACKER,
        estimate_distance_fn=bdr._estimate_distance_cm,
        bearing_fn=bdr._bearing_deg_from_x,
        calibration=calibration,
    )

    # Merge and sort detections
    all_detections = steel + ping_pong
    all_detections = bdr._prioritize_balls(all_detections)

    # Build combined mask used by downstream obstacle detection.
    combined_mask = cv2.bitwise_or(pre_steel_shield_mask, steel_mask)

    return {
        "frame": undistorted,
        "calibration": calibration,
        "detected_balls": all_detections,
        "ball_mask": combined_mask,
        "ping_pong_mask": pp_mask,
        "apriltag_mask": apriltag_mask,
        "steel_mask": steel_mask,
        "disturbance_mask": disturbance_mask,
        "steel_kalman_roi": None,
        "steel_kalman_point": None,
        "steel_hough_circles": [],
        "steel_trajectory": [],
    }

