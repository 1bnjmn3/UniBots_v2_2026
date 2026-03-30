# ─────────────────────────────────────────────────────────────────────────────
# AprilTag Navigator
# ─────────────────────────────────────────────────────────────────────────────

# The two tag IDs printed on your basket
#
# OUR_TAGS = {8, 9}  # ← UPDATE WITH YOUR ACTUAL TAG IDs - TO BE USED AFTER APRIL TAG TESTING
APRILTAG_TARGET_IDS: set[int] = {8, 9}  # ALL TAGS

# Tag family — must match what is printed. Common: "tag36h11", "tagStandard41h12"
APRILTAG_FAMILY = "tag36h11"

# Physical side-length of the printed tag square in metres
APRILTAG_TAG_SIZE_M = 0.1           # ← measure your actual tag
DISTANCE_UNTIL_DOCKING_CM = 35

# Camera intrinsics from OpenCV calibration: (fx, fy, cx, cy) in pixels
# Replace with real values from your calibration run
APRILTAG_CAMERA_PARAMS: tuple[float, float, float, float] = (
    564.98,   # fx  ← placeholder
    565.87,   # fy  ← placeholder
    303.59,   # cx  (frame_width  / 2 as rough fallback)
    282.69,   # cy  (frame_height / 2 as rough fallback)
)

# Frame dimensions must match cv2.VideoCapture resolution
APRILTAG_FRAME_SIZE: tuple[int, int] = (640, 480)   # (width, height)

""" REDUNDANT?
# ── Lock conditions ────────────────────────────────────────────────────────
# The robot is considered docked when BOTH of the following are satisfied:
#
#   1. Tag yaw is within ±APRILTAG_LOCK_YAW_DEG of head-on (perpendicular).
#   2. Tag pixel-centre is within ±APRILTAG_LOCK_CENTER_TOL_PX of the image
#      horizontal midpoint.
#
# Distance is NOT a lock condition — locking is purely alignment-based.
# In practice, the close-range one-shot alignment in the test script forces
# a manual lock, so these thresholds act as a fallback for the APF phase.
APRILTAG_LOCK_DISTANCE_CM   = 20.0   # kept for legacy callers; unused by _is_locked
APRILTAG_LOCK_CENTER_TOL_PX = 7      # ← tightened from 320 px (half-frame) to 7 px
APRILTAG_LOCK_YAW_DEG       = 10.0   # unchanged
"""