import cv2
import math
from pupil_apriltags import Detector


class AprilTagNavigator:
    """
    This class handles AprilTag detection for the robot's basket docking task.

    Main responsibilities:
    1. Detect AprilTags in a frame
    2. Keep only the basket tag IDs we care about
    3. Choose the closest valid tag
    4. Compute steering info so the robot can move toward it

    """

    def __init__(
        self,
        target_tag_ids,
        camera_params,
        tag_size_m,
        frame_size=(640, 480),
        tag_families="tagStandard41h12",
    ):

        self.target_tag_ids              = set(target_tag_ids)
        self.camera_params               = camera_params
        self.tag_size_m                  = tag_size_m
        self.frame_width, self.frame_height = frame_size

        self.detector = Detector(
            families=tag_families,
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_yaw_deg(self, rotation_matrix):
        """
        Estimate tag yaw from its rotation matrix.

        Returns degrees; 0° means the tag is facing the camera head-on
        (robot is perpendicular to the wall).

        pupil_apriltags returns pose_R in the camera frame where the tag's
        Z-axis points toward the camera.  Correct yaw (rotation about the
        camera Y-axis) is extracted via the ZYX Euler decomposition:

            yaw = atan2(-R[2, 0], R[0, 0])

        The previous formula atan2(R[0,2], R[2,2]) is only valid when pitch
        is zero; it produces incorrect yaw whenever the tag is tilted
        vertically, which corrupts the lateral_cm calculation downstream.
        """
        yaw_rad = math.atan2(-rotation_matrix[2, 0], rotation_matrix[0, 0])
        return math.degrees(yaw_rad)

    def _normalise_magnitude(self, distance_cm, max_distance_cm=100.0):
        """
        Map distance → movement magnitude in [0.2, 1.0].
        Farther away  → magnitude closer to 1.0 (move faster).
        Closer        → magnitude closer to 0.2 (move slower).
        """
        mag = min(distance_cm / max_distance_cm, 1.0)
        return max(0.2, mag)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Process one camera frame and return navigation info.

        Return dict
        -----------
        {
            "found":         bool,
            "distance_cm":   float | None,
            "yaw_deg":       float | None,
            "lateral_cm":    float | None,
            "moveTargetTag": {"angle": float, "magnitude": float} | None,
            "chosen_tag":    dict | None,
            "all_tags":      list[dict]
        }

        "moveTargetTag" is None only when no tags are found.
        Docking/lock decisions (based on distance and yaw) are left
        entirely to the caller.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size_m
        )

        valid_tags = []

        for tag in results:
            if tag.tag_id not in self.target_tag_ids:
                continue

            tx = float(tag.pose_t[0][0])
            tz = float(tag.pose_t[2][0])

            x_cm        = tx * 100.0
            z_cm        = tz * 100.0
            distance_cm = math.sqrt(x_cm**2 + z_cm**2)
            yaw_deg     = self._compute_yaw_deg(tag.pose_R)
            centre_x    = int(tag.center[0])
            centre_y    = int(tag.center[1])

            # lateral_cm: how far the robot must strafe to stand directly
            # in front of the tag (perpendicular to the wall).
            # This is NOT the same as x_cm (raw camera-frame translation).
            # x_cm is tiny when the tag is near image centre even if the
            # robot is badly angled; lateral_cm captures the true sideways
            # displacement the robot needs to correct.
            #
            #   lateral_cm = sin(yaw) * distance
            #
            # Sign convention matches yaw_deg:
            #   positive yaw → tag rotated CW  → robot must strafe right (+)
            #   negative yaw → tag rotated CCW → robot must strafe left  (-)
            lateral_cm = math.sin(math.radians(yaw_deg)) * distance_cm

            valid_tags.append({
                "tag_id":      int(tag.tag_id),
                "centre_x":    centre_x,
                "centre_y":    centre_y,
                "x_cm":        x_cm,
                "z_cm":        z_cm,
                "distance_cm": distance_cm,
                "yaw_deg":     yaw_deg,
                "lateral_cm":  lateral_cm,
            })

        if not valid_tags:
            return {
                "found":         False,
                "distance_cm":   None,
                "yaw_deg":       None,
                "lateral_cm":    None,
                "moveTargetTag": None,
                "chosen_tag":    None,
                "all_tags":      [],
            }

        chosen_tag = min(valid_tags, key=lambda t: t["distance_cm"])
        angle_deg  = math.degrees(math.atan2(chosen_tag["x_cm"], chosen_tag["z_cm"]))
        magnitude  = self._normalise_magnitude(chosen_tag["distance_cm"])

        return {
            "found":         True,
            "distance_cm":   chosen_tag["distance_cm"],
            "yaw_deg":       chosen_tag["yaw_deg"],
            "lateral_cm":    chosen_tag["lateral_cm"],
            "moveTargetTag": {"angle": angle_deg, "magnitude": magnitude},
            "chosen_tag":    chosen_tag,
            "all_tags":      valid_tags,
        }