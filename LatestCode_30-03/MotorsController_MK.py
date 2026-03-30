## Credits to Krishiv
## Rewritten for Artificial Potential Fields (APF) — smooth curved path support

from gpiozero import Motor
import time
import math

# ==========================================
# TUNING CONSTANTS  ← adjust these values
# ==========================================
DEFAULT_SPEED       = 50    # Base forward speed (100%)
MAX_TURN_ANGLE      = 70.0  # Smaller value -> stronger steering at far range
TIME_PER_DEGREE     = 0.005 # Seconds to rotate 1Â° at 60% speed - needs physical calibration
DEAD_ZONE_MAGNITUDE = 0.15  # APF magnitudes below this are treated as "stop"
DEAD_ZONE_ANGLE     = 5     # Angles within ±5° are treated as straight ahead
MIN_MOTOR_PWM       = 0.20  # Stall prevention floor when commanding movement

ALIGN_STRAFE_SPEED   = 45    # % — strafe speed used for x-offset correction
ALIGN_ROTATE_SPEED   = 20    # % — gentle rotation speed (used by close_range_align legacy)

# --- Advanced steering constants for low-friction control ---
MIN_TURN_ANGLE      = 18.0  # Smaller value -> stronger steering at close range
ALIGNMENT_BRAKING   = 0.50  # Reduce forward speed during hard turns
ALIGN_YAW_THRESHOLD_TAG  = 5.0  # degrees — ignore yaw corrections smaller than this
ALIGN_X_THRESHOLD_TAG    = 2.0   # cm — ignore x-offset corrections smaller than this


# TIME_PER_CM_STRAFE: how many seconds to strafe 1 cm at ALIGN_STRAFE_SPEED.
# - Calibrate physically: run strafe_right(ALIGN_STRAFE_SPEED) for 1 second,
#   measure how far the robot moved sideways, then set this to 1.0 / distance_cm.
TIME_PER_CM_STRAFE   = 0.05  # seconds per cm at ALIGN_STRAFE_SPEED (placeholder)


# ==========================================
# HARDWARE SETUP (L298N Motor Drivers)
# ==========================================

# Left side  - share the same logical direction
motor_fl = Motor(forward=23, backward=27, enable=22)   # Front Left  (A)
motor_rl = Motor(forward=5, backward=6, enable=26)   # Rear  Left  (B)

# Right side - share the same logical direction
motor_fr = Motor(forward=24, backward=25, enable=16)   # Front Right (C)
motor_rr = Motor(forward=20, backward=21, enable=19)   # Rear  Right (D)

# ==========================================
# LOW-LEVEL WHEEL CONTROL
# ==========================================

def _set_side(motors: list, speed: float) -> None:

    speed = max(-1.0, min(1.0, speed))
    for m in motors:
        if speed > 0:
            m.forward(speed)
        elif speed < 0:
            m.backward(abs(speed))
        else:
            m.stop()

def set_left_speed(speed: float) -> None:
    """Drive both left wheels. speed âˆˆ [-1.0, +1.0]."""
    _set_side([motor_fl, motor_rl], speed)

def set_right_speed(speed: float) -> None:
    """Drive both right wheels. speed âˆˆ [-1.0, +1.0]."""
    _set_side([motor_fr, motor_rr], speed)

def stop_robot() -> None:
    """Immediately stop all four wheels."""
    set_left_speed(0)
    set_right_speed(0)

# ==========================================
# DIFFERENTIAL DRIVE
# ==========================================

def drive(base_speed: float, steering: float) -> None:
    left_speed  = base_speed * (1.0 + steering)
    right_speed = base_speed * (1.0 - steering)
    # Normalize so neither side exceeds ±1.0 while preserving the ratio.
    max_val = max(abs(left_speed), abs(right_speed), 1.0)
    set_left_speed(left_speed / max_val)
    set_right_speed(right_speed / max_val)

# ==========================================
# APF VECTOR - MOTOR COMMAND
# ==========================================

def apf_move(angle_deg: float, magnitude: float) -> None:
    if magnitude < DEAD_ZONE_MAGNITUDE:
        stop_robot()
        return

    dynamic_turn_limit = MIN_TURN_ANGLE + (MAX_TURN_ANGLE - MIN_TURN_ANGLE) * magnitude
    clamped_angle = max(-dynamic_turn_limit, min(dynamic_turn_limit, angle_deg))
    raw_steering = clamped_angle / dynamic_turn_limit   

    speed_penalty = 1.0 - (abs(raw_steering) * ALIGNMENT_BRAKING)
    base_speed = max(MIN_MOTOR_PWM, magnitude * speed_penalty)

    # Direct steering command (no smoothing)
    drive(base_speed, raw_steering)

# ==========================================
# CONVENIENCE WRAPPERS
# ==========================================

def move_forward(speed: int = DEFAULT_SPEED) -> None:
    drive(speed / 100.0, steering=0.0)

def move_backward(speed: int = DEFAULT_SPEED) -> None:
    drive(-(speed / 100.0), steering=0.0)

def pivot_right(speed: int = DEFAULT_SPEED) -> None:
    s = speed / 100.0
    set_left_speed(s)
    set_right_speed(-s)

def pivot_left(speed: int = DEFAULT_SPEED) -> None:
    s = speed / 100.0
    set_left_speed(-s)
    set_right_speed(s)

def pivot_right_degrees(degrees: float, speed: int = 60) -> None:
    """Rotate clockwise by `degrees` using TIME_PER_DEGREE calibration."""
    pivot_right(speed)
    time.sleep(abs(degrees) * TIME_PER_DEGREE)
    stop_robot()

def pivot_left_degrees(degrees: float, speed: int = 60) -> None:
    """Rotate counter-clockwise by `degrees` using TIME_PER_DEGREE calibration."""
    pivot_left(speed)
    time.sleep(abs(degrees) * TIME_PER_DEGREE)
    stop_robot()

# ==========================================
# SPECIALISED BEHAVIOURS
# ==========================================

def slow_wall_approach(speed: int = 50, duration: float = 1.25) -> None:
    """Creep forward slowly for precise wall alignment."""
    move_forward(speed)
    time.sleep(duration)
    stop_robot()

def reverse_from_wall(speed: int = DEFAULT_SPEED, duration: float = 0.30) -> None:
    move_backward(speed)
    time.sleep(duration)
    stop_robot()

def move_forward_toStart(speed: int = 45, duration: float = 0.7) -> None:
    move_forward(speed)
    time.sleep(duration)
    stop_robot()

def little_reverse(speed: int = 30, duration: float = 0.2) -> None:
    """A short reverse to clear the claw after picking."""
    move_backward(speed)
    time.sleep(duration)
    stop_robot()

def confident_approach_toGrab(speed: int = 50, duration: float = 0.4) -> None:
    """Approach the ball confidently for a secure grab."""
    move_forward(speed)
    time.sleep(duration)
    stop_robot()

# ==========================================
# MECANUM STRAFE SUPPORT
# ==========================================
#
#  Mecanum wheel layout (viewed from above):
#    FL (/)   FR (\)
#    RL (\)   RR (/)
#
#  Strafe right: FL fwd, RL bwd, FR bwd, RR fwd
#  Strafe left:  FL bwd, RL fwd, FR fwd, RR bwd

def strafe_right(speed: int = DEFAULT_SPEED) -> None:
    s = speed / 100.0
    motor_fl.forward(s)
    motor_rl.backward(s)
    motor_fr.backward(s)
    motor_rr.forward(s)

def strafe_left(speed: int = DEFAULT_SPEED) -> None:
    s = speed / 100.0
    motor_fl.backward(s)
    motor_rl.forward(s)
    motor_fr.forward(s)
    motor_rr.backward(s)

def mecanum_move(vx: float, vy: float, omega: float) -> None:
    """
    Full mecanum velocity mixer.

    vx    : lateral velocity.  +1 = right,   -1 = left.
    vy    : forward velocity.  +1 = forward,  -1 = back.
    omega : rotation.          +1 = CW,       -1 = CCW.
    """
    fl =  vy + vx + omega
    fr =  vy - vx - omega
    rl =  vy - vx + omega
    rr =  vy + vx - omega

    max_val = max(abs(fl), abs(fr), abs(rl), abs(rr), 1.0)
    fl, fr, rl, rr = fl/max_val, fr/max_val, rl/max_val, rr/max_val

    _set_side([motor_fl], fl)
    _set_side([motor_fr], fr)
    _set_side([motor_rl], rl)
    _set_side([motor_rr], rr)

# ==========================================
# ONE-SHOT CLOSE-RANGE ALIGNMENT  ← NEW
# ==========================================

def strafe_to_align(x_cm: float) -> None:
    """
    Strafe sideways by an open-loop duration derived from the measured
    lateral x-offset.

    This is called ONCE when the robot enters close range.
    It does NOT loop or re-check — the duration is computed directly from
    x_cm using TIME_PER_CM_STRAFE, which must be calibrated physically.

    Parameters
    ----------
    x_cm : float
        Lateral offset of the tag in cm as reported by AprilTagNavigator.
        Positive = tag is to the right of centre → strafe right.
        Negative = tag is to the left  of centre → strafe left.

    Calibration
    -----------
    Run  strafe_right(ALIGN_STRAFE_SPEED)  for exactly 1 second on your
    surface, measure the lateral displacement in cm, then:

        TIME_PER_CM_STRAFE = 1.0 / measured_cm_per_second
    """
    if abs(x_cm) < ALIGN_X_THRESHOLD_TAG:
        return  # offset small enough to skip

    duration = abs(x_cm) * TIME_PER_CM_STRAFE

    if x_cm < 0:
        strafe_right(ALIGN_STRAFE_SPEED)
    else:
        strafe_left(ALIGN_STRAFE_SPEED)

    time.sleep(duration)
    stop_robot()


def rotate_to_align(yaw_deg: float) -> None:
    """
    Rotate in place by an open-loop duration derived from the measured
    yaw error.

    This is called ONCE when the robot enters close range.
    It does NOT loop or re-check — the rotation duration is computed
    directly from yaw_deg using TIME_PER_DEGREE, which must be
    calibrated physically.

    Parameters
    ----------
    yaw_deg : float
        Yaw of the tag as reported by AprilTagNavigator.
        Positive = tag is rotated clockwise relative to camera
                   → pivot right to face it head-on.
        Negative = tag is rotated counter-clockwise
                   → pivot left.

    Calibration
    -----------
    See TIME_PER_DEGREE above.  Run pivot_right_degrees(90) and
    verify the robot turns exactly 90°; adjust TIME_PER_DEGREE
    until it does.
    """
    if abs(yaw_deg) < ALIGN_YAW_THRESHOLD_TAG:
        return  # yaw small enough to skip

    if yaw_deg > 0:
        pivot_right_degrees(abs(yaw_deg))
    else:
        pivot_left_degrees(abs(yaw_deg))



