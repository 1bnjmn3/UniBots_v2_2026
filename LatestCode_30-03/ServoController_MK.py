## Credits to Raef

import time
from dataclasses import dataclass
from adafruit_servokit import ServoKit

PCA9685_I2C_ADDRESS = 0x40
PCA9685_CHANNELS = 16
PCA9685_FREQUENCY_HZ = 50

MG90S_CHANNEL = 0
CLUTCH_SERVO_CHANNEL = 15

SUPPLY_VOLTAGE = 5.0

DEFECTED_RANGE_AGJ = 0.12
DOWN_POSITION_ANGLE = 260.0

@dataclass
class ContinuousServoCal:
    STOP_THROTTLE: float = 0.0 + DEFECTED_RANGE_AGJ
    TURN_UP_THROTTLE: float = -0.6 + DEFECTED_RANGE_AGJ
    TURN_DOWN_THROTTLE: float = 0.65 + DEFECTED_RANGE_AGJ
    SECONDS_PER_180: float = 0.3                                         #CHANGE IF THE TURN DOES NOT COMPLETE A FULL 180° ROTATION

@dataclass
class PositionalServoCal:
    ACTUATION_RANGE_DEG: int = 300     #REALISTICALLY, WE LIMIT THE RANGE TO 15 - 270 DEGREES TO AVOID COLLISION WITH ROBOT BODY
    MIN_PULSE_US: int = 500
    MAX_PULSE_US: int = 2500

MG90S_CAL = ContinuousServoCal()
CLUTCH_CAL = PositionalServoCal()

kit = ServoKit(channels=PCA9685_CHANNELS, address=PCA9685_I2C_ADDRESS)

clutch = kit.servo[CLUTCH_SERVO_CHANNEL]
clutch.actuation_range = CLUTCH_CAL.ACTUATION_RANGE_DEG
clutch.set_pulse_width_range(CLUTCH_CAL.MIN_PULSE_US, CLUTCH_CAL.MAX_PULSE_US)

mg90s = kit.continuous_servo[MG90S_CHANNEL]

### GENERAL FUNCTIONS ###########################
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_sleep(seconds: float) -> None:
    time.sleep(max(0.0, seconds))

### MG90S CONTINUOUS SERVO FUNCTIONS ###########################
def mg90s_stop() -> None:
    mg90s.throttle = MG90S_CAL.STOP_THROTTLE

def mg90s_turn_by_180_up() -> None:
    mg90s.throttle = clamp(MG90S_CAL.TURN_UP_THROTTLE, -1.0, 1.0)
    safe_sleep(MG90S_CAL.SECONDS_PER_180)
    mg90s_stop()

def mg90s_turn_by_180_down() -> None:
    mg90s.throttle = clamp(MG90S_CAL.TURN_DOWN_THROTTLE, -1.0, 1.0)
    safe_sleep(MG90S_CAL.SECONDS_PER_180)
    mg90s_stop()

### CLUTCH POSITIONAL SERVO FUNCTIONS ###########################
def clutch_set_angle(deg: float) -> None:
    deg = clamp(deg, 15.0, 270.0)
    clutch.angle = deg

def clutch_up() -> None:
    clutch_set_angle(15)

def clutch_down() -> None:
    clutch_set_angle(DOWN_POSITION_ANGLE)

def clutch_grabbing_motion() -> None:
    clutch_up()
    safe_sleep(1.3)
    clutch_down()
