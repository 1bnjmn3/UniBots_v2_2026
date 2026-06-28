## Individual motor test
## Based on MotorsController_M_bugFix.py
## Tests each of the 4 motors one at a time: forward, then backward.

from gpiozero import Motor, Device
from gpiozero.pins.lgpio import LGPIOFactory, LGPIOPin
from gpiozero.pins.local import LocalPiFactory
import lgpio
import time


class RP1LGPIOFactory(LGPIOFactory):
    """gpiozero 2.0.1 LGPIOFactory ignores its chip argument and
    auto-detects gpiochip4 on the Pi 5. Newer Pi kernels expose the RP1
    GPIO bank as gpiochip0 (gpiochip4 no longer exists), so the stock
    factory raises lgpio.error: 'can not open gpiochip'. Force chip 0."""
    def __init__(self, chip=0):
        LocalPiFactory.__init__(self)
        self._handle = lgpio.gpiochip_open(chip)
        self._chip = chip
        self.pin_class = LGPIOPin


Device.pin_factory = RP1LGPIOFactory(chip=0)

# ==========================================
# HARDWARE SETUP (same pins as MotorsController_M_bugFix.py)
# ==========================================
motor_fl = Motor(forward=23, backward=27, enable=22)   # Front Left  (A)
motor_rl = Motor(forward=5,  backward=6,  enable=26)    # Rear  Left  (B)
motor_fr = Motor(forward=24, backward=25, enable=16)    # Front Right (C)
motor_rr = Motor(forward=20, backward=21, enable=19)    # Rear  Right (D)

MOTORS = [
    ("Front Left  (FL)", motor_fl),
    ("Rear  Left  (RL)", motor_rl),
    ("Front Right (FR)", motor_fr),
    ("Rear  Right (RR)", motor_rr),
]

TEST_SPEED = 0.5    # 50% duty cycle
RUN_TIME   = 1.5    # seconds per direction


def test_motor(name: str, motor: Motor, speed: float = TEST_SPEED,
               run_time: float = RUN_TIME) -> None:
    """Spin one motor forward then backward, stopping between."""
    print(f"\n=== Testing {name} ===")

    print(f"  forward  @ {int(speed * 100)}% for {run_time}s")
    motor.forward(speed)
    time.sleep(run_time)
    motor.stop()
    time.sleep(0.5)

    print(f"  backward @ {int(speed * 100)}% for {run_time}s")
    motor.backward(speed)
    time.sleep(run_time)
    motor.stop()
    time.sleep(0.5)

    print(f"  {name} done.")


def stop_all() -> None:
    for _, m in MOTORS:
        m.stop()


def run_all() -> None:
    """Test every motor in sequence."""
    for name, motor in MOTORS:
        test_motor(name, motor)
        input("  Press Enter for next motor...")


def menu() -> None:
    while True:
        print("\n--- Individual Motor Test ---")
        for i, (name, _) in enumerate(MOTORS):
            print(f"  {i}: {name}")
        print("  a: test ALL in sequence")
        print("  q: quit")
        choice = input("Select motor: ").strip().lower()

        if choice == "q":
            break
        elif choice == "a":
            run_all()
        elif choice.isdigit() and 0 <= int(choice) < len(MOTORS):
            name, motor = MOTORS[int(choice)]
            test_motor(name, motor)
        else:
            print("  invalid choice")


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_all()
        print("All motors stopped.")
