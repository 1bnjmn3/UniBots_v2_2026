#!/usr/bin/env python3


"""
main_integrated.py  –  Robot Ball Collection Algorithm  (integrated CV)
Target platform: Raspberry Pi 5

Derived from TODO/mainToChange.py.
Changes vs the original:
  1. ball_detector_process  – fully implemented with Search / Track / Capture modes.
  2. image_capture_thread   – fully implemented with cv2.VideoCapture; frame
                              distribution freezes during atomic sequences.
  3. motors_handler_thread  – restructured into a priority-ordered elif chain;
                              pickingInProcess and dockingInProcess branches now
                              execute as blocking atomic sequences.
  4. servo_handler_thread   – pickingInProcess cleared as the final step of the
                              engageClaw branch so MotorsHandler unblocks only
                              after the claw is fully reset.
  5. apriltag_detector_process – clean TODO stub with full implementation contract.

Architecture
------------
Processes : MainTimer | BallDetector | AprilTagDetector
Threads   : MotorsHandler | ServoHandler | ButtonHandler (gpiozero) | ImageFrameCapture

Pause convention
----------------
  worker_pause_event   (set = RUNNING)  –  waited on by all workers EXCEPT MainTimer
  timer_pause_event    (set = RUNNING)  –  waited on by MainTimer only
  stop_event           (set = STOP)     –  terminates every loop when set

Button behaviour
----------------
  Hold > 5 s : full reset  – clears BOTH pause events, resets alg state
  Short press : if alg running       → pause workers only (MainTimer keeps running)
                if workers paused     → resume workers
                if neither event set  → set both pause events to start the algorithm

Shared dict conventions
-----------------------
  moveTargetBall / moveTargetTag : {"angle": float, "magnitude": float}
    angle     – degrees relative to the robot's current heading.
                0° = straight ahead, positive = right, negative = left.
                Range: [-180, +180].  Must match MotorsController.apf_move().
    magnitude – normalised APF force strength in [0.0, 1.0].

  clawAdjusted : False → ServoHandler must reposition the claw before anything else.
                 True  → claw is in position; engageClaw / engageHandle may be used.

  clawBusy     : True  → a grab or unload is in progress; MotorsHandler must not
                          set engageClaw or engageHandle until this clears.
                 False → ready for the next action.

  pickingInProcess : True → ball has entered capture zone; MotorsHandler is executing
                            the atomic grab sequence; image capture is frozen.
                    False → normal operation.

  dockingInProcess : True → AprilTag has been locked; MotorsHandler is executing
                            the atomic dock/unload/reverse sequence; image capture
                            is frozen.
                    False → normal operation.

Atomic sequence guarantees
--------------------------
  lockedBall  → BallDetector sets pickingInProcess=True and lockedBall=<type>.
                ImageFrameCapture stops distributing frames immediately.
                MotorsHandler executes: confident_approach → stop → engageClaw.
                MotorsHandler then blocks until ServoHandler clears pickingInProcess.
                ImageFrameCapture and detectors resume automatically.

  lockedTag   → AprilTagDetector sets dockingInProcess=True and lockedTag=True.
                ImageFrameCapture stops distributing frames immediately.
                MotorsHandler executes: slow_wall_approach → stop → engageHandle.
                MotorsHandler blocks until ServoHandler clears engageHandle.
                If storageFull: reverse_from_wall, clear dockingInProcess, resume.
                If finilisingState: stop_event.set() – end of match.
"""

import multiprocessing
import threading
import time
import queue
import logging

import cv2
from gpiozero import Button
import queue

# ── Module imports ─────────────────────────────────────────────────────────────
import ServoController_MK as SC
import MotorsController_M_bugFix as MC
 
# K vision pipeline (ball detection)
from ball_detector_runtime_K import (
    detect_balls,
    detect_obstacles,
    compute_navigation_vector,
)
 
# M AprilTag pipeline
import configTag_M as configTag 
from AprilTagNavigator_M import AprilTagNavigator
 

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(processName)s/%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────

BUTTON_GPIO = 4           # GPIO pin for the physical button
CAMERA_INDEX = 0          # cv2.VideoCapture device index (1 = second camera)
CAPTURE_Y_THRESHOLD = 0.88  # ball bottom pixel at this fraction of frame height → Capture mode

#From Krishiv's mainK.py
CAPTURE_ALIGN_ROTATE_MAX_ABS_DEG = 10.0
CAPTURE_ALIGN_MAX_RETRIES = 1
POST_PICK_CAPTURE_COOLDOWN_S = 0.4
PING_PONG_PROFILE = "orange"

# Initial values for all shared state – used at startup and on full reset.
# Centralising them here ensures on_held() and main() are always in sync.
SHARED_INITIAL_STATE = {
    # ── Algorithm control ──────────────────────────────────────────────────
    "algHasBeenStarted":  False,    # set True by MainTimer once running
    "timeStarted":        0.0,      # epoch timestamp of algorithm start
    "finilisingState":    False,    # set True after 160 s; triggers unload+park
    "startingPosReached": False,    # True once the robot has driven to the starting position

    # ── Ball detection / navigation ────────────────────────────────────────
    # angle: degrees from robot heading, positive = right. magnitude: 0.0–1.0.
    "lockedBall":         "",       # "" | "PingPong" | "steel"
    "moveTargetBall":     None,     # dict: {angle, magnitude}  (see convention above)
    "pickingInProcess":   False,    # True  → atomic grab sequence running; image
                                    #         capture frozen; set by BallDetector,
                                    #         cleared by ServoHandler after grab.
    
    ## --- FROM KIRSHIV'S MAINK.PY ---
    "captureAlignPending": False,   # True -> MotorsHandler rotates in place before capture.
    "captureAlignAngleDeg": 0.0,    # Target in-place rotation angle from BallDetector.
    "captureAlignRetryCount": 0,    # Rotate-and-recheck retry counter.
    "captureIgnoreUntilS": 0.0,     # Suppress immediate post-pick recapture until this time.
    "lastCaptureSignature": "",    # Coarse signature used to block duplicate capture trigger.

    # ── AprilTag detection / navigation ───────────────────────────────────
    # angle: degrees from robot heading, positive = right. magnitude: 0.0–1.0.
    "moveTargetTag":      None,     # dict: {angle, magnitude}  (see convention above)
    "dockingInProcess":   False,    # True  → atomic dock sequence running; image
                                    #         capture frozen; set by AprilTagDetector,
                                    #         cleared by MotorsHandler after reverse.
    "dockingInfo":       None,      # dict with raw AprilTag pose info for the locked tag

    # ── Claw / servo control ───────────────────────────────────────────────
    "engageClaw":         False,    # True → ServoHandler should grab a ball
    "engageHandle":       False,    # True → ServoHandler should unload balls
    "clawAdjusted":       False,    # False → ServoHandler must reposition the claw
                                    # True  → claw is in position; ready for actions
    "clawBusy":           False,    # True → grab/unload in progress; block re-trigger
    "heldPingPong":        0,        # count of ping pong balls held
    "heldSteel":           0,        # count of steel balls held
    "storageFull":        False,    # True → navigate to unload zone

    # ── Button ────────────────────────────────────────────────────────────
    "btnHeld":            False,    # True if button was held for >= 5 s
}

# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  Main Timer // - Needs no changes
# ══════════════════════════════════════════════════════════════════════════════
def main_timer_process(shared, timer_pause_event, stop_event):
    """
    Tracks elapsed algorithm time.
    After 160 s of active runtime it sets finilisingState = True so
    the robot begins its end-of-game unload + park sequence.
    On the first iteration after both pause events are set, records
    timeStarted and sets algHasBeenStarted = True.
    """
    log.info("MainTimer started.")

    while not stop_event.is_set():
        timer_pause_event.wait()          # block if the timer itself is paused

        if shared["algHasBeenStarted"]:
            # ── Check how much time has passed ────────────────────────────
            elapsed = time.time() - shared["timeStarted"]
            if elapsed > 160 and not shared["finilisingState"]:
                log.info("1000 s elapsed – entering finalising state.") # to change to 160 s for actual match
                shared["finilisingState"] = True
        else:
            # ── First iteration: record start timestamp ───────────────
            shared["algHasBeenStarted"] = True
            shared["timeStarted"]       = time.time()
            log.info("Algorithm timer started.")

        time.sleep(0.02)

    log.info("MainTimer stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  Ball Detector // FROM KIRSHIV'S mainK.py 
# ══════════════════════════════════════════════════════════════════════════════
def ball_detector_process(frame_queue, shared, worker_pause_event, stop_event):
    """
    Reads the latest camera frame, runs the K vision pipeline, classifies
    detected balls, and updates shared["lockedBall"] and shared["moveTargetBall"].

    Three operating modes
    ---------------------
    SEARCH  – no balls visible:
        lockedBall=""  moveTargetBall=None  → MotorsHandler will pivot_left to scan.

    CAPTURE – highest-priority ball's bottom pixel (y + radius) has reached
              CAPTURE_Y_THRESHOLD of the frame height, meaning the ball is at the
              robot's intake zone:
        lockedBall="PingPong"|"steel"  moveTargetBall=None  pickingInProcess=True
        → MotorsHandler starts atomic grab sequence; image capture freezes.

    TRACK   – ball visible but not yet in the intake zone:
        lockedBall=""  moveTargetBall={angle, magnitude}
        → MotorsHandler calls apf_move() to navigate toward the ball.
        Obstacle repulsion is applied by compute_navigation_vector() from the
        K runtime — no additional timeout or skip logic is applied here.

    moveTargetBall angle convention: 0° = straight ahead, positive = right,
    negative = left.  Range [-180, +180].  Must match apf_move() expectation.
    """
    log.info("BallDetector started.")

    def _capture_signature(ball: dict) -> str:
        # Coarse quantisation tolerates tiny jitter while still blocking duplicate triggers.
        return (
            f"{ball['type']}:{int(float(ball['x']) // 16)}:"
            f"{int(float(ball['y']) // 16)}:{int(float(ball['radius']) // 8)}"
        )

    while not stop_event.is_set():
        worker_pause_event.wait()

        # Run only when: storage not full, not finalising, and no atomic
        # sequence is currently locking the robot.
        if (not shared["storageFull"]
                and not shared["finilisingState"]
                and not shared["pickingInProcess"]
                and not shared["dockingInProcess"]):

            # ── Grab the latest frame (non-blocking) ──────────────────────
            frame = None
            try:
                frame = frame_queue.get_nowait()
            except Exception:
                pass

            if frame is None:
                time.sleep(0.02)
                continue

            # ── Run K vision pipeline ──────────────────────────────────────
            try:
                detection = detect_balls(frame, ping_pong_profile=PING_PONG_PROFILE)
            except Exception as exc:
                log.warning(f"detect_balls raised: {exc}")
                time.sleep(0.02)
                continue

            detected_balls = detection["detected_balls"]
            frame_height   = detection["frame"].shape[0]

            # ── SEARCH mode ───────────────────────────────────────────────
            if not detected_balls:
                shared["lockedBall"]     = ""
                shared["moveTargetBall"] = None
                time.sleep(0.02)
                continue

            # Highest-priority ball (sorted by type priority, then distance, then |angle|)
            # _prioritize_balls() inside the K runtime sorts: steel first, nearest, most centred.
            target = detected_balls[0]
            now_s = time.time()

            # ── CAPTURE mode ──────────────────────────────────────────────
            ball_bottom_y = target["y"] + target["radius"]
            in_capture_zone = ball_bottom_y >= frame_height * CAPTURE_Y_THRESHOLD
            is_aligned_for_capture = abs(float(target["angle"])) <= CAPTURE_ALIGN_ROTATE_MAX_ABS_DEG
            target_signature = _capture_signature(target)
            in_post_pick_cooldown = now_s < float(shared["captureIgnoreUntilS"])

            if in_capture_zone:
                if in_post_pick_cooldown and target_signature == shared["lastCaptureSignature"]:
                    shared["lockedBall"] = ""
                    shared["captureAlignPending"] = False
                    shared["captureAlignAngleDeg"] = 0.0
                    shared["moveTargetBall"] = None
                    log.info("CAPTURE suppressed during cooldown (duplicate signature).")
                    time.sleep(0.02)
                    continue

                if is_aligned_for_capture:
                    shared["lockedBall"] = target["type"]   # "PingPong" or "steel"
                    shared["moveTargetBall"] = None
                    shared["pickingInProcess"] = True         # freeze image capture;
                                                              # trigger atomic grab sequence
                    shared["captureAlignPending"] = False
                    shared["captureAlignAngleDeg"] = 0.0
                    shared["captureAlignRetryCount"] = 0
                    shared["lastCaptureSignature"] = target_signature
                    log.info(
                        f"CAPTURE: locked {shared['lockedBall']} "
                        f"at y={ball_bottom_y:.1f} "
                        f"(threshold={frame_height * CAPTURE_Y_THRESHOLD:.1f})"
                    )
                    time.sleep(0.02)
                    continue

                if int(shared["captureAlignRetryCount"]) < CAPTURE_ALIGN_MAX_RETRIES:
                    shared["lockedBall"] = ""
                    shared["moveTargetBall"] = None
                    shared["captureAlignPending"] = True
                    shared["captureAlignAngleDeg"] = float(target["angle"])
                    log.info(
                        f"CAPTURE align pending: angle={float(target['angle']):.1f}deg "
                        f"(retry={int(shared['captureAlignRetryCount']) + 1}/{CAPTURE_ALIGN_MAX_RETRIES})"
                    )
                    time.sleep(0.02)
                    continue

                # Retry exhausted: abort this capture attempt and fall back to tracking.
                shared["captureAlignPending"] = False
                shared["captureAlignAngleDeg"] = 0.0
                shared["captureAlignRetryCount"] = 0
                log.info("CAPTURE alignment retry exhausted; returning to TRACK mode.")
            else:
                shared["captureAlignRetryCount"] = 0
                shared["captureAlignPending"] = False
                shared["captureAlignAngleDeg"] = 0.0

            # ── TRACK mode ────────────────────────────────────────────────
            shared["lockedBall"] = ""

            # Detect obstacles from the undistorted K pipeline frame and apply APF.
            obstacles = detect_obstacles(detection["frame"], detection["ball_mask"])
            nav = compute_navigation_vector(
                target, obstacles, detection["frame"].shape, detection["calibration"]
            )
            # nav is None only when target is None – guaranteed non-None here.
            shared["moveTargetBall"] = nav

        else:
            # Inactive branch.
            # IMPORTANT: do NOT clear lockedBall while pickingInProcess=True –
            # ServoHandler still needs it to identify the ball type for the
            # held-ball count update.
            if not shared["pickingInProcess"]:
                shared["lockedBall"] = ""
            shared["captureAlignPending"] = False
            shared["captureAlignAngleDeg"] = 0.0
            shared["captureAlignRetryCount"] = 0
            shared["moveTargetBall"] = None

        time.sleep(0.02)

    log.info("BallDetector stopped.")

# ══════════════════════════════════════════════════════════════════════════════
# PROCESS  –  AprilTag Detector // FROM MARK'S mainM.py
# ══════════════════════════════════════════════════════════════════════════════
def apriltag_detector_process(frame_queue, shared, worker_pause_event, stop_event):
    
    navigator = AprilTagNavigator(
        target_tag_ids          = configTag.APRILTAG_TARGET_IDS,
        camera_params           = configTag.APRILTAG_CAMERA_PARAMS,
        tag_size_m              = configTag.APRILTAG_TAG_SIZE_M,
        frame_size              = configTag.APRILTAG_FRAME_SIZE,
        tag_families            = configTag.APRILTAG_FAMILY,
    )

    log.info("AprilTagDetector started.")

    while not stop_event.is_set(): 
        worker_pause_event.wait()

        #Run this thread only if needed (if storage is full or finilisingState is True)
        if (shared["storageFull"] or shared["finilisingState"]) and not shared["dockingInProcess"]:

            frame = None
            try:
                frame = frame_queue.get_nowait()
            except Exception:
                pass

            if frame is None:
                time.sleep(0.02)
                continue

            try:  
                result = navigator.process_frame(frame)

            except Exception as exc:
                log.warning(f"AprilTagNavigator raised: {exc}")
                time.sleep(0.02)  
                continue 

            if not result["found"] :
                log.info("AprilTagDetector: no tag found in frame.")
                shared["moveTargetTag"] = None
                shared["dockingInfo"] = None
                time.sleep(0.02)
                continue
            log.info(f"AprilTagDetector: tag detected at distance {result['distance_cm']:.1f} cm, yaw {result['yaw_deg']:.1f}°, lateral {result['lateral_cm']:.1f} cm.")
            distance = result["distance_cm"] 
            shared["moveTargetTag"] = result["moveTargetTag"] 

            if distance < configTag.DISTANCE_UNTIL_DOCKING_CM and not shared["dockingInProcess"]:
                log.info("AprilTagDetector: tag locked – starting docking sequence.")
                shared["dockingInfo"] = {
                    "distance_cm": result["distance_cm"],
                    "yaw_deg": result["yaw_deg"],
                    "lateral_cm": result["lateral_cm"],
                }
                shared["dockingInProcess"] = True

        else:
            shared["moveTargetTag"] = None
            if not shared["dockingInProcess"]:   # ← don't wipe mid-sequence
                shared["dockingInfo"] = None

        time.sleep(0.02)

    log.info("AprilTagDetector stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Image Frame Capture // 
# ══════════════════════════════════════════════════════════════════════════════
def image_capture_thread(ball_frame_q, tag_frame_q, shared, worker_pause_event, stop_event):
    """
    Continuously captures camera frames and distributes the latest one to
    both detector processes via their respective single-slot queues.
    Stale frames are discarded so detectors always see the most recent image.

    Frame distribution is suspended while pickingInProcess=True or
    dockingInProcess=True so that neither detector can write new targets
    into shared state while an atomic motor/servo sequence is in progress.
    The camera keeps reading (so the driver buffer stays fresh) but frames
    are simply not forwarded until the lock is released.
    """
    log.info("ImageFrameCapture started.")

    # Force the V4L2 backend. Without it OpenCV auto-probes FFMPEG / ORBBEC
    # (obsensor) backends first, which spam errors and intermittently fail to
    # open a plain USB UVC webcam on the Pi. Retry a few times in case the
    # device is momentarily busy (e.g. a previous run still releasing it).
    cap = None
    for attempt in range(10):
        if stop_event.is_set():
            return
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if cap.isOpened():
            break
        log.warning(
            f"ImageFrameCapture: camera index {CAMERA_INDEX} not ready "
            f"(attempt {attempt + 1}/10) – retrying."
        )
        cap.release()
        time.sleep(0.5)

    if cap is None or not cap.isOpened():
        log.error(f"ImageFrameCapture: could not open camera index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # keep only the freshest frame in driver buffer

    try:
        while not stop_event.is_set():
            worker_pause_event.wait()

            ret, new_frame = cap.read()
            if not ret or new_frame is None:
                log.warning("ImageFrameCapture: cap.read() failed – skipping frame.")
                time.sleep(0.02)
                continue

            # ── Freeze distribution during atomic sequences ────────────────
            if shared["dockingInProcess"]:
                time.sleep(0.02)
                continue

            # ── Push to BallDetector queue  (drop stale frame first) ───────
            try:
                ball_frame_q.put_nowait(new_frame)
            except queue.Full:
                try:
                    ball_frame_q.get_nowait() # Purge stale
                    ball_frame_q.put_nowait(new_frame) # Replace
                except Exception:
                    pass

            # ── Push to AprilTagDetector queue  (drop stale frame first) ───
            try:
                tag_frame_q.put_nowait(new_frame)
            except queue.Full:
                try:
                    tag_frame_q.get_nowait()
                    tag_frame_q.put_nowait(new_frame)
                except Exception:
                    pass

            time.sleep(0.02)
    finally:
        cap.release()
        log.info("ImageFrameCapture: camera released.")

    log.info("ImageFrameCapture stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Motors Handler // FINISHED BY MARK, KRISHIV HELPED WITH MOTORSCONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
def motors_handler_thread(shared, worker_pause_event, stop_event):
    """
    Reads moveTargetBall / moveTargetTag and the pickingInProcess /
    dockingInProcess flags to decide how to drive the robot's wheels.

    Priority (highest → lowest)
    ---------------------------
    1. startingPosReached=False  → drive to starting position (one-shot).
    2. pickingInProcess=True     → ATOMIC: approach → stop → signal claw →
                                   block until ServoHandler clears flag.
    3. dockingInProcess=True     → ATOMIC: approach wall → stop → signal handle →
                                   block until ServoHandler clears engageHandle →
                                   reverse (or stop_event if finilisingState).
    4. finilisingState=True      → navigate toward AprilTag (pivot if not found).
    5. storageFull=True          → navigate toward AprilTag (pivot if not found).
    6. default                   → navigate toward/search for balls.

    APF angle convention: 0° = straight ahead, positive = right, negative = left.
    apf_move() in MotorsController interprets positive angle as steer-right.
    """
    log.info("MotorsHandler started.")

    # ── INITIALIZE ToF SENSOR (GPIO 12/13) ──────────────────────────────
    # Initializing inside the thread prevents multiprocessing I2C bus locks
    try:
        # Physical Pin 33 = board.D13 (SCL) | Physical Pin 32 = board.D12 (SDA)
        i2c = busio.I2C(board.D13, board.D12)
        tof_sensor = adafruit_vl53l0x.VL53L0X(i2c)
        tof_sensor.measurement_timing_budget = 20000  # Fast polling mode
        TOF_AVAILABLE = True
        log.info("ToF sensor initialized successfully on GPIO 12 & 13.")
    except Exception as e:
        log.error(f"Failed to initialize ToF sensor: {e}")
        TOF_AVAILABLE = False

    while not stop_event.is_set():
        worker_pause_event.wait()
        try:
            # ── ONE-SHOT: drive to starting position ──────────────────────────
            if not shared["startingPosReached"]:
                MC.move_forward_toStart()
                shared["startingPosReached"] = True

            # ── ATOMIC SEQUENCE: ball pick (ToF UPGRADE) ──────────────────────
            elif shared["pickingInProcess"]:
                log.info("MotorsHandler: PICK sequence started. Waiting for ToF < 6cm.")
                tof_triggered = False
                
                if TOF_AVAILABLE:
                    MC.move_forward(45)  # Controlled approach speed
                    start_time = time.time()
                    
                    # Drive forward until ToF reads < 60mm, with a 1.5s timeout safety
                    while time.time() - start_time < 1.5 and not stop_event.is_set():
                        try:
                            # ToF sensor returns distance in millimeters
                            dist_mm = tof_sensor.range
                            if dist_mm < 60:
                                tof_triggered = True
                                log.info(f"ToF triggered at {dist_mm}mm! Stopping to grab.")
                                break
                        except Exception as e:
                            pass # Ignore momentary I2C read glitches
                        time.sleep(0.01) # Rapid polling
                    
                    MC.stop_robot()
                else:
                    # Fallback to old blind logic if sensor breaks mid-match
                    MC.confident_approach_toGrab()
                    MC.stop_robot()
                    tof_triggered = True 

                # Decide whether to grab or abort based on the ToF loop
                if tof_triggered:
                    shared["engageClaw"] = True
                    shared["clawBusy"]   = True
                    
                    while not stop_event.is_set() and shared["pickingInProcess"]:
                        time.sleep(0.02)
                    MC.little_reverse()
                    log.info("MotorsHandler: PICK sequence complete.")
                else:
                    # The 1.5s timeout hit without seeing the ball. It rolled away.
                    log.warning("MotorsHandler: ToF timeout. Ball lost! Aborting pick.")
                    shared["pickingInProcess"] = False
                    shared["captureAlignPending"] = False
                    shared["captureAlignAngleDeg"] = 0.0
                    MC.little_reverse()

            # ── ATOMIC SEQUENCE: tag dock / unload ────────────────────────────
            elif shared["dockingInProcess"]:
                log.info("MotorsHandler: DOCK sequence started.")
                docking_snapshot = shared["dockingInfo"]
                if docking_snapshot is None:
                    log.warning("MotorsHandler: dockingInfo was None on entry – aborting dock.")
                    shared["dockingInProcess"] = False
                    time.sleep(0.02)
                    continue
                else:
                    MC.stop_robot()
                    time.sleep(0.3)
                    MC.rotate_to_align(docking_snapshot["yaw_deg"])
                    time.sleep(0.15)
                    MC.strafe_to_align(docking_snapshot["lateral_cm"])
                    time.sleep(0.15)
                    MC.slow_wall_approach()
                    MC.stop_robot()
                    shared["clawBusy"]     = True
                    shared["engageHandle"] = True
                    while not stop_event.is_set() and shared["engageHandle"]:
                        time.sleep(0.02)
                    log.info("MotorsHandler: unload complete.")

                if shared["finilisingState"]:
                    log.info("MotorsHandler: finalising – triggering shutdown.")
                    stop_event.set()
                else:
                    MC.reverse_from_wall()
                    shared["dockingInProcess"] = False
                    log.info("MotorsHandler: reversed from wall – resuming ball search.")

            # ── PRE-CAPTURE ALIGNMENT: rotate in place then re-evaluate ──────
            elif shared["captureAlignPending"]:
                angle_to_center = float(shared["captureAlignAngleDeg"])
                if abs(angle_to_center) > 0.2:
                    MC.stop_robot()
                    time.sleep(0.08)
                    log.info(f"MotorsHandler: pre-capture rotate {angle_to_center:.1f}deg.")
                    if angle_to_center > 0:
                        MC.pivot_right_degrees(abs(angle_to_center))
                    else:
                        MC.pivot_left_degrees(abs(angle_to_center))
                    MC.stop_robot()

                shared["captureAlignPending"] = False
                shared["captureAlignAngleDeg"] = 0.0
                shared["captureAlignRetryCount"] = int(shared["captureAlignRetryCount"]) + 1
                time.sleep(0.03)

            # ── Navigating to AprilTag ────────────────────────────────────────
            elif shared["finilisingState"] or shared["storageFull"]:
                if shared["clawAdjusted"]:
                    shared["clawAdjusted"] = False
                if shared["moveTargetTag"] is not None:
                    MC.apf_move(
                        angle_deg=shared["moveTargetTag"]["angle"],
                        magnitude=shared["moveTargetTag"]["magnitude"],
                    )
                else:
                    MC.pivot_left(35)

            # ── Normal ball collection ─────────────────────────────────────────
            else:
                if shared["moveTargetBall"] is not None:
                    MC.apf_move(
                        angle_deg=shared["moveTargetBall"]["angle"],
                        magnitude=shared["moveTargetBall"]["magnitude"],
                    )
                else:
                    MC.pivot_left(35)

            time.sleep(0.02)
        except Exception as e:
            log.error(f"MotorsHandler Error: {e}")
            MC.stop_robot()
            time.sleep(0.1)

        time.sleep(0.02)
    log.info("MotorsHandler stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Servo Handler // FINISHED BY MARK, RAEF HELPED WITH SERVOCONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
def servo_handler_thread(shared, worker_pause_event, stop_event):
    """
    Controls:
      – Claw servo (ball capture): triggered by engageClaw flag
      – Handle / unload mechanism: triggered by engageHandle flag
    Also performs the one-time initial claw adjustment on startup.

    clawAdjusted convention
    -----------------------
      False → claw needs repositioning; this thread moves it then sets True.
      True  → claw is in position; this thread processes engageClaw/engageHandle.

    clawBusy convention
    -------------------
      True  → a grab or unload is in progress; MotorsHandler must not re-trigger.
      False → ready for the next action.

    pickingInProcess handoff
    ------------------------
      MotorsHandler sets engageClaw=True and clawBusy=True, then blocks on
      pickingInProcess.  This thread performs the grab, updates counts, and
      clears pickingInProcess as the LAST step so MotorsHandler unblocks only
      after the claw is fully reset and the robot is ready to move again.
    """
    log.info("ServoHandler started.")

    while not stop_event.is_set():
        worker_pause_event.wait()
        try:
            # clawAdjusted=False means the claw needs to be moved into position.
            # This covers three cases:
            #   (a) Initial startup  (clawAdjusted starts False, storageFull=False → open)
            #   (b) Storage just became full → hide claw for transit to unload zone
            #   (c) After unload completes  → reopen claw for next collection cycle
            # Also hides the claw when finilisingState is True so it is safely
            # retracted before the robot docks against the wall.
            if not shared["clawAdjusted"]:
                # ── Hide claw for delivery or end-of-game ────────────────────
                if shared["storageFull"] or shared["finilisingState"]:
                    SC.clutch_up()
                    log.info("Claw retracted to up position for delivery/end-of-game.")
                # ── Open claw for collection ──────────────────────────────────
                else:
                    SC.clutch_down()
                    log.info("Claw opened for collection.")
                shared["clawAdjusted"] = True

            else:
                # ── Unload mechanism (engageHandle) ───────────────────────────
                if shared["engageHandle"]:
                    SC.mg90s_turn_by_180_up()
                    time.sleep(2)                    # wait for balls to slide out
                    SC.mg90s_turn_by_180_down()      # reset handle for next unload cycle

                    log.info(
                        f"Unloaded {shared['heldPingPong']} PingPong "
                        f"+ {shared['heldSteel']} steel balls."
                    )
                    shared["heldPingPong"]  = 0
                    shared["heldSteel"]     = 0
                    shared["engageHandle"]  = False
                    shared["storageFull"]   = False
                    # clawAdjusted=False → ServoHandler will call clutch_down() on
                    # the next iteration, reopening the claw for collection.
                    shared["clawAdjusted"]  = False
                    # Release grab lock; MotorsHandler unblocks from dockingInProcess
                    # wait loop once engageHandle is False (cleared above), then
                    # calls reverse_from_wall() and clears dockingInProcess itself.
                    shared["clawBusy"]      = False

                # ── Ball capture claw (engageClaw) ────────────────────────────
                elif shared["engageClaw"]:
                    SC.clutch_grabbing_motion()

                    # ── Update held-ball count ────────────────────────────────
                    # lockedBall is still set here because BallDetector is gated
                    # out by pickingInProcess=True and ImageFrameCapture has
                    # stopped distributing frames.
                    ball_type = shared["lockedBall"]   # read BEFORE clearing
                    if ball_type == "PingPong":
                        shared["heldPingPong"] = shared["heldPingPong"] + 1
                    elif ball_type == "steel":
                        shared["heldSteel"] = shared["heldSteel"] + 1

                    # Clear lockedBall and engageClaw before releasing locks.
                    shared["lockedBall"]  = ""
                    shared["engageClaw"]  = False
                    shared["clawBusy"]    = False

                    log.info(
                        f"Ball captured ({ball_type}). "
                        f"Held: PP={shared['heldPingPong']} steel={shared['heldSteel']}"
                    )

                    STORAGE_CAPACITY = 4  # PingPong=1 unit, steel=0.4 units;
                    totalVal = shared["heldPingPong"] + 0.4 * shared["heldSteel"]

                    if totalVal >= STORAGE_CAPACITY:
                        shared["storageFull"]  = True
                        shared["clawAdjusted"] = False   # triggers claw hide on next loop
                        log.info("Storage full – switching to unload mode.")

                    # Clear pickingInProcess LAST – this is the signal that unblocks
                    # MotorsHandler and re-enables image capture / detection.
                    shared["captureIgnoreUntilS"] = time.time() + POST_PICK_CAPTURE_COOLDOWN_S
                    shared["captureAlignPending"] = False
                    shared["captureAlignAngleDeg"] = 0.0
                    shared["captureAlignRetryCount"] = 0
                    shared["pickingInProcess"] = False
                    log.info("pickingInProcess cleared – robot resuming normal operation.")

            time.sleep(0.02)
        except Exception as e:
            log.error(f"ServoHandler Error: {e}")
            time.sleep(0.1)

        time.sleep(0.02)

    log.info("ServoHandler stopped.")

# ══════════════════════════════════════════════════════════════════════════════
# THREAD  –  Button Handler  (GPIO 4, via gpiozero) // FINISHED
# ══════════════════════════════════════════════════════════════════════════════
def setup_button_handler(shared, worker_pause_event, timer_pause_event, stop_event):
    """
    Configures gpiozero Button callbacks.
    Returns the Button object – the caller MUST keep a reference alive so
    gpiozero's background thread remains active.

    Button behaviour:
      Hold > 5 s  → full reset (pauses EVERYTHING incl. MainTimer, resets alg)
      Short press → if alg started: toggle pause for workers only (not MainTimer)
                    if alg not started: set both events to start the algorithm
    """
    button = Button(BUTTON_GPIO, hold_time=5, bounce_time=0.1)

    def on_held():
        """Triggered when button is held for longer than 5 seconds."""
        log.info("Button held 5 s – performing full reset.")
        shared["btnHeld"] = True

        # Pause ALL processes and threads, including the Main Timer.
        MC.stop_robot()
        worker_pause_event.clear()
        timer_pause_event.clear()

        # Restore every shared flag to its initial value so the robot
        # cannot restart in a broken state (e.g. clawBusy=True blocking all
        # future grabs, storageFull=True sending it straight to the unload
        # zone, or stale engageClaw/engageHandle firing immediately on resume).
        # btnHeld is skipped here and preserved as True so on_released() can
        # correctly identify this event as a long-hold rather than a short press.
        for key, value in SHARED_INITIAL_STATE.items():
            if key == "btnHeld":
                continue
            shared[key] = value

        log.info("Full reset complete – all state restored to initial values.")

    def on_released():
        """Triggered on every button release (short or long)."""
        if shared["btnHeld"]:
            # ── Long-hold release: just clear the flag ───────────────────
            # The robot stays paused; pressing again will start fresh.
            shared["btnHeld"] = False
            log.info("Long-hold released – robot paused and reset. Press again to start.")

        else:
            # ── Short-press release ──────────────────────────────────────
            if not worker_pause_event.is_set() and not timer_pause_event.is_set():
                # Neither event is set → first press; start the algorithm.
                worker_pause_event.set()
                timer_pause_event.set()
                log.info("Algorithm start requested.")
            else:
                if worker_pause_event.is_set():
                    # Currently running → pause workers.
                    MC.stop_robot()
                    worker_pause_event.clear()
                    log.info("Workers paused (Main Timer continues).")
                else:
                    # Currently paused → resume workers.
                    worker_pause_event.set()
                    log.info("Workers resumed.")

    button.when_held     = on_held
    button.when_released = on_released
    return button


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=== Robot Main File starting ===")

    # ── Create inter-process shared state via Manager ─────────────────────
    manager = multiprocessing.Manager()
    shared = manager.dict(SHARED_INITIAL_STATE)

    # ── Setup Events ──────────────────────────────────────────────────────
    # Convention: event.set() = RUNNING, event.clear() = PAUSED
    worker_pause_event = multiprocessing.Event()   # workers (not MainTimer)
    timer_pause_event  = multiprocessing.Event()   # MainTimer only
    stop_event         = multiprocessing.Event()   # terminates all loops

    worker_pause_event.clear()   # start paused until button press
    timer_pause_event.clear()    # start paused until button press

    # ── Frame queues  (ImageFrameCapture → Detectors) ─────────────────────
    # maxsize=1 ensures detectors always consume the latest frame.
    ball_frame_queue = multiprocessing.Queue(maxsize=1)
    tag_frame_queue  = multiprocessing.Queue(maxsize=1)

    # ── Define and start processes ────────────────────────────────────────
    processes = [
        multiprocessing.Process(
            target=main_timer_process,
            args=(shared, timer_pause_event, stop_event),
            name="MainTimer",
            daemon=True,
        ),
        multiprocessing.Process(
            target=ball_detector_process,
            args=(ball_frame_queue, shared, worker_pause_event, stop_event),
            name="BallDetector",
            daemon=True,
        ),
        multiprocessing.Process(
            target=apriltag_detector_process,
            args=(tag_frame_queue, shared, worker_pause_event, stop_event),
            name="AprilTagDetector",
            daemon=True,
        ),
    ]
    for p in processes:
        p.start()
        log.info(f"Started process: {p.name}  (PID {p.pid})")

    # ── Define and start threads ──────────────────────────────────────────
    threads = [
        threading.Thread(
            target=image_capture_thread,
            args=(ball_frame_queue, tag_frame_queue, shared, worker_pause_event, stop_event),
            name="ImageFrameCapture",
            daemon=True,
        ),
        threading.Thread(
            target=motors_handler_thread,
            args=(shared, worker_pause_event, stop_event),
            name="MotorsHandler",
            daemon=True,
        ),
        threading.Thread(
            target=servo_handler_thread,
            args=(shared, worker_pause_event, stop_event),
            name="ServoHandler",
            daemon=True,
        ),
    ]
    for t in threads:
        t.start()
        log.info(f"Started thread: {t.name}")

    # ── Button Handler (gpiozero manages its own internal thread) ─────────
    button = setup_button_handler(
        shared, worker_pause_event, timer_pause_event, stop_event
    )
    log.info(f"ButtonHandler active on GPIO {BUTTON_GPIO}.")

    # ── Main monitor loop ─────────────────────────────────────────────────
    try:
        while not stop_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – shutting down.")

    except Exception as exc:
        log.error(f"Unexpected error in main loop: {exc}", exc_info=True)

    finally:
        log.info("Stopping all processes and threads...")
        stop_event.set()
        worker_pause_event.set()   # unblock any waiting workers
        timer_pause_event.set()    # unblock MainTimer

        for p in processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
                log.warning(f"Force-terminated process: {p.name}")

        manager.shutdown()
        log.info("=== Robot Main File stopped ===")


if __name__ == "__main__":
    # Python 3.14 changed the default multiprocessing start method on Linux
    # from "fork" to "forkserver". This module creates the gpiozero Motor
    # objects at import time, so under forkserver each child re-imports this
    # module and tries to re-claim GPIO pins already held by the parent,
    # crashing the worker. Force "fork" to restore the inherited-state model
    # this design relies on.
    multiprocessing.set_start_method("fork")
    main()