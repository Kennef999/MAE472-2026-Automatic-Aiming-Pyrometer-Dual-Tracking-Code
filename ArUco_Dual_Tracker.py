"""
ArUco Laser Tracker - Raspberry Pi (IMX708)
============================================
Two tracking modes in one file:

  VELOCITY MODE (press V):
    Tracks how fast the ArUco midpoint moves and drives
    actuator proportionally with decay braking.
    Does not require laser dot detection.

  LASER MODE (press L):
    Closed loop PID using laser dot as feedback.
    Finds brightest point in frame after masking ArUco regions.
    Error = laser_dot_y - aruco_midpoint_y
    Motor runs until laser is back on ArUco crosshair.

Controls:
  V = start velocity tracking
  L = start laser tracking (aim laser at crosshair first)
  M = manual / stop tracking
  Q = quit
"""

import time
import threading
import serial
import serial.tools.list_ports
import numpy as np
import cv2
import cv2.aruco as aruco
from picamera2 import Picamera2

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SERIAL_PORT    = None
SERIAL_BAUD    = 115200

CAM_WIDTH      = 1280
CAM_HEIGHT     = 720
CAM_FPS        = 30
LENS_POSITION  = 2.0

DISPLAY_W      = 800
DISPLAY_H      = 480

ARUCO_DICT     = aruco.DICT_4X4_50
MARKER_ID_A    = 0
MARKER_ID_B    = 1

# ── VELOCITY MODE TUNING ────────────────────────────────────────────────────
V_DEAD_BAND_PX  = 1
V_MIN_SPEED     = 25
V_MAX_SPEED     = 100
V_MAX_MOVEMENT  = 25
V_DECAY         = 0.20
V_SMOOTH_ALPHA  = 0.8

# ── LASER MODE TUNING ───────────────────────────────────────────────────────
L_KP            = 1
L_KD            = 3
L_KI            = 0.0
L_DEAD_BAND_PX  = 2
L_MIN_SPEED     = 25
L_MAX_SPEED     = 120
L_I_CLAMP       = 50.0
L_SMOOTH_ALPHA  = 0.5

# Laser detection
LASER_THRESHOLD = 240   # lower if laser not detected, raise if false detections
MARKER_MASK_PAD = 20    # extra pixels masked around each ArUco
LASER_MIN_AREA  = 3

LOST_TIMEOUT_S  = 3.0

# ═══════════════════════════════════════════════════════════════════════════


class ArduinoSerial:
    def __init__(self, port, baud):
        self.port     = port
        self.baud     = baud
        self._ser     = None
        self._tx_lock = threading.Lock()

    def connect(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        print(f"[Serial] Connected on {self.port} @ {self.baud}")

    def send(self, msg):
        if self._ser and self._ser.is_open:
            with self._tx_lock:
                self._ser.write((msg + "\n").encode())

    def drive(self, direction, speed):
        signed = speed if direction == 1 else -speed
        signed = max(-255, min(255, int(signed)))
        self.send(f"DY:{signed}")

    def drive_signed(self, value):
        value = max(-255, min(255, int(value)))
        self.send(f"DY:{value}")

    def stop(self):
        self.send("STOP")

    def close(self):
        if self._ser:
            self._ser.close()


def find_arduino_port():
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        mfr  = (p.manufacturer or "").lower()
        if any(x in desc or x in mfr for x in ["arduino", "ch340", "cp210"]):
            return p.device
    for p in serial.tools.list_ports.comports():
        if "ACM" in p.device or "USB" in p.device:
            return p.device
    raise RuntimeError("No Arduino found. Set SERIAL_PORT manually.")


def open_camera():
    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": CAM_FPS}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)
    cam.set_controls({"AfMode": 0, "LensPosition": LENS_POSITION})
    print(f"[Camera] IMX708 ({CAM_WIDTH}x{CAM_HEIGHT}, focus locked)")
    return cam


def capture_frame(cam):
    frame = cam.capture_array()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def make_detector():
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    params     = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin    = 3
    params.adaptiveThreshWinSizeMax    = 53
    params.adaptiveThreshWinSizeStep   = 4
    params.adaptiveThreshConstant      = 7
    params.minMarkerPerimeterRate      = 0.01
    params.maxMarkerPerimeterRate      = 4.0
    params.polygonalApproxAccuracyRate = 0.08
    params.minCornerDistanceRate       = 0.01
    params.minDistanceToBorder         = 1
    return aruco.ArucoDetector(dictionary, params)


def find_centre(corners, ids, mid):
    if ids is None:
        return None
    for i, m in enumerate(ids.flatten()):
        if m == mid:
            c = corners[i][0]
            return float(np.mean(c[:, 0])), float(np.mean(c[:, 1]))
    return None


def find_laser(gray, corners):
    """
    Find laser dot as brightest point after masking ArUco regions.
    Returns (x, y) or None.
    """
    mask = np.ones_like(gray, dtype=np.uint8) * 255

    if corners:
        for corner in corners:
            pts = corner[0].astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            x1 = max(0, x - MARKER_MASK_PAD)
            y1 = max(0, y - MARKER_MASK_PAD)
            x2 = min(gray.shape[1], x + w + MARKER_MASK_PAD)
            y2 = min(gray.shape[0], y + h + MARKER_MASK_PAD)
            mask[y1:y2, x1:x2] = 0

    masked = cv2.bitwise_and(gray, gray, mask=mask)
    _, thresh = cv2.threshold(masked, LASER_THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid = [c for c in contours if cv2.contourArea(c) >= LASER_MIN_AREA]
    if not valid:
        return None

    brightest = max(valid, key=lambda c: cv2.mean(
        masked, mask=cv2.drawContours(
            np.zeros_like(masked), [c], -1, 255, -1))[0])

    M = cv2.moments(brightest)
    if M["m00"] == 0:
        return None

    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])


def draw_overlay(frame, mode, cA, cB, aruco_mid, laser_pos,
                 display_value, display_label, info_str):
    """
    mode: "STANDBY", "VELOCITY", "LASER"
    display_value: main number to show (movement px/f or error px)
    display_label: label for that number
    info_str: secondary info line
    """
    h, w = frame.shape[:2]
    sy   = h / CAM_HEIGHT

    # ── Top banner — mode indicator ───────────────────────────────────────────
    if mode == "VELOCITY":
        banner_col  = (0, 180, 255)   # orange
        banner_text = "VELOCITY MODE"
    elif mode == "LASER":
        banner_col  = (0, 220, 80)    # green
        banner_text = "LASER MODE"
    else:
        banner_col  = (140, 140, 140) # gray
        banner_text = "STANDBY"

    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, banner_text,
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.4, banner_col, 3)

    # ── ArUco markers ─────────────────────────────────────────────────────────
    if cA and cB:
        cv2.line(frame,
                 (int(cA[0]), int(cA[1] * sy)),
                 (int(cB[0]), int(cB[1] * sy)),
                 (200, 200, 200), 2)
    if cA:
        cv2.circle(frame, (int(cA[0]), int(cA[1] * sy)), 12, (0, 255, 80), -1)
        cv2.putText(frame, "ID0", (int(cA[0]) + 14, int(cA[1] * sy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2)
    if cB:
        cv2.circle(frame, (int(cB[0]), int(cB[1] * sy)), 12, (255, 160, 0), -1)
        cv2.putText(frame, "ID1", (int(cB[0]) + 14, int(cB[1] * sy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 0), 2)

    # ── ArUco midpoint — green crosshair ──────────────────────────────────────
    if aruco_mid:
        mx = int(aruco_mid[0])
        my = int(aruco_mid[1] * sy)
        cv2.drawMarker(frame, (mx, my), (0, 255, 0), cv2.MARKER_CROSS, 36, 3)
        cv2.line(frame, (0, my), (w, my), (0, 255, 0), 1)

    # ── Laser dot — red circle (laser mode only) ──────────────────────────────
    if laser_pos:
        lx = int(laser_pos[0])
        ly = int(laser_pos[1] * sy)
        cv2.circle(frame, (lx, ly), 14, (0, 0, 255), 3)
        cv2.drawMarker(frame, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
        cv2.putText(frame, "LASER", (lx + 16, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ── Bottom debug panel ────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 130), (w, h), (0, 0, 0), -1)

    if mode != "STANDBY" and display_value is not None:
        val_col = (0, 255, 0) if abs(display_value) < 5 else (0, 165, 255)
        cv2.putText(frame, f"{display_label}: {display_value:+.1f}px",
                    (8, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, val_col, 3)
        cv2.putText(frame, info_str,
                    (8, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
    else:
        cv2.putText(frame, info_str if info_str else "---",
                    (8, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 140, 140), 2)

    # ── Keybind hint bar ──────────────────────────────────────────────────────
    cv2.putText(frame, "V=VELOCITY   L=LASER   M=MANUAL   Q=QUIT",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

    return frame


def main():
    print("=== ArUco Dual Mode Tracker ===")
    print("V = velocity mode")
    print("L = laser mode (aim laser at crosshair first)")
    print("M = manual / stop")
    print("Q = quit")

    port = SERIAL_PORT or find_arduino_port()
    arduino = ArduinoSerial(port, SERIAL_BAUD)
    arduino.connect()
    time.sleep(2)

    cam      = open_camera()
    detector = make_detector()

    # ── Shared state ──────────────────────────────────────────────────────────
    mode       = "STANDBY"   # "STANDBY", "VELOCITY", "LASER"
    lost_since = None
    smooth_y   = None

    # ── Velocity state ────────────────────────────────────────────────────────
    prev_y        = None
    movement      = 0.0
    decayed_speed = 0.0
    last_dir      = 1

    # ── Laser / PID state ─────────────────────────────────────────────────────
    prev_error = 0.0
    integral   = 0.0
    error      = None

    info_str       = ""
    display_value  = None
    display_label  = ""
    laser_pos      = None

    cv2.namedWindow("Laser Tracker", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Laser Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            frame = capture_frame(cam)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)

            corners, ids, _ = detector.detectMarkers(gray_eq)

            cA = find_centre(corners, ids, MARKER_ID_A)
            cB = find_centre(corners, ids, MARKER_ID_B)

            both_visible = cA is not None and cB is not None

            aruco_mid = None
            if both_visible:
                raw_y    = (cA[1] + cB[1]) / 2
                raw_x    = (cA[0] + cB[0]) / 2
                smooth_y = L_SMOOTH_ALPHA * raw_y + (1 - L_SMOOTH_ALPHA) * smooth_y \
                           if smooth_y is not None else raw_y
                aruco_mid = (raw_x, smooth_y)

            # ── Laser detection — always running ──────────────────────────────
            laser_pos = find_laser(gray, corners if corners is not None else [])

            # ══════════════════════════════════════════════════════════════════
            #  VELOCITY MODE
            # ══════════════════════════════════════════════════════════════════
            if mode == "VELOCITY":
                if not both_visible:
                    arduino.stop()
                    decayed_speed = 0.0
                    info_str      = "LOST"
                    display_value = None
                    if lost_since is None:
                        lost_since = time.monotonic()
                    elif time.monotonic() - lost_since > LOST_TIMEOUT_S:
                        print("[Standby] Timeout.")
                        mode          = "STANDBY"
                        smooth_y      = None
                        prev_y        = None
                        decayed_speed = 0.0
                else:
                    lost_since = None
                    if prev_y is not None:
                        movement      = smooth_y - prev_y
                        display_value = movement
                        display_label = "MOV"

                        if abs(movement) >= V_DEAD_BAND_PX:
                            scale         = min(abs(movement) / V_MAX_MOVEMENT, 1.0)
                            target_speed  = int(V_MIN_SPEED + (V_MAX_SPEED - V_MIN_SPEED) * scale)
                            last_dir      = 0 if movement > 0 else 1
                            decayed_speed = max(decayed_speed, float(target_speed))
                        else:
                            decayed_speed = decayed_speed * V_DECAY

                        if decayed_speed >= V_MIN_SPEED * 0.5:
                            arduino.drive(last_dir, int(decayed_speed))
                            info_str = f"{'UP' if last_dir==1 else 'DN'}  spd={int(decayed_speed)}  decay={decayed_speed:.0f}"
                        else:
                            decayed_speed = 0.0
                            arduino.stop()
                            info_str = "STOPPED"

                    prev_y = smooth_y

            # ══════════════════════════════════════════════════════════════════
            #  LASER MODE
            # ══════════════════════════════════════════════════════════════════
            elif mode == "LASER":
                all_visible = both_visible and laser_pos is not None

                if not all_visible:
                    arduino.stop()
                    error         = None
                    display_value = None
                    info_str      = "LOST - " + ("no ArUco" if not both_visible else "no laser")
                    if lost_since is None:
                        lost_since = time.monotonic()
                    elif time.monotonic() - lost_since > LOST_TIMEOUT_S:
                        print("[Standby] Timeout.")
                        mode       = "STANDBY"
                        smooth_y   = None
                        prev_error = 0.0
                        integral   = 0.0
                        error      = None
                else:
                    lost_since    = None
                    error         = laser_pos[1] - smooth_y
                    display_value = error
                    display_label = "ERR"

                    if abs(error) < L_DEAD_BAND_PX:
                        arduino.stop()
                        integral   = 0.0
                        prev_error = 0.0
                        info_str   = "ON TARGET"
                    else:
                        integral   = max(-L_I_CLAMP, min(L_I_CLAMP, integral + error))
                        derivative = error - prev_error
                        output     = (L_KP * error) + (L_KI * integral) + (L_KD * derivative)

                        scale      = min(abs(output) / 100.0, 1.0)
                        speed      = int(L_MIN_SPEED + (L_MAX_SPEED - L_MIN_SPEED) * scale)
                        speed      = min(speed, L_MAX_SPEED)

                        signed_cmd = speed if output > 0 else -speed
                        arduino.drive_signed(signed_cmd)
                        info_str = f"CMD:{signed_cmd:+d}"

                    prev_error = error

            # ══════════════════════════════════════════════════════════════════
            #  STANDBY
            # ══════════════════════════════════════════════════════════════════
            else:
                arduino.stop()
                movement      = 0.0
                decayed_speed = 0.0
                error         = None
                display_value = None
                info_str      = "V=velocity  L=laser  to start"

            # ── Draw and display ──────────────────────────────────────────────
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)

            frame   = draw_overlay(frame, mode, cA, cB, aruco_mid, laser_pos,
                                   display_value, display_label, info_str)
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                                 interpolation=cv2.INTER_LINEAR)

            cv2.imshow("Laser Tracker", display)
            key = cv2.waitKey(1) & 0xFF

            # ── Key handling ──────────────────────────────────────────────────
            if key == ord("q"):
                break

            elif key == ord("v") and mode == "STANDBY":
                if both_visible:
                    mode          = "VELOCITY"
                    prev_y        = smooth_y
                    lost_since    = None
                    decayed_speed = 0.0
                    print("[Velocity] Tracking started.")
                else:
                    print("[Error] ArUco markers not visible!")

            elif key == ord("l") and mode == "STANDBY":
                if not both_visible:
                    print("[Error] ArUco markers not visible!")
                elif laser_pos is None:
                    print("[Error] Laser not detected! Lower LASER_THRESHOLD.")
                else:
                    mode       = "LASER"
                    prev_error = 0.0
                    integral   = 0.0
                    error      = 0.0
                    lost_since = None
                    print(f"[Laser] Tracking started.")
                    print(f"  Laser Y={laser_pos[1]:.1f}  ArUco Y={smooth_y:.1f}")

            elif key == ord("m") and mode != "STANDBY":
                prev_mode     = mode
                mode          = "STANDBY"
                prev_y        = None
                decayed_speed = 0.0
                prev_error    = 0.0
                integral      = 0.0
                error         = None
                arduino.stop()
                print(f"[Manual] Stopped {prev_mode} mode.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        arduino.stop()
        arduino.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
