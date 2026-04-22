#!/usr/bin/env python3
"""
Ergonomic Posture Monitor - WITH LIVE WEB VIEW
================================================
Pi 5 + Camera Module 2 + Buzzer (GPIO 17)

Opens a web stream at http://<pi-ip>:5000
View the skeleton overlay + posture score live in your browser.

Usage:
  python3 posture_web.py                        # Live camera
  python3 posture_web.py --video demo.MOV       # Video file

Then open browser: http://<pi-ip>:5000

Hardware:
  Buzzer (+) → GPIO 17 (pin 11)
  Buzzer (-) → GND (pin 9)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import sys
import json
import csv
import threading
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request, send_file


# ═══════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════

BUZZER_PIN = 17
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CALIBRATION_SECONDS = 5
BAD_POSTURE_TRIGGER = 10       # seconds bad before buzzer
GOOD_POSTURE_RECOVERY = 3      # seconds good before buzzer off
POSTURE_SCORE_THRESHOLD = 40   # 0-100, center of the good/bad band
HYSTERESIS_MARGIN = 5          # ± around threshold. Must exceed 45 to flip to bad,
                               # drop below 35 to flip back to good. In between the
                               # state is held, so threshold-line noise stops thrashing.
SCORE_SMOOTHING_ALPHA = 0.2    # EMA weight for the raw score. new = α·raw + (1-α)·prev
                               # Lower = heavier smoothing. 0.2 ≈ 5-frame window @ 10 FPS.
MEDIAPIPE_MODEL = 1            # 0=lite, 1=full, 2=heavy
VIDEO_ROTATION = 0             # Rotate video: 0=none, 90=CW, 180=flip, 270=CCW
SCREENSHOT_DIR = "screenshots"
LOG_FILE = "posture_log.json"
DATA_DIR = "data"
WEB_PORT = 5000

VALID_LABELS = {"unlabeled", "good", "slouch", "forward_head", "tilt"}


# ═══════════════════════════════════════════
#  HARDWARE SETUP
# ═══════════════════════════════════════════

HAS_BUZZER = False
buzzer = None
try:
    from gpiozero import Buzzer
    buzzer = Buzzer(BUZZER_PIN)
    HAS_BUZZER = True
    print(f"[OK] Buzzer on GPIO {BUZZER_PIN}")
except Exception as e:
    print(f"[WARN] No buzzer: {e}")

USE_PICAMERA = False
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    pass

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ═══════════════════════════════════════════
#  SHARED STATE (between monitor thread and web server)
# ═══════════════════════════════════════════

current_frame = None        # Latest processed frame (BGR with overlays)
frame_lock = threading.Lock()
current_label = "unlabeled"  # user-selected label for data collection; updated via /label
monitor_stats = {
    "score": 0,
    "status": "Starting...",
    "fps": 0,
    "alerts": 0,
    "session_start": None,
    "mode": "calibrate",
    "label": "unlabeled",
}

# Mutable runtime config — the sliders in the tuning drawer write here and the
# monitor loop reads it every frame. Defaults match the constants above.
CONFIG = {
    "threshold": POSTURE_SCORE_THRESHOLD,
    "margin": HYSTERESIS_MARGIN,
    "alpha": SCORE_SMOOTHING_ALPHA,
}

# Live snapshot used by /stats — the monitor loop writes here on every frame,
# the web thread reads without locking (dict writes are GIL-atomic).
live_state = {
    "score": 0.0,
    "score_raw": 0.0,
    "is_bad": False,
    "buzzer_on": False,
    "fps": 0.0,
    "alerts": 0,
    "rows": 0,
    "mode": "calibrate",
    "label": "unlabeled",
    "paused": False,
    "calibrating": False,
    "calibration_progress": 0.0,   # 0.0 - 1.0, non-zero only during calibration
    "sub_scores": {"nose_y": 0, "nose_z": 0, "ear_z": 0, "face_size": 0, "shoulder_tilt": 0},
}

# Ring buffer for the sparkline — 60 s of smoothed scores at ~10 FPS.
SPARKLINE_LEN = 600
sparkline_buffer = deque(maxlen=SPARKLINE_LEN)

# 30-minute heatmap as 30 one-minute buckets. Each bucket = {avg, bad_pct, ts}.
HEATMAP_BUCKETS = 30
heatmap_buckets = deque(maxlen=HEATMAP_BUCKETS)
_minute_scores = []           # accumulator for current minute
_minute_bad_count = 0
_minute_start = None

# Pause flag — when True, the state machine is frozen (buzzer won't fire, timers
# don't advance). Video stream, scoring, and CSV logging keep running so the
# session data isn't interrupted.
paused = False


# ═══════════════════════════════════════════
#  CAMERA HELPERS
# ═══════════════════════════════════════════

def _print_no_camera_help():
    print()
    print("=" * 55)
    print("  [ERROR] No live camera detected.")
    print("=" * 55)
    print("  Troubleshooting:")
    print("    1. Check the Pi Camera ribbon cable is seated.")
    print("    2. List video devices:   ls /dev/video*")
    print("    3. Test Pi camera:       libcamera-hello")
    print("    4. Or replay a recording:")
    print("         python3 posture_web.py --video <file.mp4>")
    print("=" * 55)


def init_camera(video_path=None):
    if video_path:
        print(f"[VIDEO] Opening: {video_path}")
        cam = cv2.VideoCapture(video_path)
        if not cam.isOpened():
            print(f"[ERROR] Cannot open: {video_path}")
            sys.exit(1)
        fps = cam.get(cv2.CAP_PROP_FPS)
        total = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[VIDEO] {w}x{h} @ {fps:.0f}fps, {total} frames ({total/fps:.1f}s)")
        return cam, "video"

    if USE_PICAMERA:
        try:
            print("[CAM] Trying picamera2...")
            cam = Picamera2()
            config = cam.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            # OV5647 (Rev 1.3) has slow AWB convergence and skews green/warm
            # under mixed indoor light. Nudge it with explicit controls and
            # give it a few seconds to lock before calibration starts.
            try:
                cam.set_controls({"AwbEnable": True, "AwbMode": 0, "AeEnable": True})
            except Exception as e:
                print(f"[CAM] AWB/AE control not accepted ({e}); using defaults.")
            time.sleep(4)
            print("[CAM] picamera2 ready")
            return cam, "picamera2"
        except Exception as e:
            print(f"[CAM] picamera2 unavailable ({e}); falling back to OpenCV.")

    print("[CAM] Trying OpenCV VideoCapture(0)...")
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if cam.isOpened():
        ret, _ = cam.read()
        if ret:
            print("[CAM] OpenCV VideoCapture ready")
            return cam, "opencv"
        cam.release()

    _print_no_camera_help()
    sys.exit(1)


def get_frame(cam, cam_type):
    if cam_type == "picamera2":
        # On Pi 5 + OV5647 (Rev 1.3), picamera2's "RGB888" delivers bytes in
        # RGB order — use it directly for MediaPipe and convert to BGR for
        # OpenCV drawing / JPEG encoding. (Treating it as BGR produced a
        # teal/orange cast on this sensor.)
        frame_rgb = cam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_rgb, frame_bgr
    else:
        ret, frame_bgr = cam.read()
        if not ret:
            return False, None, None
        # Rotate video frames if needed (phone videos are often sideways)
        if cam_type == "video" and VIDEO_ROTATION != 0:
            if VIDEO_ROTATION == 90:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif VIDEO_ROTATION == 180:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
            elif VIDEO_ROTATION == 270:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return True, frame_rgb, frame_bgr


def release_camera(cam, cam_type):
    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()


# ═══════════════════════════════════════════
#  POSTURE MATH
# ═══════════════════════════════════════════

def get_posture_metrics(landmarks):
    """
    Extract metrics that ACTUALLY CHANGE when posture degrades.
    Based on diagnostic data from real video:
      - nose_y drops 0.10 (head drops in frame)
      - nose_z changes 0.06 (head moves toward camera)
      - ear_z vs shoulder_z changes 0.04 (forward head posture)
      - face_size doubles (closer to camera)
      - shoulder_tilt varies (leaning sideways)
    """
    lm = landmarks.landmark

    nose        = lm[mp_pose.PoseLandmark.NOSE]
    left_ear    = lm[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear   = lm[mp_pose.PoseLandmark.RIGHT_EAR]
    left_eye    = lm[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye   = lm[mp_pose.PoseLandmark.RIGHT_EYE]
    left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    min_vis = 0.5
    key_points = [nose, left_ear, right_ear, left_shoulder, right_shoulder]
    if any(p.visibility < min_vis for p in key_points):
        return None

    ear_mid_x = (left_ear.x + right_ear.x) / 2
    ear_mid_y = (left_ear.y + right_ear.y) / 2
    ear_mid_z = (left_ear.z + right_ear.z) / 2
    shoulder_mid_z = (left_shoulder.z + right_shoulder.z) / 2

    # Face size: distance from nose to ear midpoint (grows when leaning toward camera)
    face_size = math.sqrt((nose.x - ear_mid_x)**2 + (nose.y - ear_mid_y)**2)

    return {
        # Absolute nose Y position in frame (drops when you slouch/lean forward)
        "nose_y": nose.y,
        # Z-depth of nose relative to shoulders (more negative = head further forward)
        "nose_z_depth": nose.z - shoulder_mid_z,
        # Z-depth of ears relative to shoulders (forward head posture)
        "ear_z_depth": ear_mid_z - shoulder_mid_z,
        # Face apparent size (increases when closer to camera)
        "face_size": face_size,
        # Shoulder tilt (leaning to one side)
        "shoulder_tilt": abs(left_shoulder.y - right_shoulder.y),
    }


def compute_posture_score(current, baseline):
    """
    Compute posture score: 0 = matches baseline, 100 = terrible.
    Multipliers calibrated from real diagnostic data:
      nose_y change of ~0.10 → score ~80
      nose_z change of ~0.06 → score ~60
      ear_z change of ~0.04 → score ~55
      face_size change of ~0.02 → score ~60

    Returns (total, breakdown) so the UI can show per-metric contributions.
    `breakdown` is each metric's already-weighted contribution to the total.
    """
    nose_y_drop  = min(max(0, baseline["nose_y"] - current["nose_y"]) * 800, 100)
    z_forward    = min(max(0, abs(current["nose_z_depth"]) - abs(baseline["nose_z_depth"])) * 1000, 100)
    ear_forward  = min(max(0, abs(current["ear_z_depth"]) - abs(baseline["ear_z_depth"])) * 1500, 100)
    size_growth  = min(max(0, current["face_size"] - baseline["face_size"]) * 3000, 100)
    tilt_dev     = min(max(0, current["shoulder_tilt"] - baseline["shoulder_tilt"]) * 400, 100)

    # Weighted contributions — each metric's share of the final score.
    breakdown = {
        "nose_y":        nose_y_drop * 0.30,
        "nose_z":        z_forward   * 0.25,
        "ear_z":         ear_forward * 0.20,
        "face_size":     size_growth * 0.15,
        "shoulder_tilt": tilt_dev    * 0.10,
    }
    total = min(sum(breakdown.values()), 100)
    return total, breakdown


def compute_posture_score_research(current):
    """
    Score using absolute thresholds from ergonomics literature — no personal
    baseline required. Works from the first frame.

    Signals used (all computable from a front-facing camera):
      - shoulder_tilt (|L_sh.y - R_sh.y|):  lateral shoulder asymmetry.
          Raine & Twomey (1997): normal adults sit within ~3 deg of level.
          In MediaPipe normalized coords, >0.02 corresponds to ~5 deg in a
          typical 640x480 framing — that's our "flag" threshold.
      - ear_z_depth (ear.z - shoulder.z): negative = ears closer to camera
          than shoulders = forward head posture (Yip 2008).
      - nose_z_depth (nose.z - shoulder.z): severity of forward lean;
          Hansraj (2014) showed cervical load grows sharply with forward
          head flexion.

    nose_y and face_size are intentionally excluded here — they depend on
    where the user sits relative to the camera, so they have no universal
    threshold without calibration.

    Returns (total, breakdown) mirroring compute_posture_score. Breakdown keys
    match the self-calibrate version so the UI can render the same 5 bars —
    the two unused metrics just read 0.
    """
    tilt_score = min(max(0.0, current["shoulder_tilt"] - 0.02) * 2500, 100)
    ear_score  = min(max(0.0, -current["ear_z_depth"]) * 1000, 100)
    nose_score = min(max(0.0, -current["nose_z_depth"]) * 700, 100)

    breakdown = {
        "nose_y":        0.0,
        "nose_z":        nose_score * 0.30,
        "ear_z":         ear_score  * 0.40,
        "face_size":     0.0,
        "shoulder_tilt": tilt_score * 0.30,
    }
    total = min(sum(breakdown.values()), 100)
    return total, breakdown


# ═══════════════════════════════════════════
#  BUZZER
# ═══════════════════════════════════════════

def buzzer_on():
    if HAS_BUZZER and buzzer:
        buzzer.on()

def buzzer_off():
    if HAS_BUZZER and buzzer:
        buzzer.off()

def buzzer_beep(duration=0.15):
    if HAS_BUZZER and buzzer:
        buzzer.on()
        time.sleep(duration)
        buzzer.off()


# ═══════════════════════════════════════════
#  SCREENSHOT & LOGGING
# ═══════════════════════════════════════════

def save_screenshot(frame_bgr, score, label="alert"):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOT_DIR}/{label}_{ts}_score{int(score)}.jpg"
    overlay = frame_bgr.copy()
    cv2.putText(overlay, f"Score: {int(score)}/100 - {label.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(filename, overlay)
    return filename


def log_event(event_type, score, metrics):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "score": round(score, 1),
        "metrics": {k: round(v, 4) for k, v in metrics.items()}
    }
    events = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                events = json.load(f)
        except:
            events = []
    events.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(events, f, indent=2)


# ═══════════════════════════════════════════
#  DRAW OVERLAY ON FRAME
# ═══════════════════════════════════════════

def draw_overlay(frame_bgr, landmarks, score, status, is_bad=False):
    """Draw skeleton, score bar, and status label on frame.

    ``score`` should be the smoothed score (what the state machine sees),
    and ``is_bad`` is the hysteresis-stable state — passed in so the UI
    stays in sync with the authoritative decision, instead of re-evaluating
    a raw threshold that might flicker near the line.
    """
    h, w = frame_bgr.shape[:2]

    # Draw MediaPipe skeleton
    if landmarks:
        mp_draw.draw_landmarks(
            frame_bgr, landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

    # --- Score bar (top left) ---
    bar_x, bar_y, bar_w, bar_h = 15, 15, 250, 35
    # Background
    cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (100, 100, 100), 1)
    # Fill
    fill_w = int(score / 100 * bar_w)
    if score < 30:
        bar_color = (0, 200, 0)       # green
    elif score < 60:
        bar_color = (0, 180, 255)     # orange
    else:
        bar_color = (0, 0, 230)       # red
    cv2.rectangle(frame_bgr, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  bar_color, -1)
    # Score text on bar
    cv2.putText(frame_bgr, f"{int(score)}/100", (bar_x + 8, bar_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Status label (top right) ---
    if is_bad:
        label = "BAD POSTURE"
        bg_color = (0, 0, 180)
    else:
        label = "GOOD POSTURE"
        bg_color = (0, 150, 0)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    label_x = w - tw - 25
    label_y = 40
    cv2.rectangle(frame_bgr, (label_x - 8, label_y - th - 8),
                  (label_x + tw + 8, label_y + 8), bg_color, -1)
    cv2.putText(frame_bgr, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # --- Detailed status (bottom left) ---
    cv2.putText(frame_bgr, status, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame_bgr


# ═══════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════

def calibrate_from_video(cam, cam_type, pose_detector):
    fps = cam.get(cv2.CAP_PROP_FPS) or 30
    cal_frames = int(fps * CALIBRATION_SECONDS)
    samples = []

    live_state["calibrating"] = True

    for i in range(cal_frames):
        ret, frame_rgb, frame_bgr = get_frame(cam, cam_type)
        if not ret:
            break
        results = pose_detector.process(frame_rgb)
        if results.pose_landmarks:
            metrics = get_posture_metrics(results.pose_landmarks)
            if metrics is not None:
                samples.append(metrics)

            # Show calibration on web view
            overlay = draw_overlay(frame_bgr, results.pose_landmarks, 0,
                                   f"CALIBRATING... {i+1}/{cal_frames}")
            cv2.putText(overlay, "SIT UP STRAIGHT", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            with frame_lock:
                global current_frame
                current_frame = overlay.copy()

        live_state["calibration_progress"] = (i + 1) / max(1, cal_frames)
        print(f"   Calibrating... {i+1}/{cal_frames} ({len(samples)} valid)", end="\r")

    print()
    live_state["calibrating"] = False
    live_state["calibration_progress"] = 0.0

    if len(samples) < 5:
        print(f"[ERROR] Only {len(samples)} valid samples.")
        return None

    baseline = {}
    for key in samples[0]:
        baseline[key] = float(np.mean([s[key] for s in samples]))

    cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, _, frame_bgr = get_frame(cam, cam_type)
    if ret:
        save_screenshot(frame_bgr, 0, "calibration")
    cam.set(cv2.CAP_PROP_POS_FRAMES, cal_frames)

    print(f"[OK] Calibration complete! ({len(samples)} samples)")
    print(f"     Nose Y pos:    {baseline['nose_y']:.4f}")
    print(f"     Nose Z depth:  {baseline['nose_z_depth']:.4f}")
    print(f"     Ear Z depth:   {baseline['ear_z_depth']:.4f}")
    print(f"     Face size:     {baseline['face_size']:.4f}")
    print(f"     Shoulder tilt: {baseline['shoulder_tilt']:.4f}")
    return baseline


def calibrate_live(cam, cam_type, pose_detector):
    global current_frame

    print(f"\n{'='*55}")
    print(f"   SIT UP STRAIGHT! Calibrating for {CALIBRATION_SECONDS}s...")
    print(f"{'='*55}")

    live_state["calibrating"] = True
    live_state["calibration_progress"] = 0.0

    for i in range(3, 0, -1):
        # Show countdown on web view
        ret, frame_rgb, frame_bgr = get_frame(cam, cam_type)
        if ret:
            overlay = frame_bgr.copy()
            cv2.putText(overlay, f"Starting in {i}...", (150, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            with frame_lock:
                current_frame = overlay.copy()
        buzzer_beep(0.1)
        time.sleep(1)

    buzzer_beep(0.3)
    samples = []
    start = time.time()

    while time.time() - start < CALIBRATION_SECONDS:
        ret, frame_rgb, frame_bgr = get_frame(cam, cam_type)
        if not ret:
            continue
        results = pose_detector.process(frame_rgb)
        if results.pose_landmarks:
            metrics = get_posture_metrics(results.pose_landmarks)
            if metrics is not None:
                samples.append(metrics)

            elapsed = time.time() - start
            remaining = CALIBRATION_SECONDS - elapsed
            live_state["calibration_progress"] = min(1.0, elapsed / CALIBRATION_SECONDS)
            overlay = draw_overlay(frame_bgr, results.pose_landmarks, 0,
                                   f"CALIBRATING... {remaining:.1f}s left")
            cv2.putText(overlay, "HOLD STILL - GOOD POSTURE", (80, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            with frame_lock:
                current_frame = overlay.copy()

    live_state["calibrating"] = False
    live_state["calibration_progress"] = 0.0

    if len(samples) < 10:
        print(f"[ERROR] Only {len(samples)} valid samples.")
        return None

    baseline = {}
    for key in samples[0]:
        baseline[key] = float(np.mean([s[key] for s in samples]))

    buzzer_beep(0.1)
    time.sleep(0.1)
    buzzer_beep(0.1)

    ret, _, frame_bgr = get_frame(cam, cam_type)
    if ret:
        save_screenshot(frame_bgr, 0, "calibration")

    print(f"[OK] Calibration complete! ({len(samples)} samples)")
    return baseline


# ═══════════════════════════════════════════
#  DATA COLLECTION (per-session CSV)
# ═══════════════════════════════════════════

CSV_FIELDS = [
    "timestamp", "mode", "label", "score", "score_smoothed", "is_bad",
    "nose_y", "nose_z_depth", "ear_z_depth", "face_size", "shoulder_tilt",
]


class SessionLogger:
    """Append one CSV row per processed frame. Line-buffered so crashes don't
    lose the tail of the session.

    Logs both the raw per-frame score and the smoothed score used by the
    state machine — raw is preserved so later analysis can re-smooth with
    a different window if needed.
    """

    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.rows = 0

    def log(self, mode, label, score_raw, score_smoothed, is_bad, metrics):
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "mode": mode,
            "label": label,
            "score": round(score_raw, 2),
            "score_smoothed": round(score_smoothed, 2),
            "is_bad": int(bool(is_bad)),
        }
        for key in ("nose_y", "nose_z_depth", "ear_z_depth", "face_size", "shoulder_tilt"):
            row[key] = round(metrics[key], 4)
        self.writer.writerow(row)
        self.rows += 1

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


# ═══════════════════════════════════════════
#  MONITOR THREAD
# ═══════════════════════════════════════════

def monitor_loop(video_path=None, mode="calibrate"):
    """Main posture monitoring — runs in a background thread."""
    global current_frame

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    cam, cam_type = init_camera(video_path=video_path)

    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MEDIAPIPE_MODEL,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Calibrate (self-calibrate mode only; research mode uses absolute thresholds)
    baseline = None
    if mode == "calibrate":
        if cam_type == "video":
            print(f"[VIDEO] Auto-calibrating from first {CALIBRATION_SECONDS}s...")
            baseline = calibrate_from_video(cam, cam_type, pose_detector)
        else:
            baseline = calibrate_live(cam, cam_type, pose_detector)

        if baseline is None:
            release_camera(cam, cam_type)
            return
    else:
        print(f"[RESEARCH] Using literature-backed thresholds (no calibration).")

    monitor_stats["mode"] = mode
    live_state["mode"] = mode

    print(f"\n[MONITORING] Live! Open http://<pi-ip>:{WEB_PORT} in browser")
    print(f"  Mode: {mode} | Threshold: {CONFIG['threshold']}/100 | Trigger: {BAD_POSTURE_TRIGGER}s")
    print(f"  Press Ctrl+C to stop\n")

    # Open per-session CSV for data collection
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(
        DATA_DIR,
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}.csv",
    )
    session_log = SessionLogger(csv_path)
    print(f"[DATA] Logging frames to {csv_path}")

    # State
    bad_start = None
    good_start = None
    buzzer_active = False
    smoothed_score = None   # EMA-smoothed score; None until first valid frame
    is_bad_state = False    # Hysteresis-stable classification; only flips when
                            # smoothed score clearly leaves the previous band
    frame_count = 0
    fps_timer = time.time()
    alert_screenshot_taken = False   # one alert snapshot per session, full stop
    session_scores = []
    session_score_sum = 0.0      # O(1) running mean accumulator
    session_bad_count = 0        # O(1) running bad-frame counter
    alert_count = 0
    session_start = time.time()
    live_state["session_start"] = session_start
    live_state["csv_path"] = csv_path
    live_state["summary_avg"] = 0.0
    live_state["summary_bad_pct"] = 0.0

    try:
        while True:
            ret, frame_rgb, frame_bgr = get_frame(cam, cam_type)
            if not ret:
                if cam_type == "video":
                    # Loop video for demo purposes
                    cam.set(cv2.CAP_PROP_POS_FRAMES, int((cam.get(cv2.CAP_PROP_FPS) or 30) * CALIBRATION_SECONDS))
                    print("\n  [VIDEO] Looping...")
                    continue
                continue

            frame_count += 1
            now = time.time()

            if cam_type == "video":
                video_fps = cam.get(cv2.CAP_PROP_FPS) or 30
                time.sleep(1.0 / video_fps)

            results = pose_detector.process(frame_rgb)

            if results.pose_landmarks:
                metrics = get_posture_metrics(results.pose_landmarks)

                if metrics is None:
                    continue

                if mode == "research":
                    raw_score, sub_scores = compute_posture_score_research(metrics)
                else:
                    raw_score, sub_scores = compute_posture_score(metrics, baseline)

                # EMA smoothing — kills MediaPipe's Z-axis jitter before it
                # reaches the state machine. Without this the score can swing
                # ±10 points between frames on a motionless subject.
                alpha = CONFIG["alpha"]
                if smoothed_score is None:
                    smoothed_score = raw_score
                else:
                    smoothed_score = alpha * raw_score + (1.0 - alpha) * smoothed_score

                # Hysteresis — only flip state when the smoothed score clearly
                # leaves the previous band. Between (threshold - margin) and
                # (threshold + margin) the current state is held, which stops
                # noise near the line from thrashing BAD↔GOOD every frame.
                thr = CONFIG["threshold"]
                margin = CONFIG["margin"]
                if is_bad_state:
                    if smoothed_score < (thr - margin):
                        is_bad_state = False
                else:
                    if smoothed_score > (thr + margin):
                        is_bad_state = True

                is_bad = is_bad_state
                session_scores.append(smoothed_score)
                session_score_sum += smoothed_score
                if smoothed_score > CONFIG["threshold"]:
                    session_bad_count += 1

                # Per-frame data collection row — log both raw and smoothed so
                # post-hoc analysis can re-smooth with a different window.
                session_log.log(mode, current_label, raw_score, smoothed_score,
                                is_bad, metrics)

                # --- Telemetry buffers ---
                sparkline_buffer.append(round(smoothed_score, 1))

                # Roll up 60-second heatmap buckets using wall-clock time so
                # FPS variation doesn't distort bucket widths.
                global _minute_start, _minute_bad_count
                if _minute_start is None:
                    _minute_start = now
                _minute_scores.append(smoothed_score)
                if is_bad:
                    _minute_bad_count += 1
                if now - _minute_start >= 60.0:
                    heatmap_buckets.append({
                        "avg": round(float(np.mean(_minute_scores)), 1),
                        "bad_pct": round(100.0 * _minute_bad_count / max(1, len(_minute_scores)), 1),
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    })
                    _minute_scores.clear()
                    _minute_bad_count = 0
                    _minute_start = now

                # --- State machine (pause freezes the timers + buzzer but
                # scoring/logging/telemetry keep flowing).
                if paused:
                    # Silence any active buzzer and clear the timers so the
                    # session resumes cleanly instead of inheriting stale state.
                    if buzzer_active:
                        buzzer_off()
                        buzzer_active = False
                    bad_start = None
                    good_start = None
                    status = "PAUSED — monitoring halted"
                elif is_bad:
                    good_start = None
                    if bad_start is None:
                        bad_start = now
                    bad_duration = now - bad_start

                    if bad_duration >= BAD_POSTURE_TRIGGER and not buzzer_active:
                        buzzer_active = True
                        alert_count += 1
                        buzzer_on()
                        log_event("alert_start", smoothed_score, metrics)
                        if not alert_screenshot_taken:
                            save_screenshot(frame_bgr, smoothed_score, "alert")
                            alert_screenshot_taken = True

                    if buzzer_active:
                        status = f"BUZZER ON | Alert #{alert_count}"
                    else:
                        status = f"Bad posture: {bad_duration:.0f}s / {BAD_POSTURE_TRIGGER}s"
                else:
                    bad_start = None
                    if buzzer_active:
                        if good_start is None:
                            good_start = now
                        good_duration = now - good_start
                        if good_duration >= GOOD_POSTURE_RECOVERY:
                            buzzer_active = False
                            good_start = None
                            buzzer_off()
                            log_event("recovered", smoothed_score, metrics)
                            # No recovery screenshot — we only keep calibration
                            # + the first alert of the session.
                        status = f"Recovering: {good_duration:.0f}s / {GOOD_POSTURE_RECOVERY}s"
                    else:
                        good_start = None
                        status = "Good posture"

                # FPS calc
                elapsed = now - fps_timer
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Build status line
                full_status = f"FPS: {fps:.1f} | {status}"

                # Draw overlay on frame — pass smoothed score + stable state so
                # the UI matches what the state machine is actually acting on.
                display_frame = draw_overlay(frame_bgr.copy(), results.pose_landmarks,
                                             smoothed_score, full_status, is_bad)

                # Update shared frame for web stream
                with frame_lock:
                    current_frame = display_frame

                # Update stats
                monitor_stats["score"] = smoothed_score
                monitor_stats["status"] = status
                monitor_stats["fps"] = fps
                monitor_stats["alerts"] = alert_count
                monitor_stats["label"] = current_label

                # Dashboard live snapshot — single dict drives /stats polling
                live_state["score"] = round(smoothed_score, 1)
                live_state["score_raw"] = round(raw_score, 1)
                live_state["is_bad"] = bool(is_bad)
                live_state["buzzer_on"] = bool(buzzer_active)
                live_state["fps"] = round(fps, 1)
                live_state["alerts"] = alert_count
                live_state["rows"] = session_log.rows
                live_state["mode"] = mode
                live_state["label"] = current_label
                live_state["paused"] = paused
                live_state["sub_scores"] = {k: round(v, 1) for k, v in sub_scores.items()}

                # Running session summary — O(1) from the accumulators above.
                n = len(session_scores)
                if n:
                    live_state["summary_avg"]     = round(session_score_sum / n, 1)
                    live_state["summary_bad_pct"] = round(100.0 * session_bad_count / n, 1)

                # Terminal output — show smoothed score with raw in parens for
                # quick sanity-check on how noisy the underlying signal is.
                if frame_count % 10 == 0:
                    bar_len = max(0, min(20, int(smoothed_score / 5)))
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    print(f"  [{bar}] {smoothed_score:5.1f} (raw {raw_score:5.1f}) | "
                          f"{status:30s} | FPS:{fps:.1f}", end="\r")

            else:
                if buzzer_active:
                    buzzer_off()
                    buzzer_active = False
                bad_start = None
                good_start = None
                # Reset EMA + hysteresis so the next appearance starts fresh
                # instead of inheriting stale state from the previous sit.
                smoothed_score = None
                is_bad_state = False

                live_state["is_bad"] = False
                live_state["buzzer_on"] = False
                live_state["score"] = 0.0
                live_state["score_raw"] = 0.0

                # Still update frame even without detection
                cv2.putText(frame_bgr, "No person detected", (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                with frame_lock:
                    current_frame = frame_bgr.copy()

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        buzzer_off()
        pose_detector.close()
        release_camera(cam, cam_type)
        session_log.close()

        duration = time.time() - session_start
        print(f"\n{'='*55}")
        print(f"   SESSION SUMMARY")
        print(f"{'='*55}")
        if session_scores:
            avg = np.mean(session_scores)
            bad_pct = sum(1 for s in session_scores if s > CONFIG["threshold"]) / len(session_scores) * 100
            print(f"   Mode:         {mode}")
            print(f"   Duration:     {duration/60:.1f} min")
            print(f"   Avg score:    {avg:.1f}/100")
            print(f"   Bad posture:  {bad_pct:.1f}%")
            print(f"   Alerts:       {alert_count}")
            print(f"   Avg FPS:      {len(session_scores)/duration:.1f}")
            print(f"   Rows logged:  {session_log.rows} → {csv_path}")
        print(f"{'='*55}\n")


# ═══════════════════════════════════════════
#  FLASK WEB SERVER
# ═══════════════════════════════════════════

app = Flask(__name__)

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Posture Monitor</title>
<style>
  :root {
    --bg:        hsl(0 0% 3.9%);
    --fg:        hsl(0 0% 98%);
    --card:     hsl(0 0% 7%);
    --card-hi:  hsl(0 0% 10%);
    --border:   hsl(0 0% 14.9%);
    --muted:    hsl(0 0% 14.9%);
    --mf:       hsl(0 0% 63.9%);
    --primary:  hsl(0 0% 98%);
    --primary-fg: hsl(0 0% 9%);
    --destructive: hsl(0 72% 51%);
    --success:  hsl(142 71% 45%);
    --warning:  hsl(38 92% 50%);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: var(--bg); color: var(--fg); height: 100%; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    font-size: 14px; line-height: 1.5;
    padding: 24px;
  }
  .mono { font-family: "JetBrains Mono", "SF Mono", Menlo, monospace; }
  .container { max-width: 1280px; margin: 0 auto; }

  /* ---------- Top bar ---------- */
  .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  .topbar h1 { font-size: 18px; font-weight: 600; letter-spacing: -0.01em; }
  .topbar .sub { color: var(--mf); font-size: 12px; margin-top: 2px; }
  .topbar-actions { display: flex; gap: 8px; align-items: center; }

  /* ---------- Cards ---------- */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
  }
  .card h2 {
    font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
    color: var(--mf); text-transform: uppercase; margin-bottom: 12px;
  }

  /* ---------- Main grid: video + side panel ---------- */
  .grid { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px; }
  .grid-lower { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 20px; }

  .video-card {
    padding: 0; overflow: hidden; background: #000;
    display: flex; align-items: center; justify-content: center;
    min-height: 240px;
  }
  .video-card img {
    display: block;
    max-width: 100%;
    max-height: 70vh;
    width: auto;
    height: auto;
    object-fit: contain;   /* preserve aspect ratio for portrait OR landscape */
  }

  .stat-row { display: flex; justify-content: space-between; align-items: baseline; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .stat-row:last-child { border-bottom: none; }
  .stat-row .k { color: var(--mf); font-size: 12px; }
  .stat-row .v { font-size: 14px; }
  .score-big { font-size: 48px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 6px; }
  .score-big .unit { font-size: 16px; color: var(--mf); font-weight: 400; margin-left: 4px; }

  /* ---------- Badges ---------- */
  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 6px; font-size: 11px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; }
  .badge-outline { border: 1px solid var(--border); color: var(--fg); background: transparent; }
  .badge-success { background: hsl(142 71% 45% / 0.15); color: var(--success); border: 1px solid hsl(142 71% 45% / 0.3); }
  .badge-destructive { background: hsl(0 72% 51% / 0.15); color: var(--destructive); border: 1px solid hsl(0 72% 51% / 0.3); }
  .badge-warning { background: hsl(38 92% 50% / 0.15); color: var(--warning); border: 1px solid hsl(38 92% 50% / 0.3); }
  .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; background: currentColor; }

  /* ---------- Buttons ---------- */
  .btn { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer; border: 1px solid var(--border); background: transparent; color: var(--fg); transition: all 0.12s ease; }
  .btn:hover { background: var(--muted); }
  .btn-primary { background: var(--primary); color: var(--primary-fg); border-color: var(--primary); }
  .btn-primary:hover { background: hsl(0 0% 90%); }
  .btn-destructive { background: var(--destructive); color: var(--fg); border-color: var(--destructive); }
  .btn-active { background: var(--primary); color: var(--primary-fg); border-color: var(--primary); }
  .btn-xs { padding: 4px 10px; font-size: 12px; }

  /* ---------- Sparkline ---------- */
  .spark { width: 100%; height: 50px; display: block; }
  .spark path.main { stroke: var(--fg); stroke-width: 1.5; fill: none; }
  .spark path.overlay { stroke: var(--mf); stroke-width: 1; fill: none; stroke-dasharray: 3 3; opacity: 0.6; }
  .spark line.threshold { stroke: var(--mf); stroke-width: 0.5; stroke-dasharray: 2 3; opacity: 0.4; }

  /* ---------- Metric bars ---------- */
  .metric { display: grid; grid-template-columns: 100px 1fr 40px; gap: 10px; align-items: center; padding: 5px 0; font-size: 12px; }
  .metric .label { color: var(--mf); }
  .metric .bar { height: 6px; background: var(--muted); border-radius: 3px; overflow: hidden; }
  .metric .bar .fill { height: 100%; background: var(--fg); transition: width 0.3s ease; }
  .metric .val { text-align: right; color: var(--mf); }

  /* ---------- Labels ---------- */
  .label-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .label-row .kbd { font-size: 10px; color: var(--mf); margin-left: 4px; }

  /* ---------- Heatmap ---------- */
  .heatmap { display: grid; grid-template-columns: repeat(30, 1fr); gap: 2px; height: 28px; }
  .heatmap-cell { background: var(--muted); border-radius: 2px; transition: background 0.2s ease; }
  .heatmap-cell.empty { background: hsl(0 0% 10%); }

  /* ---------- Tuning drawer ---------- */
  .tuning { display: none; margin-top: 20px; }
  .tuning.open { display: block; }
  .slider-row { display: grid; grid-template-columns: 120px 1fr 60px; gap: 12px; align-items: center; padding: 8px 0; font-size: 13px; }
  .slider-row input[type=range] { width: 100%; accent-color: var(--primary); }
  .slider-row .slider-val { font-family: "JetBrains Mono", monospace; text-align: right; color: var(--mf); }

  /* ---------- Screenshot gallery ---------- */
  .gallery { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 8px; }
  .gallery img { height: 100px; border-radius: 6px; border: 1px solid var(--border); }

  /* ---------- Sessions list ---------- */
  .session-item { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; font-size: 12px; border-bottom: 1px solid var(--border); }
  .session-item:last-child { border-bottom: none; }
  .session-item .name { color: var(--mf); }
  .session-item button { background: transparent; border: 1px solid var(--border); color: var(--fg); padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; }
  .session-item button.active { background: var(--primary); color: var(--primary-fg); }

  /* ---------- Calibration overlay ---------- */
  .calibration-overlay {
    position: fixed; inset: 0; background: hsl(0 0% 3.9% / 0.92);
    display: none; align-items: center; justify-content: center; z-index: 100;
    backdrop-filter: blur(6px);
  }
  .calibration-overlay.show { display: flex; }
  .calibration-card { text-align: center; }
  .ring {
    width: 180px; height: 180px; margin: 0 auto 24px;
    transform: rotate(-90deg);
  }
  .ring-bg { stroke: var(--muted); stroke-width: 8; fill: none; }
  .ring-fg { stroke: var(--fg); stroke-width: 8; fill: none; stroke-linecap: round; transition: stroke-dashoffset 0.2s linear; }
  .cal-title { font-size: 28px; font-weight: 600; margin-bottom: 8px; }
  .cal-sub { color: var(--mf); font-size: 14px; }
</style>
</head>

<body>

<!-- Calibration overlay — shown while self-calibrate is running -->
<div class="calibration-overlay" id="cal-overlay">
  <div class="calibration-card">
    <svg class="ring" viewBox="0 0 100 100">
      <circle class="ring-bg" cx="50" cy="50" r="45" />
      <circle class="ring-fg" id="cal-ring" cx="50" cy="50" r="45"
              stroke-dasharray="282.7" stroke-dashoffset="282.7" />
    </svg>
    <div class="cal-title">Sit up straight</div>
    <div class="cal-sub">Calibrating baseline · <span class="mono" id="cal-pct">0%</span></div>
  </div>
</div>

<div class="container">

  <!-- ───── Top bar ───── -->
  <div class="topbar">
    <div>
      <h1>Ergonomic Posture Monitor</h1>
      <div class="sub">Raspberry Pi 5 · MediaPipe Pose · <span id="clock" class="mono">00:00</span></div>
    </div>
    <div class="topbar-actions">
      <span id="mode-badge" class="badge badge-outline"><span class="dot"></span>—</span>
      <button class="btn btn-xs" onclick="togglePause()"><span id="pause-label">Pause</span></button>
      <button class="btn btn-xs" onclick="toggleTuning()">Tune</button>
      <a class="btn btn-xs" id="download-link" href="/download/csv" download>Download CSV</a>
    </div>
  </div>

  <!-- ───── Main grid: video | side panel ───── -->
  <div class="grid">
    <div class="card video-card"><img src="/video_feed" alt="Live Stream"></div>

    <div>
      <div class="card">
        <h2>Score</h2>
        <div class="score-big mono"><span id="score">0</span><span class="unit">/ 100</span></div>
        <div style="margin-top: 6px;">
          <span id="state-badge" class="badge badge-outline"><span class="dot"></span>—</span>
        </div>
        <div style="margin-top: 16px;">
          <svg class="spark" id="spark" viewBox="0 0 600 50" preserveAspectRatio="none">
            <line id="spark-thr-line" class="threshold" x1="0" y1="0" x2="600" y2="0" />
            <path class="overlay" id="spark-overlay" d="" />
            <path class="main" id="spark-path" d="" />
          </svg>
          <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--mf); margin-top: 2px;">
            <span>60 s ago</span><span>now</span>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h2>Telemetry</h2>
        <div class="stat-row"><span class="k">FPS</span><span class="v mono" id="fps">0.0</span></div>
        <div class="stat-row"><span class="k">Alerts</span><span class="v mono" id="alerts">0</span></div>
        <div class="stat-row"><span class="k">Rows logged</span><span class="v mono" id="rows">0</span></div>
        <div class="stat-row"><span class="k">Mode</span><span class="v mono" id="mode">—</span></div>
        <div class="stat-row"><span class="k">Buzzer</span><span class="v mono" id="buzzer">off</span></div>
      </div>

      <div class="card" style="margin-top: 16px;">
        <h2>Session Summary</h2>
        <div class="stat-row"><span class="k">Average score</span><span class="v mono" id="summary-avg">0.0</span></div>
        <div class="stat-row"><span class="k">Bad posture</span><span class="v mono" id="summary-bad">0.0%</span></div>
        <div class="stat-row"><span class="k">Duration</span><span class="v mono" id="summary-dur">00:00</span></div>
      </div>
    </div>
  </div>

  <!-- ───── Lower row: metrics, heatmap, sessions ───── -->
  <div class="grid-lower">
    <div class="card">
      <h2>Metric Contributions</h2>
      <div id="metrics"></div>
    </div>

    <div class="card">
      <h2>Last 30 min · Bad-posture density</h2>
      <div class="heatmap" id="heatmap"></div>
      <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--mf); margin-top: 6px;">
        <span>30 min ago</span><span>now</span>
      </div>
    </div>

    <div class="card">
      <h2>Past Sessions</h2>
      <div id="sessions" style="max-height: 180px; overflow-y: auto;"></div>
    </div>
  </div>

  <!-- ───── Labels row ───── -->
  <div class="card" style="margin-top: 20px;">
    <h2>Label Current Pose</h2>
    <div class="label-row" id="label-buttons">
      <button class="btn" data-label="good" onclick="setLabel('good')">Good<span class="kbd">1</span></button>
      <button class="btn" data-label="slouch" onclick="setLabel('slouch')">Slouch<span class="kbd">2</span></button>
      <button class="btn" data-label="forward_head" onclick="setLabel('forward_head')">Forward Head<span class="kbd">3</span></button>
      <button class="btn" data-label="tilt" onclick="setLabel('tilt')">Tilt<span class="kbd">4</span></button>
      <button class="btn" data-label="unlabeled" onclick="setLabel('unlabeled')">Clear<span class="kbd">0</span></button>
      <span style="margin-left: auto; color: var(--mf); font-size: 12px;">Current: <span id="current-label" class="mono">—</span></span>
    </div>
  </div>

  <!-- ───── Tuning drawer ───── -->
  <div class="card tuning" id="tuning">
    <h2>Tuning · Live</h2>
    <div class="slider-row">
      <label for="s-thr">Bad threshold</label>
      <input type="range" id="s-thr" min="10" max="90" step="1" value="40" oninput="onSlider()">
      <span class="slider-val" id="s-thr-v">40</span>
    </div>
    <div class="slider-row">
      <label for="s-mar">Hysteresis ±</label>
      <input type="range" id="s-mar" min="0" max="20" step="1" value="5" oninput="onSlider()">
      <span class="slider-val" id="s-mar-v">5</span>
    </div>
    <div class="slider-row">
      <label for="s-a">Smoothing α</label>
      <input type="range" id="s-a" min="0.05" max="0.5" step="0.05" value="0.20" oninput="onSlider()">
      <span class="slider-val" id="s-a-v">0.20</span>
    </div>
  </div>

  <!-- ───── Alert Screenshots ───── -->
  <div class="card" style="margin-top: 20px;">
    <h2>Alert Screenshots</h2>
    <div class="gallery" id="gallery">
      <span style="color: var(--mf); font-size: 12px;">No alerts yet.</span>
    </div>
  </div>

</div>

<audio id="alert-sound" preload="auto">
  <source src="data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" type="audio/wav">
</audio>

<script>
  // ─── state ───
  let lastAlerts = 0;
  let selectedSession = null;
  let selectedSessionScores = null;

  function fmtClock(seconds) {
    if (!seconds || seconds < 0) return "00:00";
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return (h ? String(h).padStart(2,'0')+':' : '') + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
  }

  // ─── labels ───
  const LABEL_KEYS = { '1':'good', '2':'slouch', '3':'forward_head', '4':'tilt', '0':'unlabeled' };
  function highlightLabel(name) {
    document.querySelectorAll('#label-buttons button').forEach(b => {
      b.classList.toggle('btn-active', b.dataset.label === name);
    });
    document.getElementById('current-label').textContent = name;
  }
  function setLabel(name) {
    fetch('/label/' + name).then(r => r.json()).then(d => { if (d.label) highlightLabel(d.label); });
  }
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (LABEL_KEYS[e.key]) { setLabel(LABEL_KEYS[e.key]); e.preventDefault(); }
    if (e.key === ' ') { togglePause(); e.preventDefault(); }
  });

  // ─── pause ───
  function togglePause() {
    fetch('/pause', { method: 'POST' }).then(r => r.json()).then(d => {
      document.getElementById('pause-label').textContent = d.paused ? 'Resume' : 'Pause';
    });
  }

  // ─── tuning drawer ───
  function toggleTuning() { document.getElementById('tuning').classList.toggle('open'); }
  function onSlider() {
    const thr = parseInt(document.getElementById('s-thr').value);
    const mar = parseInt(document.getElementById('s-mar').value);
    const a   = parseFloat(document.getElementById('s-a').value);
    document.getElementById('s-thr-v').textContent = thr;
    document.getElementById('s-mar-v').textContent = mar;
    document.getElementById('s-a-v').textContent   = a.toFixed(2);
    fetch('/config', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({threshold: thr, margin: mar, alpha: a})
    });
  }
  function loadConfig() {
    fetch('/config').then(r => r.json()).then(c => {
      document.getElementById('s-thr').value = c.threshold;
      document.getElementById('s-mar').value = c.margin;
      document.getElementById('s-a').value   = c.alpha;
      document.getElementById('s-thr-v').textContent = c.threshold;
      document.getElementById('s-mar-v').textContent = c.margin;
      document.getElementById('s-a-v').textContent   = Number(c.alpha).toFixed(2);
    });
  }

  // ─── sparkline ───
  function sparkPath(arr, w, h, maxVal) {
    if (!arr || !arr.length) return "";
    const n = arr.length;
    let d = "";
    for (let i = 0; i < n; i++) {
      const x = (i / Math.max(1, n - 1)) * w;
      const y = h - (Math.min(100, arr[i]) / maxVal) * h;
      d += (i === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1) + " ";
    }
    return d;
  }

  // ─── metric contributions ───
  const METRIC_ORDER = [
    ["nose_y",        "nose Y"],
    ["nose_z",        "nose Z"],
    ["ear_z",         "ear Z (FHP)"],
    ["face_size",     "face size"],
    ["shoulder_tilt", "shoulder tilt"],
  ];
  function renderMetrics(sub) {
    const host = document.getElementById('metrics');
    host.innerHTML = METRIC_ORDER.map(([k, label]) => {
      const v = sub && sub[k] != null ? sub[k] : 0;
      const pct = Math.min(100, Math.max(0, v));
      return `<div class="metric">
                <span class="label">${label}</span>
                <div class="bar"><div class="fill" style="width:${pct}%"></div></div>
                <span class="val mono">${v.toFixed(1)}</span>
              </div>`;
    }).join('');
  }

  // ─── heatmap ───
  function colorFor(badPct) {
    // 0 = green, 50 = orange, 100 = red
    const hue = 142 - (badPct / 100) * 142;   // 142 (emerald) → 0 (red)
    const sat = 60 - (50 - Math.abs(50 - badPct));
    return `hsl(${hue.toFixed(0)} 60% 35%)`;
  }
  function renderHeatmap(buckets) {
    const host = document.getElementById('heatmap');
    // Always render 30 cells, right-aligned (most recent on the right)
    const cells = [];
    const pad = 30 - buckets.length;
    for (let i = 0; i < pad; i++) cells.push('<div class="heatmap-cell empty"></div>');
    for (const b of buckets) {
      cells.push(`<div class="heatmap-cell" style="background:${colorFor(b.bad_pct)}" title="avg ${b.avg} · bad ${b.bad_pct}%"></div>`);
    }
    host.innerHTML = cells.join('');
  }

  // ─── alert sound ───
  function playBeep() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator(); const g = ctx.createGain();
      o.type = 'square'; o.frequency.value = 880;
      g.gain.value = 0.08;
      o.connect(g); g.connect(ctx.destination);
      o.start(); setTimeout(() => { o.stop(); ctx.close(); }, 180);
    } catch (e) {}
  }

  // ─── main poll ───
  function setState(s) {
    const badge = document.getElementById('state-badge');
    badge.classList.remove('badge-outline', 'badge-success', 'badge-destructive', 'badge-warning');
    if (s.paused) { badge.classList.add('badge-warning'); badge.innerHTML = '<span class="dot"></span>PAUSED'; }
    else if (s.buzzer_on && s.is_bad) { badge.classList.add('badge-destructive'); badge.innerHTML = '<span class="dot"></span>BUZZER ON'; }
    else if (s.is_bad) { badge.classList.add('badge-destructive'); badge.innerHTML = '<span class="dot"></span>BAD POSTURE'; }
    else if (s.buzzer_on) { badge.classList.add('badge-warning'); badge.innerHTML = '<span class="dot"></span>RECOVERING'; }
    else { badge.classList.add('badge-success'); badge.innerHTML = '<span class="dot"></span>GOOD POSTURE'; }
  }

  async function poll() {
    try {
      const r = await fetch('/stats');
      const s = await r.json();

      document.getElementById('score').textContent = s.score.toFixed(0);
      document.getElementById('fps').textContent   = s.fps.toFixed(1);
      document.getElementById('alerts').textContent= s.alerts;
      document.getElementById('rows').textContent  = s.rows.toLocaleString();
      document.getElementById('mode').textContent  = s.mode;
      document.getElementById('buzzer').textContent= s.buzzer_on ? 'on' : 'off';
      document.getElementById('pause-label').textContent = s.paused ? 'Resume' : 'Pause';

      // mode badge
      const mbd = document.getElementById('mode-badge');
      mbd.innerHTML = '<span class="dot"></span>' + s.mode.toUpperCase();

      // session clock + summary card
      if (s.session_start) {
        document.getElementById('clock').textContent = 'session ' + fmtClock(s.duration);
        document.getElementById('summary-dur').textContent = fmtClock(s.duration);
      }
      document.getElementById('summary-avg').textContent = s.summary_avg.toFixed(1);
      document.getElementById('summary-bad').textContent = s.summary_bad_pct.toFixed(1) + '%';

      // calibration overlay — shown while self-calibrate runs
      const overlay = document.getElementById('cal-overlay');
      if (s.calibrating) {
        overlay.classList.add('show');
        const CIRC = 282.7;                      // 2π·45
        const off = CIRC * (1 - (s.calibration_progress || 0));
        document.getElementById('cal-ring').setAttribute('stroke-dashoffset', off.toFixed(1));
        document.getElementById('cal-pct').textContent = Math.round((s.calibration_progress || 0) * 100) + '%';
      } else {
        overlay.classList.remove('show');
      }

      // state badge
      setState(s);

      // sparkline
      const spark = s.sparkline || [];
      document.getElementById('spark-path').setAttribute('d', sparkPath(spark, 600, 50, 100));
      // threshold line
      const thrY = 50 - (s.config.threshold / 100) * 50;
      document.getElementById('spark-thr-line').setAttribute('y1', thrY.toFixed(1));
      document.getElementById('spark-thr-line').setAttribute('y2', thrY.toFixed(1));
      // overlay sparkline from selected past session
      if (selectedSessionScores) {
        document.getElementById('spark-overlay').setAttribute('d', sparkPath(selectedSessionScores.slice(-600), 600, 50, 100));
      } else {
        document.getElementById('spark-overlay').setAttribute('d', '');
      }

      // metrics
      renderMetrics(s.sub_scores);

      // heatmap
      renderHeatmap(s.heatmap || []);

      // current label
      highlightLabel(s.label || 'unlabeled');

      // alert sound (play on alert count increment)
      if (s.alerts > lastAlerts) { playBeep(); lastAlerts = s.alerts; }

    } catch (e) { /* retry next tick */ }
  }

  // ─── past sessions list ───
  async function loadSessions() {
    try {
      const r = await fetch('/sessions');
      const arr = await r.json();
      const host = document.getElementById('sessions');
      if (!arr.length) { host.innerHTML = '<span style="color:var(--mf);font-size:12px;">No past sessions.</span>'; return; }
      host.innerHTML = arr.slice(0, 20).map(s => `
        <div class="session-item">
          <span class="name mono">${s.name}</span>
          <div>
            <span style="color:var(--mf);font-size:11px;margin-right:8px;">${s.rows} rows</span>
            <button data-name="${s.name}" onclick="toggleSession('${s.name}')">overlay</button>
          </div>
        </div>`).join('');
    } catch (e) {}
  }
  async function toggleSession(name) {
    const btns = document.querySelectorAll('#sessions button');
    if (selectedSession === name) {
      selectedSession = null; selectedSessionScores = null;
      btns.forEach(b => b.classList.remove('active'));
    } else {
      selectedSession = name;
      btns.forEach(b => b.classList.toggle('active', b.dataset.name === name));
      try {
        const r = await fetch('/sessions/' + encodeURIComponent(name) + '/scores');
        selectedSessionScores = await r.json();
      } catch (e) { selectedSessionScores = null; }
    }
  }

  // ─── gallery ───
  async function loadGallery() {
    try {
      const r = await fetch('/screenshots');
      const arr = await r.json();
      const host = document.getElementById('gallery');
      if (!arr.length) { host.innerHTML = '<span style="color:var(--mf);font-size:12px;">No alerts yet.</span>'; return; }
      host.innerHTML = arr.slice(-30).reverse().map(name =>
        `<img src="/screenshots/${encodeURIComponent(name)}" alt="${name}" title="${name}">`
      ).join('');
    } catch (e) {}
  }

  loadConfig();
  loadSessions();
  loadGallery();
  poll();
  setInterval(poll, 500);
  setInterval(loadSessions, 15000);
  setInterval(loadGallery, 10000);
</script>
</body>
</html>
"""


# ─── routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_stream():
    """Yield MJPEG frames for the browser."""
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.05)
                continue
            _, jpeg = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)


@app.route("/stats")
def stats():
    """Single polling endpoint — everything the dashboard needs in one JSON."""
    ss = live_state.get("session_start")
    duration = time.time() - ss if ss else 0
    return jsonify({
        "score":        live_state["score"],
        "score_raw":    live_state["score_raw"],
        "is_bad":       live_state["is_bad"],
        "buzzer_on":    live_state["buzzer_on"],
        "fps":          live_state["fps"],
        "alerts":       live_state["alerts"],
        "rows":         live_state["rows"],
        "mode":         live_state["mode"],
        "label":        live_state["label"],
        "paused":       live_state["paused"],
        "calibrating":  live_state["calibrating"],
        "calibration_progress": live_state["calibration_progress"],
        "summary_avg":      live_state.get("summary_avg", 0.0),
        "summary_bad_pct":  live_state.get("summary_bad_pct", 0.0),
        "session_start": ss,
        "duration":     duration,
        "sub_scores":   live_state["sub_scores"],
        "sparkline":    list(sparkline_buffer),
        "heatmap":      list(heatmap_buckets),
        "config":       CONFIG,
    })


@app.route("/rotate")
def rotate():
    global VIDEO_ROTATION
    VIDEO_ROTATION = (VIDEO_ROTATION + 90) % 360
    return {"rotation": VIDEO_ROTATION}


@app.route("/label")
def get_label():
    return {"label": current_label}


@app.route("/label/<name>")
def set_label(name):
    global current_label
    if name not in VALID_LABELS:
        return jsonify({"error": f"invalid label; allowed: {sorted(VALID_LABELS)}"}), 400
    current_label = name
    monitor_stats["label"] = name
    live_state["label"] = name
    return {"label": name}


@app.route("/config", methods=["GET"])
def config_get():
    return jsonify(CONFIG)


@app.route("/config", methods=["POST"])
def config_set():
    body = request.get_json(silent=True) or {}
    if "threshold" in body:
        CONFIG["threshold"] = max(0, min(100, float(body["threshold"])))
    if "margin" in body:
        CONFIG["margin"] = max(0, min(50, float(body["margin"])))
    if "alpha" in body:
        CONFIG["alpha"] = max(0.01, min(1.0, float(body["alpha"])))
    return jsonify(CONFIG)


@app.route("/pause", methods=["POST"])
def pause_toggle():
    global paused
    paused = not paused
    live_state["paused"] = paused
    return jsonify({"paused": paused})


@app.route("/download/csv")
def download_csv():
    path = live_state.get("csv_path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "no active session CSV"}), 404
    return send_file(path, as_attachment=True,
                     download_name=os.path.basename(path), mimetype="text/csv")


@app.route("/screenshots")
def screenshots_list():
    if not os.path.isdir(SCREENSHOT_DIR):
        return jsonify([])
    files = [f for f in os.listdir(SCREENSHOT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return jsonify(files)


@app.route("/screenshots/<path:name>")
def screenshots_serve(name):
    # Basic safety — no traversal
    if ".." in name or name.startswith("/"):
        return "", 400
    return send_file(os.path.join(SCREENSHOT_DIR, name))


@app.route("/sessions")
def sessions_list():
    if not os.path.isdir(DATA_DIR):
        return jsonify([])
    out = []
    for name in sorted(os.listdir(DATA_DIR), reverse=True):
        if not name.endswith(".csv"):
            continue
        full = os.path.join(DATA_DIR, name)
        try:
            with open(full) as f:
                rows = sum(1 for _ in f) - 1
        except Exception:
            rows = 0
        out.append({"name": name, "rows": max(0, rows)})
    return jsonify(out)


@app.route("/sessions/<path:name>/scores")
def session_scores(name):
    if ".." in name or name.startswith("/") or not name.endswith(".csv"):
        return jsonify([]), 400
    full = os.path.join(DATA_DIR, name)
    if not os.path.exists(full):
        return jsonify([]), 404
    # Prefer score_smoothed, fall back to score
    scores = []
    try:
        with open(full) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("score_smoothed") or row.get("score")
                if v is not None:
                    try:
                        scores.append(float(v))
                    except ValueError:
                        pass
    except Exception:
        pass
    return jsonify(scores)


# ═══════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════

def prompt_mode():
    """Blocking terminal prompt for mode selection. Accepts --mode CLI flag
    as an override so headless launches don't get stuck."""
    args = sys.argv[1:]
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args) and args[idx + 1] in ("calibrate", "research"):
            return args[idx + 1]

    print()
    print("Select scoring mode:")
    print("  1) Self-calibrate  — 5s 'sit up straight' baseline (per-user)")
    print("  2) Research        — ergonomics literature thresholds (no calibration)")
    while True:
        choice = input("Choice [1/2]: ").strip()
        if choice == "1":
            return "calibrate"
        if choice == "2":
            return "research"
        print("  Invalid. Enter 1 or 2.")


if __name__ == "__main__":
    args = sys.argv[1:]
    video_path = None

    if "--video" in args:
        idx = args.index("--video")
        if idx + 1 < len(args):
            video_path = args[idx + 1]
            if not os.path.exists(video_path):
                print(f"[ERROR] Not found: {video_path}")
                sys.exit(1)

    # Optional --port override (useful on macOS where port 5000 is held by
    # the AirPlay Receiver). Defaults to WEB_PORT defined at the top of the file.
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            try:
                WEB_PORT = int(args[idx + 1])
            except ValueError:
                print(f"[ERROR] Invalid --port value: {args[idx + 1]}")
                sys.exit(1)

    print()
    print("=" * 55)
    print("   ERGONOMIC POSTURE MONITOR")
    print(f"   Web view: http://<pi-ip>:{WEB_PORT}")
    print("=" * 55)

    mode = prompt_mode()

    monitor_stats["session_start"] = time.time()
    monitor_stats["mode"] = mode

    # Start monitor in background thread
    monitor_thread = threading.Thread(
        target=monitor_loop, args=(video_path, mode), daemon=True
    )
    monitor_thread.start()

    # Start Flask web server (main thread)
    print(f"[WEB] Starting server on port {WEB_PORT}...")
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, debug=False)
