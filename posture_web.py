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
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify


# ═══════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════

BUZZER_PIN = 17
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CALIBRATION_SECONDS = 5
BAD_POSTURE_TRIGGER = 10       # seconds bad before buzzer
GOOD_POSTURE_RECOVERY = 3      # seconds good before buzzer off
POSTURE_SCORE_THRESHOLD = 40   # 0-100, above = bad
MEDIAPIPE_MODEL = 1            # 0=lite, 1=full, 2=heavy
VIDEO_ROTATION = 180             # Rotate video: 0=none, 90=CW, 180=flip, 270=CCW
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
            time.sleep(2)
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
        # picamera2's "RGB888" format actually delivers bytes in BGR order
        # (libcamera naming quirk) — so capture_array() returns a BGR buffer.
        frame_bgr = cam.capture_array()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
    """
    scores = []

    # Nose Y: baseline - current (positive = head dropped in frame)
    nose_y_drop = baseline["nose_y"] - current["nose_y"]
    scores.append(min(max(0, nose_y_drop) * 800, 100))

    # Nose Z depth: current more negative than baseline = leaning forward
    z_forward = abs(current["nose_z_depth"]) - abs(baseline["nose_z_depth"])
    scores.append(min(max(0, z_forward) * 1000, 100))

    # Ear Z vs shoulder Z: ears moving further forward than baseline
    ear_forward = abs(current["ear_z_depth"]) - abs(baseline["ear_z_depth"])
    scores.append(min(max(0, ear_forward) * 1500, 100))

    # Face size: growing = leaning toward camera
    size_growth = current["face_size"] - baseline["face_size"]
    scores.append(min(max(0, size_growth) * 3000, 100))

    # Shoulder tilt: deviation from baseline
    tilt_dev = current["shoulder_tilt"] - baseline["shoulder_tilt"]
    scores.append(min(max(0, tilt_dev) * 400, 100))

    # Weights: Z-depth and position are most reliable
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    return min(sum(s * w for s, w in zip(scores, weights)), 100)


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
    """
    scores = []

    # Shoulder tilt: linear ramp from 0.02 (borderline) to 0.06 (severe)
    tilt_excess = max(0.0, current["shoulder_tilt"] - 0.02)
    scores.append(min(tilt_excess * 2500, 100))

    # Forward head via ears (ear_z - shoulder_z < 0 means ears forward)
    ear_fhp = max(0.0, -current["ear_z_depth"])
    scores.append(min(ear_fhp * 1000, 100))

    # Forward head via nose (more severe when the whole head thrusts forward)
    nose_fhp = max(0.0, -current["nose_z_depth"])
    scores.append(min(nose_fhp * 700, 100))

    # FHP dominates because it's the most common desk-posture issue
    weights = [0.30, 0.40, 0.30]
    return min(sum(s * w for s, w in zip(scores, weights)), 100)


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

def draw_overlay(frame_bgr, landmarks, score, status):
    """Draw skeleton, score bar, and status label on frame."""
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
    is_bad = score > POSTURE_SCORE_THRESHOLD

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

        print(f"   Calibrating... {i+1}/{cal_frames} ({len(samples)} valid)", end="\r")

    print()

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

            remaining = CALIBRATION_SECONDS - (time.time() - start)
            overlay = draw_overlay(frame_bgr, results.pose_landmarks, 0,
                                   f"CALIBRATING... {remaining:.1f}s left")
            cv2.putText(overlay, "HOLD STILL - GOOD POSTURE", (80, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            with frame_lock:
                current_frame = overlay.copy()

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
    "timestamp", "mode", "label", "score", "is_bad",
    "nose_y", "nose_z_depth", "ear_z_depth", "face_size", "shoulder_tilt",
]


class SessionLogger:
    """Append one CSV row per processed frame. Line-buffered so crashes don't
    lose the tail of the session."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="", buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.rows = 0

    def log(self, mode, label, score, is_bad, metrics):
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "mode": mode,
            "label": label,
            "score": round(score, 2),
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

    print(f"\n[MONITORING] Live! Open http://<pi-ip>:{WEB_PORT} in browser")
    print(f"  Mode: {mode} | Threshold: {POSTURE_SCORE_THRESHOLD}/100 | Trigger: {BAD_POSTURE_TRIGGER}s")
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
    frame_count = 0
    fps_timer = time.time()
    last_screenshot = 0
    session_scores = []
    alert_count = 0
    session_start = time.time()

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
                    score = compute_posture_score_research(metrics)
                else:
                    score = compute_posture_score(metrics, baseline)
                session_scores.append(score)
                is_bad = score > POSTURE_SCORE_THRESHOLD

                # Per-frame data collection row (captures whatever label the
                # user has selected in the web UI).
                session_log.log(mode, current_label, score, is_bad, metrics)

                # --- State machine ---
                if is_bad:
                    good_start = None
                    if bad_start is None:
                        bad_start = now
                    bad_duration = now - bad_start

                    if bad_duration >= BAD_POSTURE_TRIGGER and not buzzer_active:
                        buzzer_active = True
                        alert_count += 1
                        buzzer_on()
                        log_event("alert_start", score, metrics)
                        if now - last_screenshot > 30:
                            save_screenshot(frame_bgr, score, "alert")
                            last_screenshot = now

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
                            log_event("recovered", score, metrics)
                            save_screenshot(frame_bgr, score, "recovered")
                        status = f"Recovering: {good_duration:.0f}s / {GOOD_POSTURE_RECOVERY}s"
                    else:
                        good_start = None
                        status = "Good posture"

                # FPS calc
                elapsed = now - fps_timer
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Build status line
                full_status = f"FPS: {fps:.1f} | {status}"

                # Draw overlay on frame
                display_frame = draw_overlay(frame_bgr.copy(), results.pose_landmarks,
                                             score, full_status)

                # Update shared frame for web stream
                with frame_lock:
                    current_frame = display_frame

                # Update stats
                monitor_stats["score"] = score
                monitor_stats["status"] = status
                monitor_stats["fps"] = fps
                monitor_stats["alerts"] = alert_count
                monitor_stats["label"] = current_label

                # Terminal output
                if frame_count % 10 == 0:
                    bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
                    print(f"  [{bar}] {score:5.1f}/100 | {status:30s} | FPS:{fps:.1f}", end="\r")

            else:
                if buzzer_active:
                    buzzer_off()
                    buzzer_active = False
                bad_start = None
                good_start = None

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
            bad_pct = sum(1 for s in session_scores if s > POSTURE_SCORE_THRESHOLD) / len(session_scores) * 100
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

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Posture Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e;
            color: #eee;
            font-family: 'Segoe UI', Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
            color: #00d4ff;
        }
        .subtitle {
            color: #888;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
        .controls {
            margin-bottom: 15px;
        }
        .controls button {
            background: #16213e;
            color: #00d4ff;
            border: 2px solid #00d4ff;
            padding: 8px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.95em;
            margin: 0 5px;
        }
        .controls button:hover {
            background: #00d4ff;
            color: #1a1a2e;
        }
        .stream-container {
            border: 3px solid #333;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
        }
        .stream-container img {
            display: block;
            max-width: 90vw;
            max-height: 70vh;
        }
        .info {
            margin-top: 15px;
            color: #888;
            font-size: 0.85em;
        }
        .rotation-label {
            color: #00d4ff;
            font-size: 0.85em;
            margin-top: 5px;
        }
        .labels {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 14px;
        }
        .labels button {
            background: #16213e;
            color: #eee;
            border: 2px solid #555;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .labels button:hover { border-color: #00d4ff; }
        .labels button.active { background: #00d4ff; color: #1a1a2e; border-color: #00d4ff; }
        .label-status { color: #00d4ff; font-size: 0.9em; margin-top: 8px; }
    </style>
</head>
<body>
    <h1>Ergonomic Posture Monitor</h1>
    <p class="subtitle">Pi 5 + MediaPipe Pose | Real-time Analysis</p>
    <div class="controls">
        <button onclick="rotate()">↻ Rotate Video</button>
    </div>
    <p class="rotation-label" id="rot-label"></p>
    <div class="stream-container">
        <img src="/video_feed" alt="Live Stream">
    </div>
    <div class="labels" id="label-buttons">
        <button data-label="good" onclick="setLabel('good')">Good</button>
        <button data-label="slouch" onclick="setLabel('slouch')">Slouch</button>
        <button data-label="forward_head" onclick="setLabel('forward_head')">Forward Head</button>
        <button data-label="tilt" onclick="setLabel('tilt')">Tilt</button>
        <button data-label="unlabeled" onclick="setLabel('unlabeled')">Clear</button>
    </div>
    <p class="label-status" id="label-status">Labeling as: unlabeled</p>
    <p class="info">Score bar: <span style="color:#0c0">■</span> Good (0-30)
    <span style="color:#fb0">■</span> Warning (30-60)
    <span style="color:#e00">■</span> Bad (60-100) |
    Buzzer triggers after """ + str(BAD_POSTURE_TRIGGER) + """s of bad posture</p>
    <script>
        function rotate() {
            fetch('/rotate').then(r => r.json()).then(d => {
                document.getElementById('rot-label').textContent = 'Rotation: ' + d.rotation + '°';
            });
        }
        function highlightLabel(name) {
            document.querySelectorAll('#label-buttons button').forEach(b => {
                b.classList.toggle('active', b.dataset.label === name);
            });
            document.getElementById('label-status').textContent = 'Labeling as: ' + name;
        }
        function setLabel(name) {
            fetch('/label/' + name).then(r => r.json()).then(d => {
                if (d.label) highlightLabel(d.label);
            });
        }
        // Reflect current label on load (in case page was reopened mid-session)
        fetch('/label').then(r => r.json()).then(d => highlightLabel(d.label));
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/rotate")
def rotate():
    global VIDEO_ROTATION
    VIDEO_ROTATION = (VIDEO_ROTATION + 90) % 360
    print(f"  [ROTATE] Video rotation set to {VIDEO_ROTATION}°")
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
    print(f"  [LABEL] Current label → {name}")
    return {"label": name}


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
        time.sleep(0.05)  # ~20fps max for web stream


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
