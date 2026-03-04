#!/usr/bin/env python3
"""
DIAGNOSTIC: What actually changes between good and bad posture?
Prints raw landmark values at different points in the video.
Run: python3 diagnose.py demo_02.MOV
"""
import cv2
import mediapipe as mp
import sys
import math

if len(sys.argv) < 2:
    print("Usage: python3 diagnose.py <video.MOV>")
    sys.exit(1)

VIDEO_ROTATION = 0  # change if needed: 0, 90, 180, 270

cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total / fps

print(f"Video: {total} frames, {fps:.0f}fps, {duration:.1f}s")
print(f"Rotation: {VIDEO_ROTATION}°")
print()

pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5)
PL = mp.solutions.pose.PoseLandmark

# Sample at 5 evenly spaced points in the video
sample_points = [0.1, 0.3, 0.5, 0.7, 0.9]  # fractions of video length

for frac in sample_points:
    frame_num = int(frac * total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    # Rotate
    if VIDEO_ROTATION == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif VIDEO_ROTATION == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif VIDEO_ROTATION == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    time_sec = frame_num / fps
    print(f"{'='*70}")
    print(f"  FRAME {frame_num} (t={time_sec:.1f}s) — "
          f"{'START/GOOD' if frac < 0.3 else 'MID/BAD?' if frac < 0.7 else 'END'}")
    print(f"{'='*70}")

    if not results.pose_landmarks:
        print("  NO PERSON DETECTED\n")
        continue

    lm = results.pose_landmarks.landmark

    # Print ALL upper body landmarks with x, y, z, visibility
    landmarks_of_interest = [
        ("NOSE", PL.NOSE),
        ("LEFT_EYE", PL.LEFT_EYE),
        ("RIGHT_EYE", PL.RIGHT_EYE),
        ("LEFT_EAR", PL.LEFT_EAR),
        ("RIGHT_EAR", PL.RIGHT_EAR),
        ("MOUTH_LEFT", PL.MOUTH_LEFT),
        ("MOUTH_RIGHT", PL.MOUTH_RIGHT),
        ("LEFT_SHOULDER", PL.LEFT_SHOULDER),
        ("RIGHT_SHOULDER", PL.RIGHT_SHOULDER),
        ("LEFT_ELBOW", PL.LEFT_ELBOW),
        ("RIGHT_ELBOW", PL.RIGHT_ELBOW),
        ("LEFT_HIP", PL.LEFT_HIP),
        ("RIGHT_HIP", PL.RIGHT_HIP),
    ]

    print(f"  {'Landmark':<20} {'X':>8} {'Y':>8} {'Z':>8} {'Vis':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for name, idx in landmarks_of_interest:
        p = lm[idx]
        vis_mark = "" if p.visibility > 0.5 else " LOW"
        print(f"  {name:<20} {p.x:8.4f} {p.y:8.4f} {p.z:8.4f} {p.visibility:5.2f}{vis_mark}")

    # Compute derived metrics
    nose = lm[PL.NOSE]
    l_ear = lm[PL.LEFT_EAR]
    r_ear = lm[PL.RIGHT_EAR]
    l_sh = lm[PL.LEFT_SHOULDER]
    r_sh = lm[PL.RIGHT_SHOULDER]
    l_eye = lm[PL.LEFT_EYE]
    r_eye = lm[PL.RIGHT_EYE]

    ear_mid_x = (l_ear.x + r_ear.x) / 2
    ear_mid_y = (l_ear.y + r_ear.y) / 2
    ear_mid_z = (l_ear.z + r_ear.z) / 2
    sh_mid_x = (l_sh.x + r_sh.x) / 2
    sh_mid_y = (l_sh.y + r_sh.y) / 2
    sh_mid_z = (l_sh.z + r_sh.z) / 2
    eye_mid_y = (l_eye.y + r_eye.y) / 2

    # Current metrics (what our code computes)
    head_forward = ear_mid_x - sh_mid_x
    head_drop = sh_mid_y - nose.y
    shoulder_tilt = abs(l_sh.y - r_sh.y)
    ear_shoulder_dist = abs(ear_mid_y - sh_mid_y)

    print()
    print(f"  CURRENT METRICS (what our code uses):")
    print(f"    head_forward (ear_x - shoulder_x):     {head_forward:+.4f}")
    print(f"    head_drop (shoulder_y - nose_y):        {head_drop:+.4f}")
    print(f"    shoulder_tilt (|L_sh_y - R_sh_y|):      {shoulder_tilt:.4f}")
    print(f"    ear_shoulder_dist (|ear_y - sh_y|):      {ear_shoulder_dist:.4f}")

    # ALTERNATIVE metrics that might work better
    print()
    print(f"  ALTERNATIVE METRICS (potential replacements):")

    # Z-depth: nose relative to shoulders (forward lean)
    nose_depth = nose.z - sh_mid_z
    print(f"    nose_z_depth (nose.z - shoulder.z):     {nose_depth:+.4f}")

    # Nose Y position in frame (absolute, drops when slouching)
    print(f"    nose_y_absolute:                        {nose.y:.4f}")

    # Eye-to-shoulder Y distance (shrinks when head drops)
    eye_sh_dist = sh_mid_y - eye_mid_y
    print(f"    eye_to_shoulder_y:                      {eye_sh_dist:.4f}")

    # Shoulder width (may increase when leaning forward)
    sh_width = abs(l_sh.x - r_sh.x)
    print(f"    shoulder_width:                         {sh_width:.4f}")

    # Nose-to-shoulder-midpoint angle from vertical
    dx = nose.x - sh_mid_x
    dy = nose.y - sh_mid_y  # note: y increases downward in image
    neck_angle = math.degrees(math.atan2(dx, -dy))
    print(f"    neck_angle_from_vertical:               {neck_angle:+.1f}°")

    # Ear Z vs Shoulder Z (forward head posture in depth)
    ear_forward_z = ear_mid_z - sh_mid_z
    print(f"    ear_z_vs_shoulder_z:                    {ear_forward_z:+.4f}")

    # Face size (nose to ear distance, increases when closer to camera)
    face_size = math.sqrt((nose.x - ear_mid_x)**2 + (nose.y - ear_mid_y)**2)
    print(f"    face_size (nose-ear dist):              {face_size:.4f}")

    print()

cap.release()
pose.close()

print("="*70)
print("WHAT TO LOOK FOR:")
print("  Compare the numbers between early frames (good posture)")
print("  and middle frames (bad posture). Whichever metrics show")
print("  the BIGGEST change are the ones we should use.")
print("="*70)
