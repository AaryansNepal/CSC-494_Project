# IoT Ergonomic Posture Monitor

A real-time posture monitoring system using Raspberry Pi 5 and MediaPipe Pose that detects slouching and alerts you with a buzzer before bad habits set in. All processing runs locally — no cloud, no GPU.

## How It Works

```
Camera → MediaPipe Pose (33 landmarks) → Extract 7 key points
       → Compute 5 posture metrics → Score 0-100
       → State machine (10 s bad → buzzer on, 3 s good → off)
       → Live browser stream with skeleton overlay + score bar
```

### The 5 Metrics

| Metric | What It Measures | Weight |
|--------|-----------------|--------|
| `nose_y` | Body sinking in frame | 30% |
| `nose_z_depth` | Forward lean (nose Z vs shoulder Z) | 25% |
| `ear_z_depth` | Forward head posture (ear Z vs shoulder Z) | 20% |
| `face_size` | Distance to camera (nose-to-ear spread) | 15% |
| `shoulder_tilt` | Sideways leaning (left vs right shoulder Y) | 10% |

### Why These Metrics?

First attempt used X/Y coordinates (head forward offset, head drop). Scores were stuck at 0-2 out of 100 regardless of posture. A diagnostic script (`diagnose.py`) revealed the problem: with a front-facing camera, slouching is **Z-axis movement** (toward the camera), not lateral movement. The head barely shifts left/right — it moves *closer*. Face size doubled between good and bad posture, while X-based metrics changed by 0.003.

## Hardware

| Component | Details |
|-----------|---------|
| Computer | Raspberry Pi 5 (4GB RAM, 100GB SSD) |
| Camera | Pi Camera Rev 1.3 (OV5647) — plugs directly into Pi 5 CSI, no adapter |
| Alert | Active buzzer on GPIO 17 (physical pin 11) → GND (pin 9) |
| Remote Access | ZeroTier VPN |

## Software Stack

- **Pose Estimation**: MediaPipe Pose (BlazePose) — 8-12 FPS on Pi 5 CPU
- **Video Processing**: OpenCV
- **Web Stream**: Flask (MJPEG) — live skeleton overlay + score bar at `http://<pi-ip>:5000`
- **Alerts**: gpiozero (buzzer)
- **Logging**: JSON event log (`posture_log.json`) + per-session CSV (`data/session_*.csv`)

## Project Structure

```
Posture/
├── posture_web.py         # Main app — scoring, state machine, Flask stream, buzzer, labeling
├── diagnose.py            # Diagnostic tool — samples video and prints all metrics
├── setup.sh               # Installs dependencies
├── requirements.txt       # pip dependencies
├── data/                  # Per-session labeled CSV logs (gitignored)
├── screenshots/           # Auto-saved on alerts (gitignored)
├── demo/                  # Demo video link
├── slides/                # Marp source + exported PDFs
│   ├── final_presentation.md / .pdf
│   ├── learning_with_ai_topic1.md / .pdf
│   └── learning_with_ai_topic2.md / .pdf
└── posture_log.json       # Event history (alerts, recoveries)
```

## Setup (one time on the Pi)

```bash
git clone https://github.com/AaryansNepal/CSC-494_Project.git ~/ergonomic
cd ~/ergonomic
bash setup.sh
```

## How to Use

### 1. Start the app

```bash
cd ~/ergonomic
source venv/bin/activate
python3 posture_web.py
```

### 2. Pick a scoring mode

At startup the terminal will prompt:

```
Select scoring mode:
  1) Self-calibrate  — 5s 'sit up straight' baseline (per-user)
  2) Research        — ergonomics literature thresholds (no calibration)
```

- **Self-calibrate** — sit upright in front of the camera. After 3 short beeps + 1 long beep, hold still for 5 seconds. The system captures your personal baseline. All scoring is relative to your upright pose.
- **Research mode** — skips calibration. Uses absolute thresholds from published ergonomics research (Raine & Twomey 1997 for shoulder asymmetry; Yip 2008 and Hansraj 2014 for forward head posture). Works immediately, no per-user setup.

To skip the prompt (useful for scripts / systemd):

```bash
python3 posture_web.py --mode research
python3 posture_web.py --mode calibrate
```

### 3. Open the web UI

Once you see `Running on http://<pi-ip>:5000`, open that URL in any browser on the same network. You'll see:

- **Live MJPEG stream** with MediaPipe skeleton overlay
- **Score bar** top-left (green 0-30 / orange 30-60 / red 60-100)
- **GOOD POSTURE / BAD POSTURE** label top-right
- **Rotate Video** button (only useful for pre-recorded video input — do *not* click mid-session in live mode, it invalidates your calibration)
- **Labeling buttons** below the stream: `Good / Slouch / Forward Head / Tilt / Clear`

### 4. Collect labeled data

Every frame is written to `data/session_<timestamp>_<mode>.csv` with columns:

```
timestamp, mode, label, score, is_bad, nose_y, nose_z_depth, ear_z_depth, face_size, shoulder_tilt
```

Click any label in the browser and it gets stamped onto all subsequent rows until you click another one. Each session produces a usable labeled dataset for later analysis or classifier training.

### 5. Experience the buzzer alert

- Hold bad posture continuously → after **10 seconds** the buzzer turns on.
- Sit back up → after **3 seconds** of continuous good posture the buzzer silences.
- Short fidgets or glances don't trigger — the state machine resets timers on any good frame.

### 6. Exit and review

`Ctrl+C` prints a session summary and the CSV path:

```
SESSION SUMMARY
  Mode:         calibrate
  Duration:     2.3 min
  Avg score:    18.4/100
  Bad posture:  12.1%
  Alerts:       2
  Rows logged:  4127 → data/session_20260420_110502_calibrate.csv
```

## Running with Pre-Recorded Video (no camera)

Useful for development on a machine without a Pi camera, or for consistent demos:

```bash
python3 posture_web.py --video demo.MOV
python3 posture_web.py --video demo.MOV --mode research
```

The video auto-loops so the stream never stops. Phone recordings are often sideways — set `VIDEO_ROTATION` at the top of `posture_web.py` (0 / 90 / 180 / 270 degrees).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[ERROR] Only 0 valid samples.` during calibration | You weren't fully in frame during the 5 s window. Make sure nose + both ears + both shoulders are visible, then retry. Or use `--mode research` to skip calibration. |
| `[ERROR] No live camera detected.` | Check ribbon cable seating, run `ls /dev/video*`, test with `libcamera-hello`, or fall back to `--video <file>`. |
| ZeroTier IP gives "access denied" in browser | ZeroTier IPs are only reachable by devices on the same ZeroTier network. Use the LAN IP (e.g. `192.168.1.x:5000`) for local access. |
| Video looks orange / warm in live mode | Pi Camera Rev 1.3 (OV5647) has slower auto-white-balance than Module 2. Give it ~5 seconds to settle before calibration. |
| Buzzer doesn't sound | Quick test: `python3 -c "from gpiozero import Buzzer; import time; b=Buzzer(17); b.on(); time.sleep(1); b.off()"`. Check polarity (long leg to pin 11, short to pin 9). Make sure no other process holds the GPIO (`sudo fuser -v /dev/gpiochip*`). |

## Project Status

Sprint 1 + Sprint 2 complete. Core detection, alerting, live camera, two scoring modes, and labeled data collection all working end-to-end.

See `slides/final_presentation.pdf` for the full project walkthrough, and `demo/README.md` for the live demo video.

## Author

**Aaryans Nepal** — CSC 494 IoT (Spring 2026), Northern Kentucky University
GitHub: [@AaryansNepal](https://github.com/AaryansNepal)
Learning with AI: [CSC-494_Learning-with-AI](https://github.com/AaryansNepal/CSC-494_Learning-with-AI)
