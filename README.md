# IoT Ergonomic Posture Monitor

A real-time posture monitoring system using Raspberry Pi 5 and MediaPipe Pose that detects slouching and alerts you with a buzzer before bad habits set in. All processing runs locally — no cloud, no GPU.

## How It Works

Camera → MediaPipe Pose (33 landmarks) → Extract 7 key points → Compute 5 posture metrics → Compare to personal baseline → Score 0-100 → State machine → Buzzer alert if slouching >10 seconds

### The 5 Metrics

| Metric | What It Measures | Weight |
|--------|-----------------|--------|
| nose_y | Body sinking in frame | 30% |
| nose_z_depth | Forward lean (nose Z vs shoulder Z) | 25% |
| ear_z_depth | Forward head posture (ear Z vs shoulder Z) | 20% |
| face_size | Distance to camera (nose-to-ear spread) | 15% |
| shoulder_tilt | Sideways leaning (left vs right shoulder Y) | 10% |

Each metric is measured as deviation from a **personal baseline** — you sit up straight for 5 seconds at startup and the system calibrates to your body.

### Why These Metrics?

First attempt used X/Y coordinates (head forward offset, head drop). Scores were stuck at 0-2 out of 100 regardless of posture. A diagnostic script revealed the problem: with a front-facing camera, slouching is **Z-axis movement** (toward the camera), not lateral movement. The head barely shifts left/right — it moves *closer*. Face size doubled between good and bad posture, while X-based metrics changed by 0.003.

## Hardware

| Component | Details |
|-----------|---------|
| Computer | Raspberry Pi 5 (4GB RAM, 100GB SSD) |
| Camera | Pi Camera Module 2 (15-pin, needs 22-pin adapter) |
| Alert | Active buzzer on GPIO 17 |
| Remote Access | ZeroTier VPN |

## Software

- **Pose Estimation**: MediaPipe Pose (BlazePose) — 8-12 FPS on Pi 5 CPU
- **Video Processing**: OpenCV
- **Web Stream**: Flask (MJPEG) — live skeleton overlay + score bar at `http://<pi-ip>:5000`
- **Alerts**: gpiozero (buzzer)
- **Logging**: SQLite + JSON

## Project Structure

```
ergonomic/
├── posture_web.py       # Main app — scoring, state machine, Flask stream, buzzer
├── diagnose.py          # Diagnostic tool — samples video and prints all metrics
├── setup.sh             # Installs dependencies
├── requirements.txt     # pip dependencies
├── screenshots/         # Auto-saved on alerts
└── posture_log.json     # Event history
```

## Setup

```bash
git clone https://github.com/AaryansNepal/CSC-494_Project.git
cd CSC-494_Project/ergonomic
bash setup.sh
python3 posture_web.py
```

Open `http://<pi-ip>:5000` in a browser to see the live stream.

## Current Status (Week 6)

**Working**: MediaPipe at 8-12 FPS, 5 validated metrics, personal calibration, state machine with buzzer alerts, Flask live stream with skeleton overlay

**Temporary**: Using phone-recorded video as input until Pi Camera adapter arrives

## Roadmap

- **Weeks 7-10**: Live camera feed, labeled data collection, scikit-learn classifier to replace hand-tuned thresholds, dashboard with trend graphs
- **Weeks 11-12**: Benchmarking, stretch goals (mouth/eye/elbow landmarks), final presentation with live demo

## Author

**Aaryans Nepal** — CSC 494 IoT (Spring 2026), Northern Kentucky University
GitHub: [@AaryansNepal](https://github.com/AaryansNepal)
Learning with AI: [CSC-494_Learning-with-AI](https://github.com/AaryansNepal/CSC-494_Learning-with-AI)
