# IoT Theft Detection System

A privacy-first IoT theft detection system using Raspberry Pi 5 and pose estimation to identify suspicious concealment behaviors in retail environments.

## Project Overview

### Problem Statement

Retail theft costs US businesses over $100 billion annually. Traditional solutions have significant drawbacks:
1. **CCTV is reactive**: Cameras record theft but rarely prevent it — footage is reviewed after the fact
2. **Staff monitoring is inconsistent**: Employees cannot watch every customer at all times
3. **Commercial AI systems are expensive**: Enterprise loss prevention solutions cost thousands per month and require specialized hardware

Small and mid-size retailers need an affordable, real-time detection system that can flag suspicious behavior as it happens.

### Solution

A single-camera system running on a Raspberry Pi 5 that:
- Detects people in frame using **Mediapipe Pose** estimation (~8-12 fps)
- Tracks **hand-to-hip/pocket distance** in real time
- Flags **suspicious concealment events** when hands move toward pockets/waistband and stay there
- Logs events locally with **timestamp, confidence score, and screenshot** — privacy-first, no facial recognition
- Sends alerts via **Telegram bot** or a **Flask web dashboard**

## Key Features

### Phase 1: Core Detection (Weeks 1-6)
- Mediapipe Pose person detection on Raspberry Pi 5
- Real-time hand-to-hip/pocket distance tracking
- Concealment event detection with configurable thresholds
- Local event logging (timestamp, confidence, screenshot)
- Basic Flask web dashboard to view events

### Phase 2: Alerts & Monitoring (Weeks 7-10)
- Telegram bot for real-time alert notifications
- Enhanced Flask dashboard with event history and filtering
- Confidence score tuning and false-positive reduction
- Event statistics and daily summaries
- System health monitoring (FPS, CPU temp, uptime)

### Phase 3: Refinement (Weeks 11-14)
- Threshold optimization based on collected data
- Cool-down logic to prevent duplicate alerts
- Multi-zone detection (define regions of interest)
- Documentation and final presentation
- Performance benchmarking and optimization

## Technologies & Tools

### Hardware Components
| Component | Model | Purpose |
|-----------|-------|---------|
| Single Board Computer | Raspberry Pi 5 (8GB) | Main processing unit |
| Camera | Raspberry Pi Camera Module 3 | Video capture |
| Storage | 32GB+ microSD | OS + event storage |
| Power | USB-C power supply | Pi power |

### Software Stack
- **Language**: Python 3
- **Pose Estimation**: Mediapipe Pose
- **Computer Vision**: OpenCV
- **Web Dashboard**: Flask
- **Alerts**: Telegram Bot API (python-telegram-bot)
- **Data Storage**: SQLite (local event database)
- **OS**: Raspberry Pi OS (64-bit)

### Development Tools
- VS Code with Remote SSH
- Git/GitHub for version control
- Claude/AI for learning assistance

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Raspberry Pi 5 (8GB)                │
│                                                  │
│  ┌──────────────┐    ┌─────────────────────┐    │
│  │  Pi Camera   │───▶│  OpenCV Capture      │    │
│  │  Module 3    │    │  (Video Stream)       │    │
│  └──────────────┘    └──────────┬────────────┘    │
│                                 │                 │
│                      ┌──────────▼────────────┐    │
│                      │  Mediapipe Pose        │    │
│                      │  (Person Detection)    │    │
│                      └──────────┬────────────┘    │
│                                 │                 │
│                      ┌──────────▼────────────┐    │
│                      │  Concealment Detector  │    │
│                      │  (Hand-Hip Distance)   │    │
│                      └──────────┬────────────┘    │
│                                 │                 │
│              ┌──────────────────┼────────────┐    │
│              │                  │            │    │
│    ┌─────────▼──────┐  ┌──────▼─────┐ ┌───▼───┐│
│    │  Event Logger   │  │  Telegram  │ │ Flask ││
│    │  (SQLite +      │  │  Bot Alert │ │ Web   ││
│    │   Screenshots)  │  └────────────┘ │ UI    ││
│    └─────────────────┘                 └───────┘│
└─────────────────────────────────────────────────┘
```

## Success Metrics

### Technical Metrics
- Mediapipe Pose running at 8+ fps on Pi 5 (no accelerator)
- Concealment event detection with configurable confidence threshold
- <5 second latency from detection to Telegram alert
- System runs continuously for 8+ hours without crashes

### Detection Metrics
- Successfully detects simulated concealment gestures in testing
- False positive rate manageable with threshold tuning
- Events logged with accurate timestamps and screenshots

### Project Completion Metrics
- Functional end-to-end pipeline (camera → detection → alert)
- Flask dashboard displaying event history
- Telegram bot delivering real-time alerts
- Successful final presentation with live demo

## Getting Started

### Prerequisites
```bash
# Raspberry Pi 5 with Pi OS 64-bit

# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv libcap-dev

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install opencv-python mediapipe flask python-telegram-bot
```

### Hardware Setup
1. Connect Pi Camera Module 3 to the CSI port on the Raspberry Pi 5
2. Enable camera interface: `sudo raspi-config` → Interface Options → Camera
3. Mount camera with a clear view of the monitoring area

### Software Setup
```bash
# Clone repository
git clone https://github.com/AaryansNepal/CSC-494_Project.git
cd CSC-494_Project

# Install dependencies
pip install -r requirements.txt

# Configure settings
cp config.example.py config.py
# Edit config.py with your Telegram bot token and thresholds

# Run the system
python main.py
```

## Project Structure

```
CSC-494_Project/
├── main.py                  # Main application entry point
├── config.py                # Configuration (thresholds, tokens)
├── requirements.txt         # Python dependencies
├── detector/
│   ├── pose_estimator.py    # Mediapipe Pose wrapper
│   ├── concealment.py       # Hand-hip distance logic
│   └── event_logger.py      # SQLite logging + screenshots
├── alerts/
│   ├── telegram_bot.py      # Telegram alert integration
│   └── notifier.py          # Alert dispatcher
├── web/
│   ├── app.py               # Flask web dashboard
│   ├── templates/           # HTML templates
│   └── static/              # CSS/JS assets
├── events/
│   ├── screenshots/         # Captured event images
│   └── events.db            # SQLite event database
├── tests/                   # Unit tests
├── docs/                    # Documentation
│   └── hardware-setup.md
├── Individual Project.md    # Project overview
├── Individual Progress.md   # Milestone tracker
└── README.md
```

## Milestones & Timeline

### Week 1-2: Project Setup & Environment
- [x] Complete HW2 requirements
- [x] Create GitHub repositories
- [x] Write README files and project plan
- [ ] Set up Raspberry Pi 5 with OS
- [ ] Install Python, OpenCV, Mediapipe

### Week 3-4: Core Pose Detection
- [ ] Get Pi Camera streaming with OpenCV
- [ ] Run Mediapipe Pose on Pi 5, measure FPS
- [ ] Extract hand and hip landmark coordinates
- [ ] Calculate hand-to-hip distance metrics
- [ ] **PPP Presentation**

### Week 5-6: Concealment Detection
- [ ] Implement concealment threshold logic
- [ ] Add event logging (SQLite + screenshot capture)
- [ ] Test with simulated concealment gestures
- [ ] Tune detection sensitivity

### Week 7-8: Alerts & Dashboard
- [ ] Build Flask web dashboard for event viewing
- [ ] Integrate Telegram bot for real-time alerts
- [ ] Add event history and filtering
- [ ] System health monitoring

### Week 9-10: Optimization
- [ ] Reduce false positives with cool-down logic
- [ ] Optimize FPS and CPU usage
- [ ] Add confidence score refinement
- [ ] Stress test continuous operation

### Week 11-12: Polish & Documentation
- [ ] Final threshold tuning from real data
- [ ] Complete documentation
- [ ] Performance benchmarking
- [ ] Prepare final presentation
- [ ] **Final Presentation**

## Privacy Notice

This system is designed with privacy as a priority:
- **No facial recognition** — only body pose landmarks are used
- **No video recording** — only event screenshots are saved
- **Local processing only** — no data leaves the device except Telegram alerts
- **No personal identification** — events are logged as anonymous detections

## Author

**Aaryans Nepal**
- GitHub: [@AaryansNepal](https://github.com/AaryansNepal)
- Course: CSC 494 - IoT (Spring 2026)
- Institution: Northern Kentucky University

## Acknowledgments

- Course instructor and teaching assistants
- Mediapipe team at Google for the pose estimation framework
- AI assistance from Claude (Anthropic) for learning and development

---

**Note**: This project is for educational purposes. It is a proof-of-concept and is not intended for deployment as a commercial security system.
