# Demo

Live demonstration of the IoT Ergonomic Posture Monitor.

- **Google Drive video:** https://drive.google.com/drive/folders/1A5Lym7Eu_n0G-YA2YXZzJ7C9PfsxCc29?usp=sharing

## What the video shows

- Boot-up on the Raspberry Pi 5 with the Pi Camera (Rev 1.3) attached
- Terminal prompt for mode selection (self-calibrate vs ergonomics research)
- Live MediaPipe skeleton overlay + 0-100 posture score in the browser
- State machine in action: sustained bad posture → buzzer fires after 10 s; sustained good posture → buzzer silences after 3 s
- Web-UI labeling buttons writing per-frame rows to `data/session_*.csv`
