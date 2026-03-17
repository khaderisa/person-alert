# person-alert
# 👁️ Person Alert - AI-Powered Presence Detector

A real-time person detection app that visually alerts you when multiple people 
are nearby — built for anyone wearing headphones who can't rely on sound cues.

## How it works
Uses **YOLOv8** (You Only Look Once), a state-of-the-art AI object detection 
model, to analyze your webcam feed in real time and detect people in the frame.
When 2 or more people are detected simultaneously, it triggers:
- 🔴 A flashing red screen overlay
- 🔔 A Windows desktop toast notification

## Features
- AI-powered detection using YOLOv8 neural network
- Real-time bounding boxes drawn around each detected person
- Smart cooldown system to avoid notification spam
- Adjustable confidence threshold
- Lightweight — runs on CPU, no GPU required

## Requirements
- Python 3.8+
- Webcam

## Installation
pip install opencv-python ultralytics plyer win10toast

## Usage
python person_alert.py

Press Q to quit.
```

---

Short **repo description** (the one-liner under the repo name):
```
AI-powered webcam app that alerts you when multiple people are nearby — built for headphone users 🎧
