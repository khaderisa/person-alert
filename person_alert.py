"""
👁️ Person Alert - YOLO Edition
Alerts you when 2+ people are detected at the same time.
Press Q to quit | Press S to toggle sensitivity
"""

import cv2
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# ─── CONFIG ───────────────────────────────────────────────────────────────────
COOLDOWN_SECONDS   = 4      # Min seconds between alerts
FLASH_DURATION     = 1.5    # How long the red flash stays
ALERT_WHEN         = 2      # Alert when >= this many people detected
CONFIDENCE         = 0.5    # Detection confidence threshold (0.0 - 1.0)
# ──────────────────────────────────────────────────────────────────────────────

class PersonAlertApp:
    def __init__(self):
        print("⏳ Loading YOLO model (first run downloads ~6MB)...")
        self.model = YOLO("yolov8n.pt")   # nano = fastest
        print("✅ Model loaded!\n")

        self.last_alert_time  = 0
        self.flash_until      = 0
        self.total_detections = 0
        self.fps              = 0
        self.boxes            = []

        # Desktop notification
        self.notifier_available = False
        try:
            from win10toast import ToastNotifier
            self.toaster = ToastNotifier()
            self.notifier_available = True
        except Exception:
            pass

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect_people(self, frame):
        results = self.model(frame, classes=[0], conf=CONFIDENCE, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
        return boxes

    # ── Alert ─────────────────────────────────────────────────────────────────
    def trigger_alert(self, count):
        now = time.time()
        if now - self.last_alert_time < COOLDOWN_SECONDS:
            return
        self.last_alert_time   = now
        self.flash_until        = now + FLASH_DURATION
        self.total_detections  += 1

        if self.notifier_available:
            def notify():
                try:
                    self.toaster.show_toast(
                        "⚠️ Multiple people detected!",
                        f"{count} people detected at the same time!",
                        duration=3,
                        threaded=True
                    )
                except Exception:
                    pass
            threading.Thread(target=notify, daemon=True).start()

    # ── Draw ──────────────────────────────────────────────────────────────────
    def draw_ui(self, frame, boxes):
        import math, numpy as np
        h, w = frame.shape[:2]
        now      = time.time()
        is_alert = now < self.flash_until
        count    = len(boxes)

        # Red flash overlay
        if is_alert:
            alpha   = 0.4 + 0.15 * abs(math.sin((now - self.flash_until + FLASH_DURATION) * 5))
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 220), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw boxes around each person
        for (x1, y1, x2, y2, conf) in boxes:
            color = (0, 60, 255) if is_alert else (0, 220, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"Person {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Top status bar
        bar_color = (0, 0, 160) if is_alert else (25, 25, 25)
        cv2.rectangle(frame, (0, 0), (w, 52), bar_color, -1)

        if is_alert:
            msg = f"!!  {count} PEOPLE DETECTED AT ONCE  !!"
            cv2.putText(frame, msg, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 220, 255), 2)
        elif count == 1:
            cv2.putText(frame, "1 person — watching...", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2)
        else:
            cv2.putText(frame, "All Clear", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 230, 100), 2)

        cv2.putText(frame, f"{self.fps:.0f} FPS", (w - 90, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, f"Alerts: {self.total_detections}", (w - 110, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Bottom bar
        cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(frame,
                    f"[Q] Quit   Alert when >= {ALERT_WHEN} people   {datetime.now().strftime('%H:%M:%S')}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("✅ Camera started. Press Q to quit.\n")

        prev_time   = time.time()
        frame_count = 0
        boxes       = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Run YOLO every other frame for speed
            if frame_count % 2 == 0:
                boxes = self.detect_people(frame)
                if len(boxes) >= ALERT_WHEN:
                    self.trigger_alert(len(boxes))

            now = time.time()
            self.fps  = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            frame = self.draw_ui(frame, boxes)
            cv2.imshow("Person Alert - YOLO", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. Total alerts: {self.total_detections}")


if __name__ == "__main__":
    app = PersonAlertApp()
    app.run()