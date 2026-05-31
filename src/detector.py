"""Object detection for ARGOS — general categories via YOLO26.

This is the "general" recognition track (80 pretrained COCO classes). The
"personal" few-shot track will live in learner.py (Phase 3). Live preview:

    python -m src.detector

Press q or ESC to quit. First run downloads the YOLO26n weights (~6 MB);
they are gitignored.
"""
from __future__ import annotations

import sys
import time

import cv2
from ultralytics import YOLO
from ultralytics import settings as yolo_settings

from . import config
from .camera import Camera, CameraError

# Privacy first: turn off Ultralytics analytics/telemetry (no phone-home).
if config.DISABLE_ULTRALYTICS_TELEMETRY:
    try:
        yolo_settings.update({"sync": False})
    except Exception:
        pass


class Detector:
    """Wraps a YOLO26 model for single-frame inference."""

    def __init__(self, model: str | None = None,
                 device: str | None = None, conf: float = 0.25) -> None:
        name = model or config.YOLO_MODEL
        local = config.MODELS_DIR / name
        # Load from models/ if present, else let Ultralytics fetch the weights.
        self.model = YOLO(str(local) if local.exists() else name)
        self.device = device or config.DEVICE
        self.conf = conf

    def detect(self, frame):
        """Run inference on one BGR frame; return the Ultralytics Results."""
        return self.model.predict(
            frame, device=self.device, conf=self.conf, verbose=False
        )[0]

    def annotate(self, frame):
        """Return the frame with detection boxes + labels drawn."""
        return self.detect(frame).plot()


def run_detection_preview() -> None:
    window = "ARGOS - detection (press q to quit)"
    det = Detector()
    print(f"[ARGOS] YOLO geladen, device={det.device}. Warmlaufen...")
    fps = 0.0
    prev = time.time()
    try:
        with Camera() as cam:
            for frame in cam.frames():
                annotated = det.annotate(frame)
                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                cv2.putText(
                    annotated, f"{fps:4.1f} FPS  ({det.device})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow(window, annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except CameraError as exc:
        print(f"[ARGOS] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection_preview()
    