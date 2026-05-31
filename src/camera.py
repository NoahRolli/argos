"""Camera capture for ARGOS.

Phase 1: a source-agnostic camera wrapper around OpenCV. Works with the
built-in MacBook camera now (CAMERA_SOURCE=0) and later with USB or RTSP
sources (e.g. CAMERA_SOURCE="rtsp://...") without changing any callers.

Run the live preview from the project root (venv active):

    python -m src.camera

Press q or ESC to quit.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Iterator

import cv2

from . import config


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class Camera:
    """Thin wrapper around cv2.VideoCapture with a context-manager interface."""

    def __init__(self, source: int | str | None = None) -> None:
        self.source = config.CAMERA_SOURCE if source is None else source
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        self._cap = cv2.VideoCapture(self.source)
        if self._cap is None or not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera source {self.source!r}.\n"
                "On macOS, grant camera access to your terminal (or VS Code) "
                "under System Settings > Privacy & Security > Camera, then retry."
            )
        # AVFoundation often returns empty frames at first; warm up briefly.
        for _ in range(10):
            ok, _frame = self._cap.read()
            if ok:
                break
            time.sleep(0.1)
        return self

    def read(self):
        if self._cap is None:
            raise CameraError("Camera is not open — call open() first.")
        ok, frame = self._cap.read()
        if not ok:
            raise CameraError(
                "Failed to read a frame. If this happens immediately, macOS "
                "may be blocking camera access (check Privacy & Security)."
            )
        return frame

    def frames(self) -> Iterator:
        while True:
            yield self.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


def run_preview() -> None:
    """Open the camera and show a live preview with an FPS overlay."""
    window = "ARGOS - camera preview (press q to quit)"
    fps = 0.0
    prev = time.time()
    try:
        with Camera() as cam:
            for frame in cam.frames():
                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                h, w = frame.shape[:2]
                cv2.putText(
                    frame, f"{w}x{h}  {fps:4.1f} FPS",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow(window, frame)
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
    run_preview()