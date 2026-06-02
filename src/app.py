"""ARGOS application logic — the unified live view, encapsulated."""
from __future__ import annotations

import datetime
import re
import sys
import time
from pathlib import Path

import cv2

from . import config
from .camera import Camera, CameraError
from .detector import Detector
from .faces import FaceID
from .learner import Embedder, EmbeddingStore

_DATASETS_DIR = config.DATA_DIR / "datasets"
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip()) or "unlabeled"


def _count_collected() -> int:
    return sum(1 for _ in _DATASETS_DIR.rglob("*.jpg")) if _DATASETS_DIR.exists() else 0


class ArgosApp:
    """Holds the models/state and runs the unified live view."""

    def __init__(self) -> None:
        self.det = Detector()
        self.faces = FaceID()
        self.embedder = Embedder()
        self.store = EmbeddingStore()
        self.collected = _count_collected()
        self.frozen = False
        self.fps = 0.0
        self._prev = time.time()
        self._snap_clean = None
        self._snap_boxes: list = []
        self._snap_disp = None

    def _save_example(self, crop, label: str) -> Path:
        d = _DATASETS_DIR / _slug(label)
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = d / f"{ts}.jpg"
        cv2.imwrite(str(path), crop)
        return path

    def _process(self, frame) -> None:
        clean = frame.copy()      # sauberes Bild zum Ausschneiden
        disp = frame              # auf dieses wird gezeichnet
        boxes = []
        result = self.det.detect(clean)
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for b, c in zip(xyxy, cls):
                name = result.names.get(int(c), str(c))
                x1, y1, x2, y2 = b
                if name == "person":
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 0), 1)
                    continue
                idx = len(boxes)
                if idx >= 10:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 200, 0), 1)
                    continue
                text, color = f"[{idx}] {name}", (0, 200, 0)
                if self.store.labels:
                    crop = clean[max(0, y1):y2, max(0, x1):x2]
                    if crop.size:
                        m = self.store.match(self.embedder.embed(crop), name,
                                             config.RECOGNITION_THRESHOLD)
                        if m:
                            text = f"[{idx}] {name} | {m[0]} ({m[1]:.2f})"
                            color = (0, 165, 255)
                boxes.append((b, name))
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(disp, text, (x1, max(20, y1 - 8)), _FONT, 0.6, color, 2, cv2.LINE_AA)

        for face in self.faces.detect(clean):
            x, y, w, h = face[:4].astype(int)
            ftext, fcolor = "unbekannt", (0, 0, 255)
            try:
                m = self.faces.identify(self.faces.embed(clean, face))
                if m:
                    ftext, fcolor = m[0], (255, 200, 0)
            except cv2.error:
                continue
            cv2.rectangle(disp, (x, y), (x + w, y + h), fcolor, 2)
            cv2.putText(disp, ftext, (x, max(20, y - 8)), _FONT, 0.6, fcolor, 2, cv2.LINE_AA)

        self._snap_clean, self._snap_boxes, self._snap_disp = clean, boxes, disp

    def _overlay(self):
        shown = self._snap_disp.copy()
        cv2.putText(shown,
                    f"ARGOS {self.fps:4.1f}FPS  taught={len(self.store.labels)} "
                    f"collected={self.collected} conf={self.det.conf:.2f}",
                    (10, 28), _FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if self.frozen:
            cv2.putText(shown, "EINGEFROREN - Ziffer waehlen, Leertaste = weiter",
                        (10, shown.shape[0] - 15), _FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return shown

    def _label_box(self, idx: int) -> None:
        if idx >= len(self._snap_boxes):
            print(f"[ARGOS] keine Box mit Index {idx}")
            return
        (x1, y1, x2, y2), name = self._snap_boxes[idx]
        crop = self._snap_clean[max(0, y1):y2, max(0, x1):x2]
        if not crop.size:
            return
        ans = input(
            f"[ARGOS] [{idx}] '{name}': [Enter]=korrekt bestätigen | "
            f"!<klasse>=korrigieren | <text>=eigenes Label | q=abbrechen\n> "
        ).strip()
        if ans == "":
            p = self._save_example(crop, name); self.collected += 1
            print(f"[ARGOS] bestätigt: {name} -> {p.parent.name}/ ({self.collected} gesamt)")
        elif ans.lower() in ("q", "c"):
            print("[ARGOS] abgebrochen")
        elif ans.startswith("!"):
            corrected = ans[1:].strip()
            if corrected:
                p = self._save_example(crop, corrected); self.collected += 1
                print(f"[ARGOS] korrigiert: {name} -> {p.parent.name}/ ({self.collected} gesamt)")
            else:
                print("[ARGOS] keine Klasse angegeben")
        else:
            self.store.add(ans, name, self.embedder.embed(crop))
            print(f"[ARGOS] gelernt: {name} | {ans!r} ({len(self.store.labels)} gesamt)")

    def _handle_key(self, key: int) -> bool:
        """Return False to quit."""
        if key in (ord("q"), 27):
            return False
        if key == 32:  # Leertaste
            self.frozen = not self.frozen
        elif key == ord("u"):
            removed = self.store.undo()
            print(f"[ARGOS] undo: {removed!r}" if removed else "[ARGOS] nichts da")
        elif key in (ord("-"), ord("_")):
            self.det.conf = max(0.05, round(self.det.conf - 0.05, 2))
            print(f"[ARGOS] conf={self.det.conf:.2f} (niedriger = mehr Objekte)")
        elif key in (ord("+"), ord("=")):
            self.det.conf = min(0.90, round(self.det.conf + 0.05, 2))
            print(f"[ARGOS] conf={self.det.conf:.2f}")
        elif ord("0") <= key <= ord("9"):
            self._label_box(key - ord("0"))
        return True

    def _tick_fps(self) -> None:
        now = time.time(); dt = now - self._prev; self._prev = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

    def run(self) -> None:
        print(f"[ARGOS] bereit — {len(self.store.labels)} Few-Shot-Beispiele, "
              f"{len(set(self.faces.names))} Gesichter, {self.collected} gesammelte Bilder.")
        print("[ARGOS] Leertaste=einfrieren, dann 0-9 wählen. -/+ = Empfindlichkeit, "
              "u=undo, q/ESC=beenden.")
        window = "ARGOS"
        try:
            with Camera() as cam:
                for frame in cam.frames():
                    if not self.frozen:
                        self._process(frame)
                        self._tick_fps()
                    cv2.imshow(window, self._overlay())
                    if not self._handle_key(cv2.waitKey(1) & 0xFF):
                        break
        except CameraError as exc:
            print(f"[ARGOS] {exc}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()