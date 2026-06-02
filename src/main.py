"""ARGOS — unified live view with inline teaching and feedback collection.

    python -m src.main

Controls in the window:
    0-9   select that numbered object box, then in the terminal:
            Enter      confirm correct -> save as training example (YOLO class)
            !<class>   correct a wrong detection -> save under the right class
            <text>     give your own few-shot label (live personal recognition)
            q          cancel
    u     undo the last few-shot example
    q/ESC quit

Training crops go to data/datasets/<class>/ (gitignored). Persons are
identified by face recognition (python -m src.faces enroll), not object teaching.
"""
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


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s.strip()) or "unlabeled"


def _save_example(crop, label: str) -> Path:
    d = _DATASETS_DIR / _slug(label)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = d / f"{ts}.jpg"
    cv2.imwrite(str(path), crop)
    return path


def run() -> None:
    det = Detector()
    face_id = FaceID()
    obj_embedder = Embedder()
    obj_store = EmbeddingStore()
    print(f"[ARGOS] bereit — {len(obj_store.labels)} Few-Shot-Beispiele, "
          f"{len(set(face_id.names))} bekannte Gesichter.")
    print("[ARGOS] Im Fenster: 0-9 Objekt wählen, dann Enter=korrekt / !klasse / Label / q im Terminal. "
          "u = undo, q/ESC = beenden.")

    window = "ARGOS (0-9 select, u undo, q quit)"
    fps, prev = 0.0, time.time()
    try:
        with Camera() as cam:
            for frame in cam.frames():
                result = det.detect(frame)
                boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                    cls = result.boxes.cls.cpu().numpy().astype(int)
                    for b, c in zip(xyxy, cls):
                        name = result.names.get(int(c), str(c))
                        x1, y1, x2, y2 = b
                        if name == "person":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                            continue
                        idx = len(boxes)
                        if idx >= 10:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                            continue
                        text, color = f"[{idx}] {name}", (0, 200, 0)
                        if obj_store.labels:
                            crop = frame[max(0, y1):y2, max(0, x1):x2]
                            if crop.size:
                                m = obj_store.match(obj_embedder.embed(crop), name,
                                                    config.RECOGNITION_THRESHOLD)
                                if m:
                                    text = f"[{idx}] {name} | {m[0]} ({m[1]:.2f})"
                                    color = (0, 165, 255)
                        boxes.append((b, name))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                for face in face_id.detect(frame):
                    x, y, w, h = face[:4].astype(int)
                    text, color = "unbekannt", (0, 0, 255)
                    try:
                        m = face_id.identify(face_id.embed(frame, face))
                        if m:
                            text, color = m[0], (255, 200, 0)
                    except cv2.error:
                        continue
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                now = time.time(); dt = now - prev; prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                cv2.putText(frame, f"ARGOS  {fps:4.1f} FPS  taught={len(obj_store.labels)}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("u"):
                    removed = obj_store.undo()
                    print(f"[ARGOS] undo: {removed!r}" if removed else "[ARGOS] nichts da")
                elif ord("0") <= key <= ord("9"):
                    idx = key - ord("0")
                    if idx >= len(boxes):
                        print(f"[ARGOS] keine Box mit Index {idx}")
                        continue
                    (x1, y1, x2, y2), name = boxes[idx]
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if not crop.size:
                        continue
                    ans = input(
                        f"[ARGOS] [{idx}] '{name}': [Enter]=korrekt bestätigen | "
                        f"!<klasse>=korrigieren | <text>=eigenes Label | q=abbrechen\n> "
                    ).strip()
                    if ans == "":
                        p = _save_example(crop, name)
                        print(f"[ARGOS] bestätigt: {name} -> {p.parent.name}/")
                    elif ans.lower() in ("q", "c"):
                        print("[ARGOS] abgebrochen")
                    elif ans.startswith("!"):
                        corrected = ans[1:].strip()
                        if corrected:
                            p = _save_example(crop, corrected)
                            print(f"[ARGOS] korrigiert: {name} -> {p.parent.name}/")
                        else:
                            print("[ARGOS] keine Klasse angegeben")
                    else:
                        obj_store.add(ans, name, obj_embedder.embed(crop))
                        print(f"[ARGOS] gelernt: {name} | {ans!r} ({len(obj_store.labels)} gesamt)")
    except CameraError as exc:
        print(f"[ARGOS] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()