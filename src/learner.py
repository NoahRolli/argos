"""Few-shot personal recognition for ARGOS (Phase 3).

The "personal" learning track. YOLO26 (detector.py) proposes object boxes;
here you teach ARGOS your own names for them. Labeling a box stores an
embedding vector locally; future frames are matched by cosine similarity
(nearest neighbor) — no retraining.

    python -m src.learner

Controls in the preview window:
    0-9   select that numbered box and label it (type the label in the terminal)
    u     undo the last taught example
    q/ESC quit

Taught examples persist in data/embeddings/ (gitignored — as sensitive as
captured images). First run downloads the MobileNet weights once (~10 MB).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from . import config
from .camera import Camera, CameraError
from .detector import Detector

_STORE_PATH = config.EMBEDDINGS_DIR / "personal.npz"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Embedder:
    """Turns an image crop (BGR ndarray) into an L2-normalized feature vector."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or config.DEVICE
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, crop_bgr) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        feat = self.model(t).squeeze(0).float().cpu().numpy()
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 0 else feat


class EmbeddingStore:
    """Labels + their example vectors, persisted to an .npz file."""

    def __init__(self, path: Path = _STORE_PATH) -> None:
        self.path = path
        self.labels: list[str] = []
        self.vectors: list[np.ndarray] = []
        if self.path.exists():
            data = np.load(self.path, allow_pickle=True)
            self.labels = list(data["labels"])
            self.vectors = [np.asarray(v, dtype=np.float32) for v in data["vectors"]]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        vecs = (np.stack(self.vectors).astype(np.float32)
                if self.vectors else np.zeros((0, 1), np.float32))
        np.savez(self.path, labels=np.array(self.labels, dtype=object), vectors=vecs)

    def add(self, label: str, vector: np.ndarray) -> None:
        self.labels.append(label)
        self.vectors.append(vector)
        self.save()

    def undo(self) -> str | None:
        if not self.labels:
            return None
        label = self.labels.pop()
        self.vectors.pop()
        self.save()
        return label

    def match(self, vector: np.ndarray, threshold: float):
        """Return (label, score) of the best match above threshold, else None."""
        if not self.vectors:
            return None
        sims = [float(np.dot(vector, v)) for v in self.vectors]
        idx = int(np.argmax(sims))
        return (self.labels[idx], sims[idx]) if sims[idx] >= threshold else None


def run_learning_preview() -> None:
    window = "ARGOS - learning (0-9 label, u undo, q quit)"
    det = Detector()
    emb = Embedder()
    store = EmbeddingStore()
    print(f"[ARGOS] learner bereit ({len(store.labels)} gelernte Beispiele).")
    fps, prev = 0.0, time.time()
    try:
        with Camera() as cam:
            for frame in cam.frames():
                result = det.detect(frame)
                boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
                    cls = result.boxes.cls.cpu().numpy().astype(int)
                    for i, (b, c) in enumerate(zip(xyxy, cls)):
                        if i >= 10:
                            break
                        boxes.append((b, result.names.get(int(c), str(c))))

                for i, (b, yolo_name) in enumerate(boxes):
                    x1, y1, x2, y2 = b
                    text, color = f"[{i}] {yolo_name}", (0, 200, 0)
                    if store.labels:
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size:
                            m = store.match(emb.embed(crop), config.RECOGNITION_THRESHOLD)
                            if m:
                                text, color = f"[{i}] {m[0]} ({m[1]:.2f})", (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                now = time.time(); dt = now - prev; prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                cv2.putText(frame, f"{fps:4.1f} FPS  taught={len(store.labels)}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("u"):
                    removed = store.undo()
                    print(f"[ARGOS] undo: {removed!r}" if removed else "[ARGOS] nichts da")
                elif ord("0") <= key <= ord("9"):
                    idx = key - ord("0")
                    if idx < len(boxes):
                        b, yolo_name = boxes[idx]
                        x1, y1, x2, y2 = b
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size:
                            label = input(f"[ARGOS] Label für Box {idx} (YOLO: {yolo_name}): ").strip()
                            if label:
                                store.add(label, emb.embed(crop))
                                print(f"[ARGOS] gelernt: {label!r} ({len(store.labels)} gesamt)")
                            else:
                                print("[ARGOS] abgebrochen")
                    else:
                        print(f"[ARGOS] keine Box mit Index {idx}")
    except CameraError as exc:
        print(f"[ARGOS] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_learning_preview()