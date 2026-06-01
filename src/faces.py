"""Face recognition for ARGOS — "recognize me" from enrolled photos.

Uses OpenCV's built-in YuNet (detection) + SFace (128-d identity embedding),
so it needs no dependency beyond the OpenCV you already have — sidestepping the
Python 3.14 incompatibilities of insightface / facenet-pytorch / deepface.

Models live in models/ (gitignored). Enroll yourself by putting a few photos in
data/enroll/<YourName>/ (jpg/png), then:

    python -m src.faces enroll      # build the face DB from data/enroll/
    python -m src.faces             # live recognition (q/ESC to quit)

Enrolled photos and the face DB stay local and gitignored.
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from . import config
from .camera import Camera, CameraError

_DETECTOR_MODEL = config.MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_RECOGNIZER_MODEL = config.MODELS_DIR / "face_recognition_sface_2021dec.onnx"
_DB_PATH = config.EMBEDDINGS_DIR / "faces.npz"
_ENROLL_DIR = config.DATA_DIR / "enroll"

# SFace cosine threshold for "same identity" (OpenCV default). Higher = stricter.
COSINE_THRESHOLD = 0.363


class FaceID:
    def __init__(self) -> None:
        for m in (_DETECTOR_MODEL, _RECOGNIZER_MODEL):
            if not m.exists():
                raise FileNotFoundError(
                    f"Missing model {m.name} in models/. Download it first "
                    "(see the curl commands in the README/docstring)."
                )
        self.detector = cv2.FaceDetectorYN.create(
            str(_DETECTOR_MODEL), "", (320, 320), 0.9, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(_RECOGNIZER_MODEL), "")
        self.names: list[str] = []
        self.vectors: list[np.ndarray] = []
        if _DB_PATH.exists():
            data = np.load(_DB_PATH, allow_pickle=True)
            self.names = list(data["names"])
            self.vectors = [np.asarray(v, np.float32) for v in data["vectors"]]

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def embed(self, frame, face_row) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face_row)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 0 else feat

    def identify(self, vector):
        if not self.vectors:
            return None
        sims = [float(np.dot(vector, v)) for v in self.vectors]
        idx = int(np.argmax(sims))
        return (self.names[idx], sims[idx]) if sims[idx] >= COSINE_THRESHOLD else None

    def _save_db(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        vecs = (np.stack(self.vectors).astype(np.float32)
                if self.vectors else np.zeros((0, 1), np.float32))
        np.savez(_DB_PATH, names=np.array(self.names, dtype=object), vectors=vecs)

    def enroll_from_folder(self) -> None:
        if not _ENROLL_DIR.exists():
            raise FileNotFoundError(
                f"{_ENROLL_DIR} fehlt. Lege data/enroll/<Name>/ an und füge Fotos hinzu."
            )
        self.names, self.vectors = [], []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for person_dir in sorted(p for p in _ENROLL_DIR.iterdir() if p.is_dir()):
            count = 0
            for img_path in sorted(person_dir.glob("*")):
                if img_path.suffix.lower() not in exts:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[ARGOS]   konnte {img_path.name} nicht lesen")
                    continue
                faces = self.detect(img)
                if faces.shape[0] == 0:
                    print(f"[ARGOS]   kein Gesicht in {img_path.name}")
                    continue
                best = max(faces, key=lambda f: float(f[2] * f[3]))  # größtes Gesicht
                self.names.append(person_dir.name)
                self.vectors.append(self.embed(img, best))
                count += 1
            print(f"[ARGOS] {person_dir.name}: {count} Foto(s) eingelernt")
        self._save_db()
        print(f"[ARGOS] Face-DB gespeichert ({len(self.names)} Vektoren).")


def run_recognition_preview() -> None:
    fid = FaceID()
    if not fid.vectors:
        print("[ARGOS] Leere Face-DB — erst 'python -m src.faces enroll' ausführen.")
    window = "ARGOS - face recognition (q to quit)"
    fps, prev = 0.0, time.time()
    try:
        with Camera() as cam:
            for frame in cam.frames():
                for face in fid.detect(frame):
                    x, y, w, h = face[:4].astype(int)
                    text, color = "unbekannt", (0, 0, 255)
                    try:
                        m = fid.identify(fid.embed(frame, face))
                        if m:
                            text, color = f"{m[0]} ({m[1]:.2f})", (0, 200, 0)
                    except cv2.error:
                        continue
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                now = time.time(); dt = now - prev; prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                cv2.putText(frame, f"{fps:4.1f} FPS  known={len(set(fid.names))}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "enroll":
        FaceID().enroll_from_folder()
    else:
        run_recognition_preview()


if __name__ == "__main__":
    main()