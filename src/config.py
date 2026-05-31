"""Central configuration for vision-learn.

Loads everything from the environment (.env) per SECURITY_SETUP.md Layer 4:
all config via env, and NO silent defaults for anything that must be set —
better to crash on startup than to run mis-configured.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env from project root if present

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

for _d in (MODELS_DIR, CAPTURES_DIR, EMBEDDINGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {key} (see .env.example)")
    return val


def _bool(key: str, default: str = "false") -> bool:
    return _env(key, default).strip().lower() in {"1", "true", "yes", "on"}


# --- Runtime ----------------------------------------------------------------
# CAMERA_SOURCE is "0" (index) for the built-in cam, or an RTSP URL later.
_raw_source = _env("CAMERA_SOURCE", "0")
CAMERA_SOURCE: int | str = int(_raw_source) if _raw_source.isdigit() else _raw_source

DEVICE = _env("DEVICE", "mps")  # mps on M3, cpu as fallback

# --- Models -----------------------------------------------------------------
YOLO_MODEL = _env("YOLO_MODEL", "yolo26n.pt")
GESTURE_MODEL = _env("GESTURE_MODEL", "gesture_recognizer.task")

# --- Learning ---------------------------------------------------------------
RECOGNITION_THRESHOLD = float(_env("RECOGNITION_THRESHOLD", "0.75"))

# --- Privacy / Security -----------------------------------------------------
DISABLE_ULTRALYTICS_TELEMETRY = _bool("DISABLE_ULTRALYTICS_TELEMETRY", "true")
PERSIST_FRAMES = _bool("PERSIST_FRAMES", "false")

