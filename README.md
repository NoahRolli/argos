# ARGOS

Local, privacy-first computer vision for macOS (Apple Silicon). ARGOS sees what
your camera sees, recognizes objects in real time, learns new ones from your
feedback, and reads hand gestures as commands — **fully on-device, no cloud**.

## Why fully local

Camera frames are personal, sometimes biometric, data. Keeping every step on the
machine means that data never leaves it — that is ARGOS's primary security
property, not an afterthought.

## Architecture

| Module            | Role                                                        |
|-------------------|-------------------------------------------------------------|
| `src/camera.py`   | Frame source — built-in cam now, USB/RTSP later (OpenCV)    |
| `src/detector.py` | General object detection (YOLO26, MPS-accelerated)          |
| `src/learner.py`  | Personal few-shot recognition (embeddings + NN) — planned   |
| `src/gestures.py` | Hand landmarks + gesture classification (MediaPipe) — planned |
| `src/config.py`   | Loads all configuration from `.env`                         |

Two learning tracks: general categories come pretrained from YOLO26; your own
objects are learned few-shot — an embedding is stored with the label you give,
and recognition is a nearest-neighbor lookup. No GPU retraining.

## Status

- [x] Phase 0 — Foundation & security (gitignore, pre-commit hook, config)
- [x] Phase 1 — Camera loop with live preview
- [x] Phase 2 — Real-time object detection (YOLO26)
- [ ] Phase 3 — Few-shot learning from feedback
- [ ] Phase 4 — Hand-gesture commands
- [ ] Phase 5 — Additional camera sources (USB / RTSP)

## Setup (Apple Silicon)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
cp .env.example .env

chmod +x scripts/pre-commit-hook.sh
ln -sf ../../scripts/pre-commit-hook.sh .git/hooks/pre-commit
```

macOS prompts for camera access on first run; grant it under
System Settings → Privacy & Security → Camera. Run the previews from the
project root:

```bash
python -m src.camera      # plain camera feed
python -m src.detector    # feed with object detection
```

## Security

ARGOS extends a repo-level baseline with the camera-data dimension:

- Never committed: captured frames, datasets, learned embeddings, model weights
  (all gitignored; embeddings are treated as sensitive as secrets).
- A pre-commit hook blocks media files, embedding blobs, model weights, and
  common secret patterns before they can be committed.
- No telemetry, no network calls in normal operation, no silent frame saving.

To verify the hook is active, try committing a fake secret — it must be blocked:

```bash
printf 'API_KEY = "%s"\n' 'sk-1234567890abcdefghijklmnop' > test_evil.py
git add test_evil.py && git commit -m test
git rm --cached test_evil.py 2>/dev/null; rm -f test_evil.py
```

## License

No license chosen yet. Note that Ultralytics YOLO is AGPL-3.0, which can affect
your options if you redistribute — worth deciding before going further.
