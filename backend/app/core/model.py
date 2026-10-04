"""Model singleton + preprocessing / quality helpers for /predict.

Loads the frozen MobileNetV2 artifact trained in ml/ (95% test accuracy).
Preprocessing is identical to training: 224x224 RGB -> MobileNetV2
preprocess_input ([-1, 1]).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["home_ground", "market_bought"]
IMG_SIZE = (224, 224)
INVALID_MSG = "Upload a clear JPG/PNG turmeric powder photo."
RETAKE_TIPS = (
    "Use bright diffuse light, fill frame with powder, "
    "no shadows/containers, hold steady."
)

BACKEND_DIR = Path(__file__).resolve().parents[2]  # backend/
REPO_ROOT = BACKEND_DIR.parent
CANDIDATE_MODELS = [
    Path(os.environ["TURMERIC_MODEL_PATH"])
    if os.environ.get("TURMERIC_MODEL_PATH")
    else None,
    REPO_ROOT / "ml" / "model" / "turmeric_mobilenet.keras",
    REPO_ROOT / "ml" / "artifacts" / "turmeric_mobilenet.keras",
]

_model = None
_model_path: Path | None = None
_lock = threading.Lock()


def find_model() -> Path | None:
    for p in CANDIDATE_MODELS:
        if p is not None and p.is_file():
            return p
    return None


def get_model():
    """Lazily load the .keras artifact once (thread-safe)."""
    global _model, _model_path
    if _model is not None:
        return _model, _model_path
    with _lock:
        if _model is not None:
            return _model, _model_path
        import tensorflow as tf

        path = find_model()
        if path is None:
            raise FileNotFoundError(
                "Model artifact not found. Train first (`python train.py` in ml/) "
                "or set TURMERIC_MODEL_PATH."
            )
        _model = tf.keras.models.load_model(path)
        _model_path = path
        return _model, _model_path


def decode_image(raw: bytes) -> Image.Image | None:
    """Return PIL RGB image, or None for corrupt/unreadable bytes."""
    try:
        import io

        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode -> raises on corrupt files
        return img.convert("RGB")
    except Exception:
        return None


def preprocess(pil_img: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype(np.float32)
    return np.expand_dims(preprocess_input(arr), 0)  # float32[1,224,224,3]


def quality_warnings(pil_img: Image.Image) -> list[str]:
    """Heuristic dark/blurry warnings (guidance only, not a hard reject)."""
    warnings: list[str] = []
    small = np.asarray(pil_img.convert("RGB").resize((256, 256))).astype(np.float32)
    if float(small.mean()) < 60:
        warnings.append(
            f"Image looks dark (brightness {small.mean():.0f}/255). {RETAKE_TIPS}"
        )
    g = np.asarray(pil_img.convert("L").resize((256, 256))).astype(np.float32)
    try:
        from scipy.signal import convolve2d  # type: ignore

        lap = convolve2d(
            g, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
            mode="valid",
        )
    except Exception:
        lap = (
            np.roll(g, 1, 0)
            + np.roll(g, -1, 0)
            + np.roll(g, 1, 1)
            + np.roll(g, -1, 1)
            - 4 * g
        )[1:-1, 1:-1]
    if float(lap.var()) < 60:
        warnings.append(f"Image looks blurry (low sharpness). {RETAKE_TIPS}")
    return warnings


def predict_array(x: np.ndarray) -> tuple[str, float, dict[str, float]]:
    model, _ = get_model()
    probs = [float(p) for p in np.asarray(model.predict(x, verbose=0))[0]]
    idx = int(np.argmax(probs))
    return (
        CLASS_NAMES[idx],
        round(probs[idx] * 100, 1),
        {CLASS_NAMES[i]: round(p * 100, 1) for i, p in enumerate(probs)},
    )
