"""5-class colour model + preprocessing / quality helpers for /predict.

Loads the nearest-centroid artifact trained in ml/ on the *serving* feature
pipeline (white-balance against the paper, mask the powder, mean colour ->
chromaticity). Leave-one-spread-out: 98.2% of 600 photos, 19/19 spreads.

No TensorFlow is needed for inference.
"""
from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = REPO_ROOT / "ml"

INVALID_MSG = "Upload a clear JPG/PNG turmeric powder photo."
RETAKE_TIPS = (
    "Use bright diffuse light, fill frame with powder and keep some white "
    "paper visible, no shadows/containers, hold steady."
)

CANDIDATE_MODELS = [
    Path(os.environ["TURMERIC_COLOR_MODEL"])
    if os.environ.get("TURMERIC_COLOR_MODEL")
    else None,
    REPO_ROOT / "ml" / "model" / "color_centroids_5class_fullframe.json",
    REPO_ROOT / "ml" / "model" / "color_centroids_5class.json",
]

_model = None
_model_path: Path | None = None
_lock = threading.Lock()


def _color() -> object:
    """Import ml/color_model lazily, without requiring PYTHONPATH."""
    if str(ML_DIR) not in sys.path:
        sys.path.insert(0, str(ML_DIR))
    import color_model  # noqa: PLC0415

    return color_model


def find_model() -> Path | None:
    for p in CANDIDATE_MODELS:
        if p is not None and p.is_file():
            return p
    return None


def get_model() -> tuple[dict, Path]:
    """Lazily load the JSON model once (thread-safe)."""
    global _model, _model_path
    if _model is not None:
        return _model, _model_path
    with _lock:
        if _model is not None:
            return _model, _model_path
        path = find_model()
        if path is None:
            raise FileNotFoundError(
                "Colour model artifact not found. Train first "
                "(`python train_color.py` in ml/) or set TURMERIC_COLOR_MODEL."
            )
        _model = _color().load_model(path)
        _model_path = path
        return _model, _model_path


def decode_image(raw: bytes) -> Image.Image | None:
    """Return PIL RGB image, or None for corrupt/unreadable bytes."""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode -> raises on corrupt files
        return img.convert("RGB")
    except Exception:
        return None


def predict_image(pil_img: Image.Image) -> dict:
    """Return label, distances, OOD flag and a display summary.

    Raises ValueError when the powder or the paper reference cannot be
    found, so the route can return a helpful 400 instead of a wrong label.
    """
    cm = _color()
    model, _ = get_model()
    rgb_mean = cm.powder_rgb(pil_img)
    feats = cm.to_model_features(rgb_mean, model)
    result = cm.predict_features(feats, model)
    result["summary"] = cm.image_summary(pil_img)
    result["features"] = {
        "r_chroma": round(float(feats[0]), 5),
        "g_chroma": round(float(feats[1]), 5),
    }
    return result


def model_card() -> dict:
    """Everything the showcase page shows about the model itself."""
    model, path = get_model()
    cv = model.get("cv", {})
    return {
        "classes": model["classes"],
        "pretty": model["pretty"],
        "features": model.get("features", "rgb"),
        "feature_pipeline": model.get("feature_pipeline", ""),
        "artifact": path.name if path else None,
        "artifact_bytes": path.stat().st_size if path else None,
        "n_photos": model.get("n_photos"),
        "n_spreads": model.get("n_spreads"),
        "per_class": model.get("per_class", {}),
        "confusion": model.get(
            "confusion", {"labels": model["classes"], "matrix": []}
        ),
        "cv": cv,
        "labels": model.get("labels", {}),
        "notes": model.get("notes", ""),
    }


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
