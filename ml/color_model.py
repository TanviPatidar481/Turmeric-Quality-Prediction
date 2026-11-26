"""Colour-based 5-class turmeric impurity model.

The classifier is a nearest-centroid model on the white-balanced mean colour
of the powder region. Feature sets:

    chromaticity_rg  r/(r+g+b), g/(r+g+b)   -> the shipped model (98.2% LOSO)
    rgb              mean R, G, B           -> legacy artifacts
    gb               mean G, B              -> legacy ablation

The paper reference around the spread is what makes the features
comparable between sessions: raw RGB drifts when the camera white
balance changes, the corrected powder colour does not.

Artifacts:
    model/color_centroids_5class_fullframe.json  shipped (serving pipeline)
    model/color_centroids_5class.json            crop-trained reference
    model/color_centroids_6class.json            6-class reference
    model/color_centroids.json                   legacy RGB model

CLI:
    python color_model.py path/to/photo.jpg
    python color_model.py path/to/photo.jpg --model model/color_centroids_5class.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare_crop import load_rgb, masks, wb_gains  # noqa: E402

MODEL_PATH = HERE / "model" / "color_centroids_5class_fullframe.json"
PROCESS_LONG_SIDE = 640
MIN_PAPER_PIXELS = 500
MIN_POWDER_PIXELS = 1000


def powder_rgb(img: Image.Image, long_side: int = PROCESS_LONG_SIDE) -> np.ndarray:
    """White-balanced mean RGB of the powder region.

    Returns a float32 vector [R, G, B]. Raises ValueError when the frame
    does not contain both a paper reference and a powder mass.
    """
    im = img.convert("RGB")
    im.thumbnail((long_side, long_side), Image.BILINEAR)
    rgb = np.asarray(im).astype(np.float32)

    powder, paper = masks(rgb)
    if int(paper.sum()) < MIN_PAPER_PIXELS:
        raise ValueError("paper reference not found in frame")
    if int(powder.sum()) < MIN_POWDER_PIXELS:
        raise ValueError("powder region not found in frame")

    gains = wb_gains(rgb, paper)
    return np.clip(rgb[powder] * gains, 0.0, 255.0).mean(axis=0)


def features_from_rgb(rgb_mean: np.ndarray, channels: list[int] | tuple[int, ...]) -> np.ndarray:
    """Select the model channels from a mean RGB vector."""
    return np.asarray(rgb_mean, dtype=np.float64)[list(channels)]


def to_model_features(rgb_mean: np.ndarray, model: dict) -> np.ndarray:
    """Build the feature vector the artifact was trained on.

    ``chromaticity_rg`` models take r/(r+g+b), g/(r+g+b); legacy models take a
    subset of the raw mean RGB channels.
    """
    spec = model.get("features", "rgb")
    if spec == "chromaticity_rg":
        v = np.asarray(rgb_mean, dtype=np.float64).ravel()[:3]
        total = max(float(v.sum()), 1e-6)
        return np.array([v[0] / total, v[1] / total])
    if spec == "rgb":
        return features_from_rgb(rgb_mean, model.get("channels", [0, 1, 2]))
    raise ValueError(f"unknown feature spec: {spec}")


def image_summary(img: Image.Image, max_side: int = 360) -> dict:
    """What the pipeline saw: powder coverage, measured colour, small preview.

    Returned for display only. The preview is the bounding box of the largest
    powder mass, white-balanced, so a human can check the model is looking at
    the powder and not at the paper or the table.
    """
    import base64
    import io

    from scipy import ndimage as ndi

    im = img.convert("RGB")
    im.thumbnail((PROCESS_LONG_SIDE, PROCESS_LONG_SIDE), Image.BILINEAR)
    rgb = np.asarray(im).astype(np.float32)
    powder, paper = masks(rgb)
    total = float(rgb.shape[0] * rgb.shape[1])

    gains = wb_gains(rgb, paper) if int(paper.sum()) > 0 else np.ones(3, np.float32)
    balanced = np.clip(rgb * gains, 0, 255).astype(np.uint8)

    out = {
        "powder_fraction": round(float(powder.sum()) / total, 4),
        "paper_fraction": round(float(paper.sum()) / total, 4),
        "preview": None,
        "swatch": None,
    }
    if int(powder.sum()) >= MIN_POWDER_PIXELS:
        lab, n = ndi.label(powder)
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            comp = lab == int(sizes.argmax())
            ys, xs = np.where(comp)
            pad = int(0.06 * max(xs.max() - xs.min(), ys.max() - ys.min()))
            y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
            x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
            patch = Image.fromarray(balanced[y0:y1, x0:x1])
            patch.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            patch.save(buf, format="JPEG", quality=88)
            out["preview"] = "data:image/jpeg;base64," + base64.b64encode(
                buf.getvalue()).decode("ascii")
        mean = np.clip(rgb[powder] * gains, 0, 255).mean(axis=0)
        out["swatch"] = "#%02x%02x%02x" % tuple(int(max(0, min(255, v))) for v in mean)
    return out


def extract_features(path: str | Path, model: dict | None = None) -> np.ndarray:
    model = model or load_model()
    return to_model_features(powder_rgb(load_rgb(Path(path))), model)


def load_model(path: str | Path = MODEL_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def predict_features(feats: np.ndarray, model: dict) -> dict:
    """Classify one feature vector. Returns label, confidence, distances."""
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    centroids = np.asarray(model["centroids"], dtype=np.float64)

    z = (np.asarray(feats, dtype=np.float64) - mean) / std
    dist = np.linalg.norm(centroids - z, axis=1)
    idx = int(np.argmin(dist))

    # Heuristic only: softmax over distance scaled by the typical
    # within-class distance. Not a calibrated probability.
    scale = float(model.get("distance_scale", 1.0))
    probs = _softmax(-0.5 * (dist / max(scale, 1e-6)) ** 2)
    order = np.argsort(dist)
    margin = float(dist[order[1]] - dist[order[0]]) if len(order) > 1 else 0.0
    ood = bool(float(dist[idx]) > float(model.get("ood_threshold", 1e9)))
    confidence = round(float(probs[idx]) * 100, 1)
    warning = None
    if ood:
        # Distance-based confidence is meaningless outside the training range.
        confidence = min(confidence, 50.0)
        warning = "Outside the trained colour range; treat this prediction as unreliable."
    return {
        "label": model["classes"][idx],
        "pretty": model["pretty"][model["classes"][idx]],
        "confidence": confidence,
        "probs": {
            c: round(float(p) * 100, 1) for c, p in zip(model["classes"], probs)
        },
        "distance": round(float(dist[idx]), 3),
        "margin": round(margin, 3),
        "distances": {
            c: round(float(d), 3) for c, d in zip(model["classes"], dist)
        },
        "ood": ood,
        "warning": warning,
    }


def predict_image(path: str | Path, model: dict | None = None) -> dict:
    model = model or load_model()
    return predict_features(extract_features(path, model), model)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classify one turmeric photo.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    args = ap.parse_args()
    print(json.dumps(predict_image(args.image, load_model(args.model)), indent=2))
