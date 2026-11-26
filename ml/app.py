"""Turmeric impurity classifier — Streamlit inference demo (600-image model).

Run:  streamlit run app.py   (from the ml/ directory)

Serves the shipped 5-class colour nearest-centroid model
(model/color_centroids_5class_fullframe.json): paper-white-balanced powder
chromaticity, 98.2% of 600 photos correct with 19/19 spreads correct under
leave-one-spread-out. No confidence percentage is shown — the distance-based
score does not track real accuracy — the next-nearest level is shown instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from color_model import (
    load_model as read_model,
    predict_features,
    powder_rgb,
    to_model_features,
)

HERE = Path(__file__).resolve().parent
CANDIDATE_MODELS = [
    HERE / "model" / "color_centroids_5class_fullframe.json",
    HERE / "model" / "color_centroids_5class.json",
]
RETAKE_TIPS = (
    "Use bright diffuse light, fill frame with powder and keep some white "
    "paper visible, no shadows/containers, hold steady."
)
ERROR_MSG = "Upload a clear JPG/PNG turmeric powder photo."


def find_model() -> Path | None:
    for p in CANDIDATE_MODELS:
        if p.is_file():
            return p
    return None


@st.cache_resource
def load_model():
    path = find_model()
    if path is None:
        return None, None
    return read_model(path), path


def classify(image: Image.Image, model: dict) -> dict:
    rgb_mean = powder_rgb(image)
    feats = to_model_features(rgb_mean, model)
    return predict_features(feats, model)


def quality_checks(pil_img: Image.Image) -> list[str]:
    """Heuristic blur / darkness warnings (guidance only, not a hard reject)."""
    warnings: list[str] = []
    gray = np.asarray(pil_img.convert("RGB").resize((256, 256))).astype(np.float32)
    brightness = gray.mean()  # 0-255
    if brightness < 60:
        warnings.append(f"Image looks dark (brightness {brightness:.0f}/255).")
    g = np.asarray(pil_img.convert("L").resize((256, 256))).astype(np.float32)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    try:
        from scipy.signal import convolve2d  # type: ignore
        lap = convolve2d(g, kernel, mode="valid")
    except Exception:
        lap = (np.roll(g, 1, 0) + np.roll(g, -1, 0)
               + np.roll(g, 1, 1) + np.roll(g, -1, 1) - 4 * g)[1:-1, 1:-1]
    if float(lap.var()) < 60:
        warnings.append("Image looks blurry (low sharpness).")
    return warnings


def read_upload(file) -> Image.Image | None:
    try:
        img = Image.open(file)
        img.load()  # force decode -> raises on corrupt files
        img = img.convert("RGB")
        return img
    except Exception:
        return None


# ------------------------------------------------------------------ UI
st.set_page_config(page_title="Turmeric Impurity Classifier", layout="centered")
st.title("Turmeric Impurity Classifier")
st.caption(
    "5 levels: 0, 10, 20, 40, 50% corn starch · paper-white-balanced "
    "powder chromaticity · 98.2% of photos, 19/19 spreads (leave-one-spread-out)"
)

model, model_path = load_model()
if model is None:
    st.error("Model artifact not found. Train first: `python train_color_fullframe.py` "
             "(expects `model/color_centroids_5class_fullframe.json`).")
    st.stop()

classes = model["classes"]
pretty = model["pretty"]
with st.expander("Model card", expanded=False):
    st.write(f"**Features:** {', '.join(model['feature_names']).upper()}")
    st.write(f"**Images:** {model.get('n_photos', model.get('n_images'))} · **spreads:** {model['n_spreads']}")
    st.json(model["cv"])

uploaded = st.file_uploader("Upload a turmeric powder photo",
                            type=["jpg", "jpeg", "png"])
camera = st.camera_input("Or capture with camera")

source = uploaded if uploaded is not None else camera
if source is None:
    st.info("Upload a JPG/PNG or capture a photo to classify.")
    st.stop()

image = read_upload(source)
if image is None:
    st.error(ERROR_MSG)
    st.stop()

st.image(image, caption="Input preview", use_container_width=True)
warnings = quality_checks(image)
for w in warnings:
    st.warning(w + " " + RETAKE_TIPS)

if st.button("Predict", type="primary"):
    try:
        with st.spinner("Classifying…"):
            result = classify(image, model)
    except ValueError as exc:
        st.error(f"{exc}. {RETAKE_TIPS}")
        st.stop()

    st.subheader(result["pretty"])

    ranked = sorted(result["distances"].items(), key=lambda kv: kv[1])
    if len(ranked) > 1:
        st.caption(
            f"Next nearest level: {model['pretty'][ranked[1][0]]} "
            f"(margin {result['margin']:.3f})"
        )

    st.caption(
        f"Nearest-centroid distance: {result['distance']:.2f} "
        f"(typical within-class {model['distance_scale']:.2f}; "
        f"out-of-range above {model['ood_threshold']:.2f})"
    )

    if result["ood"]:
        st.warning(result["warning"])

    if warnings:
        st.warning("Photo quality is low — " + RETAKE_TIPS)
