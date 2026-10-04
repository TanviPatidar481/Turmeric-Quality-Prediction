"""Turmeric Powder Classifier — Streamlit inference app (inference only, no retraining).

Run:  streamlit run app.py   (from the ml/ directory)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
CANDIDATE_MODELS = [
    HERE / "model" / "turmeric_mobilenet.keras",
    HERE / "artifacts" / "turmeric_mobilenet.keras",
]
CLASS_NAMES = ["home_ground", "market_bought"]
PRETTY = {"home_ground": "Home-ground (dried roots)",
          "market_bought": "Market-bought"}
IMG_SIZE = (224, 224)
RETAKE_TIPS = ("Use bright diffuse light, fill frame with powder, "
               "no shadows/containers, hold steady.")
ERROR_MSG = "Upload a clear JPG/PNG turmeric powder photo."


def find_model() -> Path | None:
    for p in CANDIDATE_MODELS:
        if p.is_file():
            return p
    return None


@st.cache_resource
def load_model():
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # noqa: F401
    path = find_model()
    if path is None:
        return None, None
    model = tf.keras.models.load_model(path)
    return model, path


def preprocess(pil_img: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype(np.float32)
    arr = preprocess_input(arr)  # [-1, 1], identical to training
    return np.expand_dims(arr, 0)  # float32[1,224,224,3]


def quality_checks(pil_img: Image.Image) -> list[str]:
    """Heuristic blur / darkness warnings (guidance only, not a hard reject)."""
    warnings: list[str] = []
    gray = np.asarray(pil_img.convert("RGB").resize((256, 256))).astype(np.float32)
    brightness = gray.mean()  # 0-255
    if brightness < 60:
        warnings.append(f"Image looks dark (brightness {brightness:.0f}/255).")
    # Laplacian variance blur metric via manual kernel (no hard cv2 dependency).
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
st.set_page_config(page_title="Turmeric Powder Classifier", layout="centered")
st.title("Turmeric Powder Classifier")
st.caption("Prototype: `market_bought` vs `home_ground` · MobileNetV2 · CPU inference")

model, model_path = load_model()
if model is None:
    st.error("Model artifact not found. Train first: `python train.py` "
             "(expects `model/turmeric_mobilenet.keras`).")
    st.stop()

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
    x = preprocess(image)
    with st.spinner("Classifying…"):
        probs = np.asarray(model.predict(x, verbose=0))[0]
    idx = int(probs.argmax())
    conf = float(probs[idx]) * 100
    label = CLASS_NAMES[idx]
    st.subheader(f"{PRETTY[label]}")
    st.progress(min(max(conf / 100, 0.0), 1.0), text=f"Confidence: {conf:.1f}%")
    st.write({CLASS_NAMES[i]: f"{float(p) * 100:.1f}%" for i, p in enumerate(probs)})
    if warnings:
        st.warning("Photo quality is low — " + RETAKE_TIPS)
    if conf < 60:
        st.warning("Uncertain — retake recommended. " + RETAKE_TIPS)
