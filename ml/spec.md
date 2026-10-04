# Turmeric Powder Classifier — Spec

## Problem Statement
Build a lab-research prototype to classify turmeric powder photos into two classes: `market_bought` vs `home_ground_dried_roots`. Dataset is 100 images (50 per class) collected under consistent conditions. End users are lab researchers and professors viewing a live demo. Success is ≥85% accuracy on a held-out test set plus a working Streamlit demo that accepts upload + camera input and returns label + confidence. Future variety ID (Erode, Alleppey, etc.) is explicitly out of scope.

## Acceptance Criteria
- WHEN trained and evaluated on a stratified held-out test set never seen in training, THE SYSTEM SHALL achieve accuracy ≥85%.
- WHEN a user uploads JPG/PNG/JPEG or captures via camera in the Streamlit app, THE SYSTEM SHALL return `{label, confidence 0-100%}` in <10s on CPU laptop.
- WHEN input is non-image, corrupt, or unreadable, THE SYSTEM SHALL reject with message "Upload a clear JPG/PNG turmeric powder photo." and SHALL NOT crash.
- WHEN image is blurry/dark or model confidence <60%, THE SYSTEM SHALL show warning + retake guidance ("Use bright diffuse light, fill frame with powder, no shadows/containers, hold steady.") and SHALL NOT present low-confidence output as certain.
- WHEN test photos match training conditions (top-down powder fill, similar lighting), THE SYSTEM SHALL produce stable predictions across 3 repeat uploads.

## Technical Design
Architecture: offline training → frozen artifact → inference-only Streamlit app. No retraining in UI.

```
data/
  market_bought/*.jpg        # ~50 images
  home_ground/*.jpg          # ~50 images
train.py                     # load data, augment, transfer-learn, evaluate, save model
model/
  turmeric_mobilenet.keras   # saved artifact
app.py                       # Streamlit inference only
requirements.txt
README.md
```

Create training pipeline: Load images at 224x224 RGB, normalize to [-1,1] for MobileNetV2. Split stratified 80/20 (80 train, 20 test, 10+10 per class) with `random_state=42`; carve 20% validation from train via `validation_split=0.2`. Apply augmentation: rotation ±20°, horizontal flip, zoom ±0.2, brightness ±0.2, to combat 100-image overfitting. Use MobileNetV2 (ImageNet pretrained, `include_top=False`), add GlobalAveragePooling + Dropout 0.3 + Dense 2 softmax. Freeze base, train head 15 epochs (Adam 1e-3), then unfreeze last 30 layers, fine-tune 15 epochs (Adam 1e-5) with early stopping (patience 5) and ModelCheckpoint on val accuracy.

Create inference flow in `app.py`: Load `.keras` once with `@st.cache_resource`. Preprocess upload/camera frame identically to training. Run `model.predict`, take argmax + max prob*100. Display label + confidence bar. Flag <60% as "Uncertain — retake recommended."

Data contracts:
- Input: image bytes (JPG/PNG), converted to `float32[1,224,224,3]`.
- Output: `{label: "market_bought"|"home_ground", confidence: float}`.
- Model artifact: single Keras file <50MB, CPU-loadable with no GPU requirement.

## Constraints & Guardrails
- Use Python 3.10+, TensorFlow-Keras + MobileNetV2, Streamlit, scikit-learn, Pillow/NumPy. Target local CPU laptop; keep Streamlit Cloud compatible (no local paths, use relative `model/`).
- Do NOT implement variety classification, adulterant percentage, batch CSV, or in-app training.
- Do NOT fetch external data or require internet at inference time after install.
- Do NOT hardcode absolute dataset paths. Do NOT commit large raw zips; expect `data/` locally.
- Do NOT display confidence as fact when <60%; always pair with retake guidance. Do NOT attempt to classify non-powder objects — reject with guidance.
- Keep input permissive (no hard MB cap beyond Streamlit default) but validate type and readability before inference.

## Implementation Plan
1. Create `requirements.txt` to pin `streamlit, tensorflow, scikit-learn, pillow, numpy, matplotlib`.
2. Create `data/` layout to hold the two class folders and add a loader in `train.py` using `image_dataset_from_directory` with 80/20 stratified split.
3. Create augmentation + MobileNetV2 transfer model in `train.py` to train head then fine-tune as specified.
4. Create evaluation in `train.py` to print test accuracy, confusion matrix, and classification report and save artifact to `model/turmeric_mobilenet.keras`.
5. Create `app.py` to implement title, uploader + `st.camera_input`, preview, Predict button, label + confidence display, and low-quality/uncertain warnings with retake tips.
6. Create input guards in `app.py` to reject non-images and check brightness/blur heuristically before prediction.
7. Create `README.md` to document setup (`pip install -r requirements.txt`), training (`python train.py`), run (`streamlit run app.py`), and expected accuracy.

## Definition of Done
- `python train.py` runs end-to-end on the 100 images and saves `model/turmeric_mobilenet.keras` with logged test accuracy ≥85%.
- `streamlit run app.py` launches locally, accepts both upload and camera capture, and returns label + confidence in <10s per image.
- Invalid upload shows error without crash; low-confidence/poor photo shows retake guidance.
- Deliver repo with training script, saved model, app, requirements, and README. No professor report or deployment beyond local demo required.