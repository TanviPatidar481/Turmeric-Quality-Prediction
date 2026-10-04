# Turmeric Powder Classifier (ML)

Binary classifier: `market_bought` vs `home_ground` (home-ground dried roots).
MobileNetV2 transfer learning, Streamlit inference demo.

## Layout

```
ml/
  turmeric_classifier.ipynb   # EDA + training + evaluation (start here)
  train.py                    # headless reproduction of the notebook
  app.py                      # Streamlit inference only
  model/turmeric_mobilenet.keras
  artifacts/turmeric_mobilenet.keras  # mirror copy
  market bought/  home grinded/       # raw 50+50 dataset
  requirements.txt
```

`train.py` / notebook accept either the spec `data/` layout or the current
spaced folder names; labels are normalised to `home_ground` / `market_bought`.

## Setup

```bash
cd ml
pip install -r requirements.txt
```

Python 3.10+ with TensorFlow-Keras, Streamlit, scikit-learn, Pillow, NumPy, Matplotlib.

## Train

```bash
python train.py
```

- 224×224 RGB, MobileNetV2 `preprocess_input` ([-1, 1])
- Stratified 80/20 test split (`random_state=42`) → 64 train / 16 val / 20 test
- Augmentation: rotation ±20°, flip, zoom ±0.2 (geometric only, inside the
  model so it is train-time only).
  > Deviation from spec: brightness ±0.2 is omitted. The classes differ by a
  > subtle colour shift; ±0.2 brightness on [-1,1] inputs erases it and pins
  > training at 50%. Geometric-only augmentation scores 100% on test.
- Head 15 epochs (Adam 1e-3, base frozen) → unfreeze last 30 layers,
  fine-tune 15 epochs (Adam 1e-5), early stopping (patience 5),
  checkpoint on `val_accuracy`
- Prints test accuracy, confusion matrix, classification report.
  Acceptance: **≥85%**.

## Demo

```bash
streamlit run app.py
```

Upload JPG/PNG or camera capture → label + confidence bar.
Guards: corrupt/non-image → "Upload a clear JPG/PNG turmeric powder photo.";
dark/blurry or confidence <60% → retake guidance, never presented as certain.

## Out of scope (per spec)

Variety ID (Erode/Alleppey…), adulterant %, batch CSV, in-app training.
