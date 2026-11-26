# Turmeric Impurity Prediction

Phone-photo classifier for **corn-starch adulteration in turmeric powder**:
drop a photo of a powder spread and get back an impurity band —
**0%, 10%, 20%, 40% or 50%**.

The model is a **nearest-centroid classifier on paper-white-balanced powder
chromaticity**: 14 parameters, 3,176 bytes, CPU-only, no ML framework at inference.
It white-balances each photo against the white paper in the frame, masks everything
that isn't powder, and reads the colour that is left.

## Benchmark (leave-one-spread-out — whole spreads held out, never random split)

- **Photos: 98.2%** (589/600) · **Spreads: 19/19** · **Macro F1: 0.978**
- Per-class recall — 0%: 96 · 10%: 94 · 20%: 99 · 40%: 100 · 50%: 100
- All errors sit on the 0%/10% boundary; 20/40/50% were never confused.

## Run the demo

```bash
pip install -r ml/requirements.txt
cd ml
streamlit run app.py        # -> http://localhost:8501
```

A photo must contain **white paper around the spread** (the white-balance
reference). Powder-only crops are rejected with retake guidance instead of a guess.
No confidence percentage is shown — distance-based scores don't track real
accuracy, so the app shows the next-nearest level instead.

## Layout

```
ml/
  app.py                  Streamlit demo (this page)
  color_model.py          inference: powder_rgb -> chromaticity -> centroid
  prepare_crop.py         shared segmentation / white-balance helpers
  model/color_centroids_5class_fullframe.json   shipped model (3,176 bytes)
```

Raw photos, training scripts and analysis files live on disk alongside this repo
but are intentionally not tracked — see `.gitignore`.
