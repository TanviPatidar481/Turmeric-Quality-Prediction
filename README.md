# Turmeric Impurity Prediction

Phone-photo classifier for **corn-starch adulteration in turmeric powder**: drop a
photo of a powder spread and get back an impurity band — **0%, 10%, 20%, 40% or 50%**.

The model is a **nearest-centroid classifier on paper-white-balanced powder
chromaticity**: 14 parameters, 3,176 bytes, CPU-only, no ML framework at inference.
It white-balances each photo against the white paper in the frame, masks everything
that isn't powder, and reads the colour that is left. No texture, no labels, no paper.

## Benchmark (leave-one-spread-out — whole spreads held out, never random split)

| Level | Score | 95% CI |
|---|---|---|
| Photos (600) | **98.2%** (589/600) | 96.7 – 99.1 |
| Spreads (19) | **100%** (19/19) | 82.4 – 100 |
| Macro F1 | **0.978** | — |

Per-class recall — 0%: 96 · 10%: 94 · 20%: 99 · 40%: 100 · 50%: 100.
All 11 errors sit on the 0%/10% boundary; 20/40/50% were never confused.
See [`DATASET_BENCHMARK.md`](DATASET_BENCHMARK.md) for the full benchmark.

## Quickstart

```bash
# backend (port 8001)
cd backend
cp .env.example .env          # optional; defaults work for local dev
PYTHONPATH=. python3 -m uvicorn app.main:app --port 8001

# frontend (port 5173) — new terminal
cd frontend
npm install
npm run dev -- --port 5173
```

Open http://127.0.0.1:5173 — try the built-in sample photos or drop your own.
A photo must contain **white paper around the spread** (the white-balance reference);
powder-only crops are rejected with retake guidance instead of a guess.

## Retrain (~1 min)

```bash
cd ml
./venv/bin/python train_color_fullframe.py   # LOSO eval + artifact (serving feature)
./venv/bin/python train_color_crops.py       # crop-level reference + ablations
./venv/bin/python make_crops.py              # rebuild the 1,800 analysis crops
```

## Layout

```
ml/
  color_model.py                 inference: powder_rgb -> chromaticity -> centroid
  train_color_fullframe.py       shipped 5-class model (serving feature, LOSO)
  train_color_crops.py           crop-level + ablation reference
  make_crops.py                  1,800 powder-only crops from source photos
  prepare_crop.py                shared segmentation / white-balance helpers
  model/color_centroids_5class_fullframe.json   shipped artifact (3,176 bytes)
  model/fullframe_color_features.csv            600 photo features + labels
  crop_dataset/manifest.csv      crop index with spread ids for group-aware CV
backend/                         FastAPI: /predict /model /samples /health
frontend/                        React + Vite showcase (scores read from the artifact)
docs/                            methodology + diagrams
```

## Data note

Source photos (619 phone HEICs, ~7.4 GB) and generated intermediates
(`tmp_newset_*`, `crop_dataset/batch_*`, legacy sets) are **not in git** — see
`.gitignore`. The repo ships everything needed to run and retrain except the raw
photos: `backend/app/samples/` (10 demo photos), feature CSVs, manifests and the
trained artifact. Point `make_crops.py` / the trainer at your own photo folder to
reproduce.

## Caveats (also shown in the app)

- 19 independent spreads, not 600 — consecutive frames are near-duplicates.
- Only two spreads cover 50%; 0% vs 10% is the one fuzzy boundary.
- The paper must stay in the shot; lighting should be diffuse.
- No confidence percentage is displayed — distance-based scores don't track real
  accuracy, so the UI shows the next-nearest level instead.
