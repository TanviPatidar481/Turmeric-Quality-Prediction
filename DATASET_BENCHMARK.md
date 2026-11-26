# Turmeric Impurity Dataset — Benchmark

5-class corn-starch impurity classification (0 / 10 / 20 / 40 / 50%) from phone
photos of powder spreads on white paper. 30% is folded into 40%: the two source
batches were found swapped and only ~0.0024 apart in colour, so they ship as one class.

## Data

- 600 source photos (100 per source folder), 19 independent capture spreads
  (0%: 5, 10%: 5, 20%: 3, 40%: 4 incl. both jars, 50%: 2)
- 1,800 powder-only analysis crops (384 px, 3 sections/photo, >=97% powder)
- Features: paper-white-balanced powder mean colour -> chromaticity r,g
- Model: nearest centroid, 14 parameters, 3,176 bytes, CPU-only, no ML framework

## Benchmark (leave-one-spread-out — whole spreads held out, never random split)

| Level | Score | 95% CI |
|---|---|---|
| Photos (600) | **98.2%** (589/600) | 96.7 – 99.1 |
| Spreads (19) | **100%** (19/19) | 82.4 – 100 |
| Macro F1 (photo) | **0.978** | — |

Per-class recall: 0%: 96 · 10%: 94 · 20%: 99 · 40%: 100 · 50%: 100

Confusion matrix (rows true, cols pred, order 0/10/20/40/50):

```
96   4   0   0   0
 6  94   0   0   0
 0   1  99   0   0
 0   0   0 200   0
 0   0   0   0 100
```

All 11 errors sit on the 0%/10% boundary. 20/40/50% were never confused.

## Reference points (same protocol)

- Crop-level features: 92.7% crops, 19/19 spreads, F1 0.917
- 6-class with 30/40 split: 94.2% photos, 19/19 spreads, F1 0.941
- Feature ablation (crop LOSO): raw RGB 53.6% · G/B 57.6% · chromaticity 92.7%

## Files

- `ml/model/color_centroids_5class_fullframe.json` — shipped artifact
- `ml/model/fullframe_color_features.csv` — 600 photo features + labels
- `ml/crop_dataset/` — 1,800 crops in 3 batches of 600 + `manifest.csv` (spread ids)
- Rebuild: `cd ml && ./venv/bin/python train_color_fullframe.py` (~35 s)
