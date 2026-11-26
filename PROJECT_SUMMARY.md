# Project Summary — Turmeric Quality Prediction

> Handoff document. Written so another AI model or developer can continue without
> re-deriving decisions. Last updated: 2026-10-29.

---

## 1. What this project is

Binary image classifier for turmeric powder photos:

| Class | Meaning |
|---|---|
| `market_bought` | Powder bought at market |
| `home_ground` | Powder ground at home from dried roots |

Primary user: lab researchers / professors watching a live demo.
**Spec source of truth:** `ml/spec.md`.

Out of scope (explicitly): turmeric variety ID (Erode, Alleppey…), adulterant
percentage, batch CSV, in-app training, deployment beyond a local demo.

Acceptance criteria from spec:

- ≥85% accuracy on a stratified held-out test set
- Streamlit app accepts upload + camera, returns `{label, confidence}` in <10s on CPU
- Non-image / corrupt input → `"Upload a clear JPG/PNG turmeric powder photo."`, no crash
- Blurry/dark image or confidence <60% → retake guidance, never presented as certain
- Stable predictions across 3 repeat uploads

**Status: all acceptance criteria met.** Verified test accuracy 95.0%.

---

## 2. Architecture

Three independent pieces (spec only required the first two):

```
ml/            training + Streamlit demo (spec deliverable)
backend/       FastAPI REST wrapper around the same model (extension)
frontend/      React + Vite demo UI (extension)
```

The frozen `.keras` artifact is the single source of truth for inference.
Both `ml/app.py` (Streamlit) and `backend/app/core/model.py` load that same file.

```
image bytes
   → reject if not JPG/PNG or not decodable
   → RGB, resize 224×224
   → MobileNetV2 preprocess_input  (pixel/127.5 - 1 → range [-1, 1])
   → float32[1,224,224,3]
   → model.predict → softmax[2]
   → argmax + max*100
```

---

## 3. Dataset

- Location: `ml/market bought/` (50 JPGs) and `ml/home grinded/` (50 JPGs)
- Note the **spaces** in the folder names — always quote paths in shell commands
- All ~2400×2400 RGB JPGs, cropped, top-down powder fill, consistent lighting
- `ml/train.py` auto-detects layout: it also accepts the spec layout
  `ml/data/market_bought/` + `ml/data/home_ground/`, plus `home_grinded`
  and `home_ground_dried_roots` variants (first match wins)

Label order used everywhere (do not change):

```python
CLASS_NAMES = ["home_ground", "market_bought"]   # index 0 = home, 1 = market
```

Splits: stratified, `random_state=42`, 80/20 test, then 20% of the remainder
for validation → **64 train / 16 val / 20 test** (10+10 per class in test).

---

## 4. Model

- Base: MobileNetV2, ImageNet weights, `include_top=False`, input 224×224×3, frozen
- Head: `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(2, softmax)`
- Stage 1: head only, Adam `1e-3`, up to 15 epochs
- Stage 2: unfreeze **last 30** base layers, Adam `1e-5`, up to 15 epochs
- Callbacks: `EarlyStopping(val_accuracy, patience=5, restore_best_weights)`,
  `ModelCheckpoint(model/best_tmp.keras, monitor=val_accuracy, save_best_only)`
- Augmentation lives **inside the model** (train-time only, pass-through at inference)

### Critical deviation from spec: no brightness augmentation

The spec asks for `brightness ±0.2`. **This was removed on purpose.**

Reasoning, verified experimentally:

1. First run with brightness jitter hit exactly **50%** test accuracy (collapse to one class).
2. Data is separable: mean-RGB logistic regression → 100% CV; frozen MobileNetV2
   features + logistic regression → 100% CV.
3. Cause: the classes differ by a **subtle few-pixel mean-colour shift**
   (market ≈ RGB 163/120/80, home ≈ 160/125/90). On [-1,1] inputs, `±0.2`
   brightness is ≈ ±25 pixel levels — it destroys the only discriminative cue.
   Validation/test images are unaugmented, so train and val distributions diverge.
4. Geometric-only augmentation (flip, rotation ±20°, zoom ±0.2) preserves colour
   → 100% validation, 95% test on the spec split.

Rotation/flip/zoom **are** kept. If anyone re-adds brightness jitter, accuracy
returns to chance. This is documented in `ml/train.py`, `ml/README.md`, and
notebook §4.

### Results (last verified run, spec split)

```
test accuracy: 95.00%          PASS (>=85%)
confusion matrix: [[10, 0],
                   [ 1, 9]]    rows=true home/market, cols=pred home/market
classification report: macro F1 0.9499
```

Artifact: `ml/model/turmeric_mobilenet.keras`, ~21.9 MB (<50 MB budget).
Mirrored to `ml/artifacts/turmeric_mobilenet.keras` (that directory is gitignored).

Inference timing on this Mac (CPU): ~7s for the very first call (TF graph
warmup), then <1s per image. Both within the 10s requirement.

---

## 5. Files

| Path | Purpose |
|---|---|
| `ml/spec.md` | Original spec / acceptance criteria |
| `ml/train.py` | Headless training: load → split → augment → 2-stage train → eval → save |
| `ml/turmeric_classifier.ipynb` | Notebook version (EDA, class balance, mean RGB, augmentation preview, curves, confusion matrix, stability check) |
| `ml/app.py` | Streamlit inference demo (upload + camera), input guards, low-confidence warnings |
| `ml/model/turmeric_mobilenet.keras` | Trained artifact (tracked in git) |
| `ml/artifacts/turmeric_mobilenet.keras` | Mirror copy (gitignored) |
| `ml/model/best_tmp.keras` | Checkpoint left by training; safe to delete |
| `ml/requirements.txt` | streamlit, tensorflow, scikit-learn, pillow, numpy, matplotlib |
| `ml/README.md` | ML setup/train/run docs |
| `backend/app/main.py` | FastAPI app, CORS, `/`, `/health` |
| `backend/app/routes/predict.py` | `POST /predict` endpoint |
| `backend/app/core/model.py` | Model singleton, preprocessing, quality heuristics |
| `backend/requirements.txt` | FastAPI stack + tensorflow/pillow/numpy (was UTF-16, now UTF-8) |
| `backend/.env.example` | `TURMERIC_MODEL_PATH`, `FRONTEND_ORIGIN` |
| `frontend/src/App.jsx` | Demo UI: upload, camera, preview, predict, result |
| `frontend/src/services/api.js` | `predictImage(blob)`, label prettifier, retake tips |
| `frontend/src/App.css` | Demo styling, dark-mode aware |
| `frontend/.env.example` | `VITE_API_BASE_URL` |

---

## 6. API contract (backend)

`POST /predict` — multipart/form-data, field name `file`.

Success (200):

```json
{
  "label": "market_bought",
  "confidence": 87.6,
  "probs": { "home_ground": 12.4, "market_bought": 87.6 },
  "warnings": []
}
```

- `label` ∈ {`home_ground`, `market_bought`}; `confidence` is 0–100 (percent)
- `warnings` contains dark/blur/low-confidence retake guidance sentences

Errors:

- 400 `{"detail": "Upload a clear JPG/PNG turmeric powder photo."}`
  for wrong content type, empty body, or undecodable/corrupt bytes
- 503 if the model artifact cannot be found

`GET /health` returns `{status, model_loaded, model_path}`.

CORS: allowed origins from `FRONTEND_ORIGIN`, default
`http://localhost:5173,http://127.0.0.1:5173`.

Model lookup order: `$TURMERIC_MODEL_PATH` → `ml/model/…` → `ml/artifacts/…`.

---

## 7. Frontend contract

- `VITE_API_BASE_URL` (defaults to `http://localhost:8000`); set it to whatever
  port the API actually runs on
- Client rejects non-JPG/PNG before upload with the spec error message
- Confidence <60% → "Uncertain — retake recommended." + retake tips
- Camera via `getUserMedia({video: {facingMode: "environment"}})`, canvas capture
- Camera permission denied → falls back to upload with a message

---

## 8. How to run (macOS, current machine)

Use **Python 3.12**, not Homebrew's `python3` (that is 3.14 and has no TF wheels).

```bash
cd ml
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train.py                 # ~5–15 min CPU, prints 95% + saves artifact
streamlit run app.py            # http://localhost:8501
```

Backend:

```bash
cd backend
source venv/bin/activate        # a venv with the backend requirements
uvicorn app.main:app --port 8001
```

> Port **8000 is occupied** on this machine by an unrelated Django project.
> Use 8001 (or set `FRONTEND_ORIGIN`/`VITE_API_BASE_URL` accordingly).

Frontend:

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8001 npm run dev   # http://localhost:5173
```

### Windows

```bat
cd ml
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Install 64-bit Python 3.11/3.12 from python.org with "Add to PATH".
Keep the repo on a short path; TF dislikes long Windows paths.

---

## 9. Environment pitfalls (learned the hard way)

1. **Two venvs exist in this repo and only one works.**
   - `./venv/` at repo root → Python **3.14.7**, no tensorflow, no streamlit (broken)
   - `ml/venv/` → Python **3.12.1**, TF 2.21.0, streamlit 1.66.1 (working)
   Activate the right one, or `python train.py` fails with
   `ModuleNotFoundError: No module named 'tensorflow'` and
   `streamlit run app.py` fails with `command not found`.
2. **TensorFlow version drift.** The model was originally trained with TF 2.19;
   the current `ml/venv` has 2.21. The artifact loads fine under 2.21 (verified),
   but Keras 3 `.keras` files are version-sensitive — if loading ever breaks,
   retrain rather than install mismatched versions.
3. **Folder names contain spaces** (`market bought`, `home grinded`). Quote them.
4. `ml/artifacts/` is gitignored but `ml/model/` is tracked, so a fresh clone has
   the model — no retrain needed to run the app.
5. `backend/requirements.txt` used to be **UTF-16** encoded; it is now UTF-8.
   If it ever looks garbled, that is why.

---

## 10. Verified evidence

| Check | Result |
|---|---|
| `python train.py` end-to-end | 95.0% test accuracy, PASS |
| Model artifact size | 21.9 MB, <50 MB |
| Model load + predict (fresh process) | OK under TF 2.21 |
| 3× repeat prediction on same image | identical label + confidence |
| Streamlit app script execution | 0 exceptions (Streamlit `AppTest`) |
| `POST /predict` market sample | `market_bought` 87.6% |
| `POST /predict` home sample | `home_ground` 63.7% |
| Text file uploaded as image | 400 + spec message |
| Random bytes named `.jpg` | 400 + spec message |
| CORS preflight from :5173 | 200 |
| `npm run build` / `npm run lint` | pass, 0 errors |

---

## 11. Suggested next steps

1. Root `README.md` tying together ml / backend / frontend run commands.
2. Automated tests: `pytest` for `/predict` guards, a scripted accuracy regression check.
3. Delete `ml/model/best_tmp.keras` after training (or have `train.py` clean it up).
4. Consider `ml/model/` in git LFS if the artifact keeps being committed and changed.
5. Optional: Docker Compose for the three services.
