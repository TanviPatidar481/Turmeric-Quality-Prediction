"""GET /model, /samples — the showcase page's read-only endpoints.

`/model` returns the model card (scores, per-class recall, caveats) straight
from the artifact, so the page can never drift from the trained model.
`/samples` serves a few real photos so a visitor can try the classifier
without having their own picture to hand.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.model import get_model, model_card

router = APIRouter()

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"

SAMPLES = [
    {"id": "0_percent_a", "label": "0_percent", "pretty": "0% impurity",
     "note": "pure turmeric"},
    {"id": "0_percent_b", "label": "0_percent", "pretty": "0% impurity",
     "note": "pure turmeric"},
    {"id": "10_percent_a", "label": "10_percent", "pretty": "10% impurity",
     "note": "light starch"},
    {"id": "10_percent_b", "label": "10_percent", "pretty": "10% impurity",
     "note": "light starch"},
    {"id": "20_percent_a", "label": "20_percent", "pretty": "20% impurity",
     "note": "clearly lighter"},
    {"id": "20_percent_b", "label": "20_percent", "pretty": "20% impurity",
     "note": "clearly lighter"},
    {"id": "40_percent_a", "label": "40_percent", "pretty": "40% impurity",
     "note": "heavy starch, jar A"},
    {"id": "40_percent_b", "label": "40_percent", "pretty": "40% impurity",
     "note": "heavy starch, jar B"},
    {"id": "50_percent_a", "label": "50_percent", "pretty": "50% impurity",
     "note": "half starch"},
    {"id": "50_percent_b", "label": "50_percent", "pretty": "50% impurity",
     "note": "half starch"},
]

_BY_ID = {s["id"]: s for s in SAMPLES}


@router.get("/model")
def model_info():
    card = model_card()
    card["samples"] = [{k: s[k] for k in ("id", "label", "pretty", "note")}
                       for s in SAMPLES]
    return card


@router.get("/samples")
def list_samples():
    return SAMPLES


@router.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    if sample_id not in _BY_ID:
        raise HTTPException(status_code=404, detail="unknown sample")
    path = SAMPLES_DIR / f"{sample_id}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="sample image missing")
    return FileResponse(path, media_type="image/jpeg")
