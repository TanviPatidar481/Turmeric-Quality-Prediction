"""POST /predict — turmeric powder classification."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.model import (
    INVALID_MSG,
    RETAKE_TIPS,
    decode_image,
    predict_array,
    preprocess,
    quality_warnings,
)

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTS = (".jpg", ".jpeg", ".png")


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if file.content_type not in ALLOWED_TYPES and not name.endswith(ALLOWED_EXTS):
        raise HTTPException(status_code=400, detail=INVALID_MSG)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=INVALID_MSG)

    image = decode_image(raw)
    if image is None:
        raise HTTPException(status_code=400, detail=INVALID_MSG)

    warnings = quality_warnings(image)
    try:
        label, confidence, probs = predict_array(preprocess(image))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if confidence < 60:
        warnings.append(f"Uncertain — retake recommended. {RETAKE_TIPS}")

    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
        "warnings": warnings,
    }
