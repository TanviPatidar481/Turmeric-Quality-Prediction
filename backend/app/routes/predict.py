"""POST /predict — 5-class turmeric impurity classification."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.model import (
    INVALID_MSG,
    RETAKE_TIPS,
    decode_image,
    predict_image,
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

    try:
        result = predict_image(image)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{exc}. {RETAKE_TIPS}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    warnings = quality_warnings(image)
    if result["ood"]:
        warnings.append(
            result.get("warning")
            or "Outside the trained colour range; treat this prediction as unreliable."
        )

    return {
        "label": result["label"],
        "pretty": result["pretty"],
        "distance": result["distance"],
        "margin": result["margin"],
        "distances": result["distances"],
        "ood": result["ood"],
        "features": result["features"],
        "summary": result["summary"],
        "warnings": warnings,
    }
