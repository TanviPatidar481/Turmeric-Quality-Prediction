import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.routes.meta import router as meta_router  # noqa: E402
from app.routes.predict import router as predict_router  # noqa: E402

app = FastAPI(
    title="Turmeric Impurity Prediction API",
    description=(
        "Classify turmeric powder photos into 0, 10, 20, 40 or 50% corn-starch "
        "impurity using a colour nearest-centroid model validated "
        "leave-one-spread-out."
    ),
    version="3.0.0",
)

origins = [
    o.strip()
    for o in os.environ.get(
        "FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(meta_router)


@app.get("/")
def root():
    return {"message": "Turmeric Quality Prediction API is running"}


@app.get("/health")
def health_check():
    from app.core.model import find_model, get_model

    path = find_model()
    classes: list[str] = []
    if path is not None:
        try:
            classes = get_model()[0]["classes"]
        except Exception:  # noqa: BLE001 - health check must not raise
            classes = []
    return {
        "status": "healthy",
        "model_loaded": path is not None,
        "model_path": str(path) if path else None,
        "classes": classes,
    }
