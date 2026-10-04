import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.routes.predict import router as predict_router  # noqa: E402

app = FastAPI(
    title="Turmeric Quality Prediction API",
    description="Classify turmeric powder photos: market_bought vs home_ground",
    version="1.0.0",
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


@app.get("/")
def root():
    return {"message": "Turmeric Quality Prediction API is running"}


@app.get("/health")
def health_check():
    from app.core.model import find_model

    path = find_model()
    return {
        "status": "healthy",
        "model_loaded": path is not None,
        "model_path": str(path) if path else None,
    }
