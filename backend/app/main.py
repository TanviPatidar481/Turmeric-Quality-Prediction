from fastapi import FastAPI

app = FastAPI(
    title="Turmeric Quality Prediction API",
    description="Backend API for the Turmeric Quality Prediction project",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Turmeric Quality Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}