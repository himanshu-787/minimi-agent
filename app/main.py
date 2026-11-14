from fastapi import FastAPI

from app.config import settings

app = FastAPI(title="minimi chat agent")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": settings.model_name,
        "has_api_key": bool(settings.anthropic_api_key),
    }