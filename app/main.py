import anthropic
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.loop import run_agent
from app.config import settings

app = FastAPI(title="minimi chat agent")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    searches: list[str]
    steps: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": settings.model_name,
        "has_api_key": bool(settings.anthropic_api_key),
        "has_search_key": bool(settings.search_api_key),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await run_agent([{"role": "user", "content": req.message}])
    except anthropic.APITimeoutError:
        raise HTTPException(504, "The model took too long to respond. Please try again.")
    except anthropic.APIError:
        raise HTTPException(502, "The model service had a problem. Please try again.")
    return ChatResponse(**asdict(result))