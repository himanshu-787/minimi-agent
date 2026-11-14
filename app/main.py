import asyncio
import json
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.loop import run_agent
from app.config import settings
from app.memory.store import MemoryStore, format_digest
from app.timeutil import format_ts, now_iso

log = logging.getLogger("minimi")
store = MemoryStore(settings.db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()  # create tables on startup
    yield


app = FastAPI(title="minimi chat agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[dict]
    searches: list[str]
    steps: int


def _to_llm_message(m: dict) -> dict:
    """Stamp user messages with their send time so the model can answer 'when'."""
    if m["role"] == "user":
        return {"role": "user", "content": f"[{format_ts(m['created_at'])}] {m['content']}"}
    return {"role": "assistant", "content": m["content"]}


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
    cid = req.conversation_id or uuid.uuid4().hex
    asked_at = now_iso()

    # sqlite3 is blocking, so run it in a worker thread and keep the event loop free.
    history = await asyncio.to_thread(store.get_history, cid, settings.max_history_messages)
    other_chats = await asyncio.to_thread(
        store.recent_user_questions, settings.digest_size, cid
    )

    messages = [_to_llm_message(m) for m in history]
    messages.append({"role": "user", "content": f"[{format_ts(asked_at)}] {req.message}"})

    try:
        result = await run_agent(messages, memory_digest=format_digest(other_chats))
    except anthropic.APITimeoutError:
        raise HTTPException(504, "The model took too long to respond. Please try again.")
    except anthropic.APIError:
        raise HTTPException(502, "The model service had a problem. Please try again.")

    # Save only after success. If saving fails, the user still gets their answer.
    try:
        await asyncio.to_thread(
            store.save_turn, cid, req.message, asked_at, result.answer,
            {"searches": result.searches, "sources": result.sources},
        )
    except sqlite3.Error:
        log.exception("Could not save turn for conversation %s", cid)

    return ChatResponse(conversation_id=cid, **asdict(result))


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, limit: int = 50):
    rows = await asyncio.to_thread(store.get_history, conversation_id, min(limit, 200))
    if not rows:
        raise HTTPException(404, "Conversation not found.")
    return {"conversation_id": conversation_id, "messages": rows}