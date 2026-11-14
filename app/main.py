import asyncio
import json
import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.loop import run_agent, stream_agent
from app.config import settings
from app.memory.store import MemoryStore, format_digest
from app.timeutil import format_ts, now_iso

log = logging.getLogger("minimi")
store = MemoryStore(settings.db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
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
    grounding: str
    latency_ms: int


def _to_llm_message(m: dict) -> dict:
    if m["role"] == "user":
        return {"role": "user", "content": f"[{format_ts(m['created_at'])}] {m['content']}"}
    return {"role": "assistant", "content": m["content"]}


async def _prepare(req: ChatRequest):
    """Shared by both endpoints: load memory and build the model's messages."""
    cid = req.conversation_id or uuid.uuid4().hex
    asked_at = now_iso()
    history = await asyncio.to_thread(store.get_history, cid, settings.max_history_messages)
    other_chats = await asyncio.to_thread(store.recent_user_questions, settings.digest_size, cid)
    messages = [_to_llm_message(m) for m in history]
    messages.append({"role": "user", "content": f"[{format_ts(asked_at)}] {req.message}"})
    return cid, asked_at, messages, format_digest(other_chats)


async def _save(cid: str, message: str, asked_at: str, result) -> None:
    """Save after success. A failed DB write must never block the user's answer."""
    try:
        await asyncio.to_thread(
            store.save_turn, cid, message, asked_at, result.answer,
            {"searches": result.searches, "sources": result.sources, "grounding": result.grounding},
        )
    except sqlite3.Error:
        log.exception("Could not save turn for conversation %s", cid)


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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
    t0 = time.perf_counter()
    cid, asked_at, messages, digest = await _prepare(req)
    try:
        result = await run_agent(messages, memory_digest=digest)
    except anthropic.APITimeoutError:
        raise HTTPException(504, "The model took too long to respond. Please try again.")
    except anthropic.APIError:
        raise HTTPException(502, "The model service had a problem. Please try again.")
    await _save(cid, req.message, asked_at, result)
    return ChatResponse(
        conversation_id=cid, latency_ms=int((time.perf_counter() - t0) * 1000), **asdict(result)
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    cid, asked_at, messages, digest = await _prepare(req)

    async def events():
        t0 = time.perf_counter()
        first_token_ms = None
        yield _sse("start", {"conversation_id": cid})
        try:
            async for ev in stream_agent(messages, memory_digest=digest):
                kind = ev["type"]
                if kind == "token":
                    if first_token_ms is None:
                        first_token_ms = int((time.perf_counter() - t0) * 1000)
                    yield _sse("token", {"text": ev["text"]})
                elif kind == "searching":
                    yield _sse("searching", {"queries": ev["queries"]})
                elif kind == "done":
                    result = ev["result"]
                    await _save(cid, req.message, asked_at, result)
                    yield _sse("done", {
                        "conversation_id": cid,
                        **asdict(result),
                        "latency_ms": int((time.perf_counter() - t0) * 1000),
                        "first_token_ms": first_token_ms,
                    })
        except anthropic.APITimeoutError:
            yield _sse("error", {"message": "The model took too long to respond. Please try again."})
        except anthropic.APIError:
            yield _sse("error", {"message": "The model service had a problem. Please try again."})
        except Exception:
            log.exception("Unexpected error while streaming")
            yield _sse("error", {"message": "Something went wrong. Please try again."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, limit: int = 50):
    rows = await asyncio.to_thread(store.get_history, conversation_id, min(limit, 200))
    if not rows:
        raise HTTPException(404, "Conversation not found.")
    return {"conversation_id": conversation_id, "messages": rows}