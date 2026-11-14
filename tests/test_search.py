import asyncio

import httpx
import pytest

from app.config import settings
from app.tools import web_search as ws

RAW = {"results": [{"title": "T", "url": "http://x.com", "content": "c"}]}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    ws.clear_cache()
    monkeypatch.setattr(settings, "search_backoff_s", 0)   # no real waiting in tests
    yield
    ws.clear_cache()


def status_error(code):
    req = httpx.Request("POST", ws.TAVILY_URL)
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, request=req))


def test_cache_hit_skips_network(monkeypatch):
    calls = []

    async def fake_fetch(q):
        calls.append(q)
        return RAW

    monkeypatch.setattr(ws, "_fetch", fake_fetch)
    a = asyncio.run(ws.web_search("Python  release"))
    b = asyncio.run(ws.web_search("python release"))   # same query after normalising
    assert a == b and len(calls) == 1


def test_retries_then_succeeds(monkeypatch):
    attempts = {"n": 0}

    async def flaky(q):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise status_error(503)
        return RAW

    monkeypatch.setattr(ws, "_fetch", flaky)
    result = asyncio.run(ws.web_search("flaky"))
    assert "error" not in result and attempts["n"] == 3


def test_bad_key_is_not_retried(monkeypatch):
    attempts = {"n": 0}

    async def unauthorized(q):
        attempts["n"] += 1
        raise status_error(401)

    monkeypatch.setattr(ws, "_fetch", unauthorized)
    result = asyncio.run(ws.web_search("anything"))
    assert "error" in result and attempts["n"] == 1


def test_failures_are_not_cached(monkeypatch):
    async def down(q):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(ws, "_fetch", down)
    assert "error" in asyncio.run(ws.web_search("same query"))

    async def up(q):
        return RAW

    monkeypatch.setattr(ws, "_fetch", up)
    assert "error" not in asyncio.run(ws.web_search("same query"))