import asyncio
import time

import httpx

from app.config import settings

TAVILY_URL = "https://api.tavily.com/search"

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current or factual information. Use it for recent "
            "events, prices, news, anything time-sensitive, or any fact you are not "
            "sure about. Do not use it for greetings, math, or things you know well."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short, specific search query (3-8 words works best).",
                }
            },
            "required": ["query"],
        },
    },
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# ---- tiny in-memory TTL cache -------------------------------------------
_cache: dict[str, tuple[float, dict]] = {}


def _key(query: str) -> str:
    return " ".join(query.lower().split())  # "Python  Release" == "python release"


def _cache_get(key: str) -> dict | None:
    hit = _cache.get(key)
    if hit is None:
        return None
    expires_at, value = hit
    if time.monotonic() > expires_at:
        _cache.pop(key, None)
        return None
    return value


def _cache_put(key: str, value: dict) -> None:
    if len(_cache) >= settings.search_cache_max:
        oldest = min(_cache, key=lambda k: _cache[k][0])  # evict the soonest-to-expire
        _cache.pop(oldest, None)
    _cache[key] = (time.monotonic() + settings.search_cache_ttl_s, value)


def clear_cache() -> None:
    _cache.clear()


# ---- network ---------------------------------------------------------------
async def _fetch(query: str) -> dict:
    async with httpx.AsyncClient(timeout=settings.search_timeout_s) as client:
        resp = await client.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {settings.search_api_key}"},
            json={
                "query": query,
                "max_results": settings.search_max_results,
                "search_depth": "basic",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def web_search(query: str) -> dict:
    """Cached, retried search. NEVER raises: failures come back as {"error": ...}."""
    key = _key(query)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    failure = ""
    for attempt in range(settings.search_retries + 1):
        try:
            data = await _fetch(query)
            break
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code not in RETRYABLE_STATUS:
                # 401/403/400: retrying cannot help (wrong key, bad request).
                return {"error": f"The search service rejected the request (HTTP {code}). Tell the user search is unavailable."}
            failure = f"HTTP {code}"
        except httpx.TimeoutException:
            failure = "timeout"
        except httpx.HTTPError as e:
            failure = type(e).__name__

        if attempt < settings.search_retries:
            await asyncio.sleep(settings.search_backoff_s * (2 ** attempt))
    else:
        return {"error": f"The search failed after {settings.search_retries + 1} attempts ({failure}). Tell the user the web is unavailable."}

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:600],
        }
        for r in data.get("results", [])
    ]
    result = {"query": query, "results": results}
    _cache_put(key, result)  # only successes are cached
    return result
