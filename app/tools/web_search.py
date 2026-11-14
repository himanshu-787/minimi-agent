import httpx

from app.config import settings

TAVILY_URL = "https://api.tavily.com/search"

# This schema is what the LLM reads to decide WHEN and HOW to call the tool.
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current or factual information. Use it for recent "
        "events, prices, news, anything time-sensitive, or any fact you are not "
        "sure about. Do not use it for greetings, math, or things you know well."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A short, specific search query (3-8 words works best).",
            }
        },
        "required": ["query"],
    },
}


async def web_search(query: str) -> dict:
    """Run a search. NEVER raises: failures come back as {"error": ...}
    so the model can react to them instead of the whole request crashing."""
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
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
            data = resp.json()
    except httpx.TimeoutException:
        return {"error": "The search timed out. Try a simpler query or answer with what you know, and say so."}
    except httpx.HTTPError as e:
        return {"error": f"The search failed ({type(e).__name__}). Tell the user the web is unavailable."}

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:600],  # trim: less text = faster LLM
        }
        for r in data.get("results", [])
    ]
    return {"query": query, "results": results}