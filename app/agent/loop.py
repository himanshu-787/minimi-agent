import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from app.agent.prompts import build_system_prompt
from app.config import settings
from app.tools.web_search import WEB_SEARCH_TOOL, web_search

# 1-2 digits only, so a year like [2024] is never mistaken for a citation.
CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass
class AgentResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    steps: int = 0
    grounding: str = "not_needed"   # not_needed | cited | uncited | search_failed


def _text(content) -> str:
    return "".join(b.text for b in content if b.type == "text").strip()


async def _run_tool(block, search_fn) -> dict:
    if block.name != "web_search":
        return {"error": f"Unknown tool: {block.name}"}
    query = ((block.input or {}).get("query") or "").strip()
    if not query:
        return {"error": "Empty search query."}
    return await search_fn(query)


def finalize(answer: str, sources: dict[str, dict], searches: list[str]):
    """Verify citations in code. Returns (clean_answer, sources_to_show, grounding)."""
    by_id = {s["id"]: s for s in sources.values()}
    cited: list[int] = []
    dropped = False

    def keep(m: re.Match) -> str:
        nonlocal dropped
        n = int(m.group(1))
        if n in by_id:
            if n not in cited:
                cited.append(n)
            return m.group(0)
        dropped = True          # the model cited a source that does not exist
        return ""

    clean = CITATION.sub(keep, answer)
    if dropped:
        clean = re.sub(r"(?<=\S) {2,}", " ", clean)
        clean = re.sub(r"(?<=\S) +([.,;])", r"\1", clean)

    if not searches:
        grounding = "not_needed"
    elif not sources:
        grounding = "search_failed"
    elif not cited:
        grounding = "uncited"
    else:
        grounding = "cited"

    shown = [by_id[n] for n in sorted(cited)] if cited else list(by_id.values())
    return clean, shown, grounding


async def _llm_turn(client, system, messages, forced: bool) -> AsyncIterator[dict]:
    """One streamed model call. Yields token events, then the final message."""
    kwargs = dict(
        model=settings.model_name,
        max_tokens=settings.max_output_tokens,
        system=system,
        tools=[WEB_SEARCH_TOOL],
        messages=messages,
    )
    if forced:
        kwargs["tool_choice"] = {"type": "none"}   # out of budget: answer now
    async with client.messages.stream(**kwargs) as stream:
        async for text in stream.text_stream:
            yield {"type": "token", "text": text}
        final = await stream.get_final_message()
    yield {"type": "message", "message": final}


async def stream_agent(messages: list[dict], client=None, search_fn=web_search,
                       memory_digest: str = "") -> AsyncIterator[dict]:
    """Yields: {"type":"token"}, {"type":"searching"}, then one {"type":"done"}."""
    client = client or AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.request_timeout_s,
        max_retries=settings.llm_max_retries,
    )
    messages = list(messages)
    system = build_system_prompt(memory_digest)
    sources: dict[str, dict] = {}      # url -> {"id", "title", "url"}; ids are stable
    searches: list[str] = []
    resp = None

    # One extra iteration at the end is the forced "answer now" call.
    for step in range(1, settings.max_agent_steps + 2):
        forced = step > settings.max_agent_steps
        async for ev in _llm_turn(client, system, messages, forced):
            if ev["type"] == "message":
                resp = ev["message"]
            else:
                yield ev

        if forced or resp.stop_reason != "tool_use":
            break

        messages.append({"role": "assistant", "content": resp.content})
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        queries = [(b.input or {}).get("query", "") for b in tool_blocks]
        # Tells the client: discard any partial text (it was just a preamble), show status.
        yield {"type": "searching", "queries": queries}

        outputs = await asyncio.gather(*(_run_tool(b, search_fn) for b in tool_blocks))

        tool_results = []
        for query, block, out in zip(queries, tool_blocks, outputs):
            searches.append(query)
            payload = out
            if "results" in out:
                numbered = []
                for r in out["results"]:
                    src = sources.setdefault(
                        r["url"], {"id": len(sources) + 1, "title": r["title"], "url": r["url"]}
                    )
                    numbered.append({"id": src["id"], **r})   # new dicts: never mutate the cache
                payload = {**out, "results": numbered}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(payload),
                "is_error": "error" in out,
            })
        messages.append({"role": "user", "content": tool_results})

    answer, shown, grounding = finalize(_text(resp.content), sources, searches)
    steps = min(step, settings.max_agent_steps)
    yield {"type": "done", "result": AgentResult(answer, shown, searches, steps, grounding)}


async def run_agent(messages: list[dict], client=None, search_fn=web_search,
                    memory_digest: str = "") -> AgentResult:
    """Non-streaming wrapper: same logic, just waits for the end."""
    result = None
    async for ev in stream_agent(messages, client, search_fn, memory_digest):
        if ev["type"] == "done":
            result = ev["result"]
    return result