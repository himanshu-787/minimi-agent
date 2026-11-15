import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from groq import AsyncGroq

from app.agent.prompts import build_system_prompt
from app.config import settings
from app.tools.web_search import WEB_SEARCH_TOOL, web_search

# 1-2 digits only, so a year like [2024] is never mistaken for a citation.
CITATION = re.compile(r"\[(\d{1,2})\]")
THINK_TAGS = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class AgentResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    steps: int = 0
    grounding: str = "not_needed"   # not_needed | cited | uncited | search_failed


async def _run_tool_call(tc: dict, search_fn) -> dict:
    fn = tc.get("function", {})
    name = fn.get("name")
    if name != "web_search":
        return {"error": f"Unknown tool: {name}"}
    raw_args = fn.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}
    else:
        args = raw_args or {}
    query = (args.get("query") or "").strip()
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

    clean = THINK_TAGS.sub("", answer)
    clean = re.sub(r"^.*?</think>\s*", "", clean, flags=re.DOTALL)
    clean = CITATION.sub(keep, clean)
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
    """One streamed model call via Groq. Yields token events, then the final message payload."""
    full_messages = [{"role": "system", "content": system}] + messages
    kwargs = dict(
        model=settings.model_name,
        max_tokens=settings.max_output_tokens,
        tools=[WEB_SEARCH_TOOL],
        tool_choice="none" if forced else "auto",
        messages=full_messages,
        stream=True,
    )
    stream = await client.chat.completions.create(**kwargs)
    accumulated_text = ""
    accumulated_tools: dict[int, dict] = {}
    finish_reason = None

    async for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        choice = chunk.choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
        delta = getattr(choice, "delta", None)
        if not delta:
            continue
        if getattr(delta, "content", None):
            accumulated_text += delta.content
            yield {"type": "token", "text": delta.content}
        if getattr(delta, "tool_calls", None):
            for tc in delta.tool_calls:
                idx = getattr(tc, "index", 0)
                if idx not in accumulated_tools:
                    tc_fn = getattr(tc, "function", None)
                    accumulated_tools[idx] = {
                        "id": getattr(tc, "id", "") or "",
                        "name": getattr(tc_fn, "name", "") or "",
                        "arguments": getattr(tc_fn, "arguments", "") or "",
                    }
                else:
                    if getattr(tc, "id", None):
                        accumulated_tools[idx]["id"] += tc.id
                    tc_fn = getattr(tc, "function", None)
                    if tc_fn:
                        if getattr(tc_fn, "name", None):
                            accumulated_tools[idx]["name"] += tc_fn.name
                        if getattr(tc_fn, "arguments", None):
                            accumulated_tools[idx]["arguments"] += tc_fn.arguments

    tool_calls = None
    if accumulated_tools:
        tool_calls = [
            {
                "id": t["id"] or f"call_{i}",
                "type": "function",
                "function": {
                    "name": t["name"],
                    "arguments": t["arguments"],
                },
            }
            for i, t in sorted(accumulated_tools.items())
        ]

    yield {
        "type": "message",
        "content": accumulated_text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
    }


async def stream_agent(messages: list[dict], client=None, search_fn=web_search,
                       memory_digest: str = "") -> AsyncIterator[dict]:
    """Yields: {"type":"token"}, {"type":"searching"}, then one {"type":"done"}."""
    client = client or AsyncGroq(
        api_key=settings.groq_api_key,
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
                resp = ev
            else:
                yield ev

        if forced or not resp or not resp.get("tool_calls"):
            break

        tool_calls = resp["tool_calls"]
        messages.append({
            "role": "assistant",
            "content": resp["content"] or None,
            "tool_calls": tool_calls,
        })

        queries = []
        for tc in tool_calls:
            raw_args = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            queries.append(args.get("query", ""))

        # Tells the client: discard any partial text (it was just a preamble), show status.
        yield {"type": "searching", "queries": queries}

        outputs = await asyncio.gather(*(_run_tool_call(tc, search_fn) for tc in tool_calls))

        for tc, query, out in zip(tool_calls, queries, outputs):
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
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(payload),
            })

    final_text = resp["content"] if resp else ""
    answer, shown, grounding = finalize(final_text, sources, searches)
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
