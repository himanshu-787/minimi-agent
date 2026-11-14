import asyncio
import json
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

from app.agent.prompts import build_system_prompt
from app.config import settings
from app.tools.web_search import WEB_SEARCH_TOOL, web_search


@dataclass
class AgentResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    steps: int = 0


def _text(content) -> str:
    return "".join(b.text for b in content if b.type == "text").strip()


async def _run_tool(block, search_fn) -> dict:
    """Dispatch one tool request from the model."""
    if block.name != "web_search":
        return {"error": f"Unknown tool: {block.name}"}
    query = ((block.input or {}).get("query") or "").strip()
    if not query:
        return {"error": "Empty search query."}
    return await search_fn(query)


async def run_agent(messages: list[dict], client=None, search_fn=web_search) -> AgentResult:
    # client and search_fn are injectable so tests can fake them (see tests/).
    client = client or AsyncAnthropic(
        api_key=settings.anthropic_api_key, timeout=settings.request_timeout_s
    )
    messages = list(messages)  # never mutate the caller's list
    system = build_system_prompt()
    sources: dict[str, dict] = {}   # keyed by URL so duplicates vanish
    searches: list[str] = []

    for step in range(1, settings.max_agent_steps + 1):
        resp = await client.messages.create(
            model=settings.model_name,
            max_tokens=settings.max_output_tokens,
            system=system,
            tools=[WEB_SEARCH_TOOL],
            messages=messages,
        )

        # No tool requested -> the model is ready to answer. Done.
        if resp.stop_reason != "tool_use":
            return AgentResult(_text(resp.content), list(sources.values()), searches, step)

        # Tool requested: record the model's turn, run the tools, feed results back.
        messages.append({"role": "assistant", "content": resp.content})
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]

        # If the model asks for several searches at once, run them in parallel.
        outputs = await asyncio.gather(*(_run_tool(b, search_fn) for b in tool_blocks))

        tool_results = []
        for block, out in zip(tool_blocks, outputs):
            searches.append((block.input or {}).get("query", ""))
            for r in out.get("results", []):
                sources[r["url"]] = {"title": r["title"], "url": r["url"]}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(out),
                "is_error": "error" in out,
            })
        messages.append({"role": "user", "content": tool_results})

    # Step budget used up: force a final answer with what we have (no more tools).
    resp = await client.messages.create(
        model=settings.model_name,
        max_tokens=settings.max_output_tokens,
        system=system,
        tools=[WEB_SEARCH_TOOL],
        tool_choice={"type": "none"},
        messages=messages,
    )
    return AgentResult(_text(resp.content), list(sources.values()), searches, settings.max_agent_steps)