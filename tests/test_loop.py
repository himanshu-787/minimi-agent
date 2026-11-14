import asyncio
from types import SimpleNamespace as NS

from app.agent.loop import run_agent


class FakeClient:
    """Plays back scripted model responses."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = self

    async def create(self, **kwargs):
        return self.responses.pop(0)


def tool_call(query):
    return NS(stop_reason="tool_use", content=[
        NS(type="tool_use", id="t1", name="web_search", input={"query": query})
    ])


def final(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


async def fake_search(query):
    return {"query": query, "results": [{"title": "T", "url": "http://x.com", "snippet": "s"}]}


def test_searches_then_answers():
    client = FakeClient([tool_call("python release"), final("Python 3.13 is out.")])
    result = asyncio.run(run_agent([{"role": "user", "content": "hi"}], client, fake_search))
    assert result.answer == "Python 3.13 is out."
    assert result.searches == ["python release"]
    assert result.sources[0]["url"] == "http://x.com"


def test_no_search_when_not_needed():
    client = FakeClient([final("360")])
    result = asyncio.run(run_agent([{"role": "user", "content": "15% of 240"}], client, fake_search))
    assert result.searches == []
    assert result.steps == 1


def test_step_limit_forces_answer():
    # Model keeps asking for searches; the loop must still end with an answer.
    from app.config import settings
    n = settings.max_agent_steps
    client = FakeClient([tool_call(f"q{i}") for i in range(n)] + [final("best effort")])
    result = asyncio.run(run_agent([{"role": "user", "content": "x"}], client, fake_search))
    assert result.answer == "best effort"