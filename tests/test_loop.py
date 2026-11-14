import asyncio
from types import SimpleNamespace as NS

from app.agent.loop import run_agent, stream_agent


class FakeStream:
    """Imitates anthropic's `async with client.messages.stream(...)`."""
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for b in self.response.content:
                if b.type == "text":
                    yield b.text
        return gen()

    async def get_final_message(self):
        return self.response


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = self

    def stream(self, **kwargs):
        return FakeStream(self.responses.pop(0))


def tool_call(query):
    return NS(stop_reason="tool_use", content=[
        NS(type="tool_use", id="t1", name="web_search", input={"query": query})
    ])


def final(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


async def fake_search(query):
    return {"query": query, "results": [{"title": "T", "url": "http://x.com", "snippet": "s"}]}


async def failing_search(query):
    return {"error": "search is down"}


USER = [{"role": "user", "content": "hi"}]


def run(responses, search=fake_search):
    return asyncio.run(run_agent(USER, FakeClient(responses), search))


def test_searches_then_answers_with_citation():
    result = run([tool_call("python release"), final("Python 3.13 is out [1].")])
    assert result.answer == "Python 3.13 is out [1]."
    assert result.searches == ["python release"]
    assert result.sources[0]["url"] == "http://x.com" and result.sources[0]["id"] == 1
    assert result.grounding == "cited"


def test_no_search_when_not_needed():
    result = run([final("360")])
    assert result.searches == [] and result.steps == 1
    assert result.grounding == "not_needed"


def test_step_limit_forces_answer():
    from app.config import settings
    n = settings.max_agent_steps
    result = run([tool_call(f"q{i}") for i in range(n)] + [final("best effort")])
    assert result.answer == "best effort"
    assert result.steps == n


def test_invalid_citation_is_removed_but_years_survive():
    result = run([tool_call("q"), final("Real [1] and fake [7] in [2024].")])
    assert "[7]" not in result.answer
    assert "[1]" in result.answer and "[2024]" in result.answer
    assert "  " not in result.answer


def test_uncited_answer_is_flagged_and_sources_still_shown():
    result = run([tool_call("q"), final("Python 3.13 is out.")])
    assert result.grounding == "uncited"
    assert len(result.sources) == 1


def test_search_failure_is_labelled():
    result = run([tool_call("q"), final("I could not verify this.")], search=failing_search)
    assert result.grounding == "search_failed"
    assert result.sources == []


def test_stream_emits_events_in_order():
    client = FakeClient([tool_call("q"), final("Hello [1]")])

    async def collect():
        return [e async for e in stream_agent(USER, client, fake_search)]

    kinds = [e["type"] for e in asyncio.run(collect())]
    assert kinds == ["searching", "token", "done"]