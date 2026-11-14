"""Run the agent through evals/cases.json and print a report card.

Usage (server must be running):
    python -m evals.run_evals [--base http://127.0.0.1:8000] [--no-judge]
"""
import argparse
import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from anthropic import AsyncAnthropic

from app.config import settings

CASES = Path(__file__).parent / "cases.json"
RESULTS = Path(__file__).parent / "results.json"

# The internal "[Sat 14 Nov 2026, ..." stamp must never leak into an answer.
LEAK = re.compile(r"\[(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{2} [A-Z][a-z]{2} \d{4}")


def fill(s: str) -> str:
    year = datetime.now(ZoneInfo(settings.display_timezone)).year
    return s.replace("{year}", str(year))


async def ask(http: httpx.AsyncClient, base: str, message: str, cid: str | None) -> dict:
    """Send one message to /chat/stream and collect everything that comes back."""
    payload = {"message": message, "client_id": "eval-client"}
    if cid:
        payload["conversation_id"] = cid
    out = {"cid": cid, "answer": "", "searches": [], "sources": [], "grounding": "",
           "latency_ms": None, "first_token_ms": None, "error": None}
    t0 = time.perf_counter()
    try:
        async with http.stream("POST", f"{base}/chat/stream", json=payload) as resp:
            resp.raise_for_status()
            event = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    if event == "start":
                        out["cid"] = data["conversation_id"]
                    elif event == "token" and out["first_token_ms"] is None:
                        out["first_token_ms"] = int((time.perf_counter() - t0) * 1000)
                    elif event == "done":
                        out.update(answer=data["answer"], searches=data["searches"],
                                   sources=data["sources"], grounding=data["grounding"])
                    elif event == "error":
                        out["error"] = data["message"]
    except httpx.HTTPError as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return out


def check(turn: dict, r: dict) -> list[str]:
    """Deterministic checks. Returns a list of failure messages (empty = pass)."""
    if r["error"]:
        return [f"request error: {r['error']}"]
    fails, ans = [], r["answer"].lower()

    expect = turn.get("expect_search")
    if expect is True and not r["searches"]:
        fails.append("expected a web search but none happened")
    if expect is False and r["searches"]:
        fails.append(f"unnecessary search: {r['searches']}")

    wanted = [fill(x).lower() for x in turn.get("must_contain_any", [])]
    if wanted and not any(x in ans for x in wanted):
        fails.append(f"answer lacks any of {wanted}")
    for bad in turn.get("must_not_contain", []):
        if bad.lower() in ans:
            fails.append(f"answer contains forbidden text: {bad!r}")

    if "grounding_in" in turn and r["grounding"] not in turn["grounding_in"]:
        fails.append(f"grounding was {r['grounding']!r}, wanted one of {turn['grounding_in']}")
    if len(r["sources"]) < turn.get("min_sources", 0):
        fails.append(f"only {len(r['sources'])} sources shown")
    if LEAK.search(r["answer"]):
        fails.append("leaked the internal timestamp format")
    return fails


async def judge(llm: AsyncAnthropic, question: str, answer: str, rubric: str) -> tuple[bool, str]:
    """A second model grades the answer against a rubric. Offline only."""
    try:
        resp = await llm.messages.create(
            model=settings.model_name,
            max_tokens=200,
            system="You are a strict grader. Reply with ONLY a JSON object: "
                   '{"pass": true or false, "reason": "one short sentence"}.',
            messages=[{"role": "user", "content":
                       f"Question: {question}\n\nAnswer: {answer}\n\nRubric: {rubric}"}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        verdict = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        return bool(verdict["pass"]), str(verdict.get("reason", ""))
    except Exception as e:  # a broken judge must not crash the whole run
        return False, f"judge error: {type(e).__name__}"


async def run_case(http, llm, base, case, use_judge) -> dict:
    cid, failures, turns_out = None, [], []
    for i, turn in enumerate(case["turns"], 1):
        if turn.get("new_conversation"):
            cid = None
        r = await ask(http, base, turn["say"], cid)
        cid = r["cid"] or cid
        problems = check(turn, r)
        if turn.get("rubric") and use_judge and not r["error"]:
            ok, why = await judge(llm, turn["say"], r["answer"], turn["rubric"])
            if not ok:
                problems.append(f"judge: {why}")
        failures += [f"turn {i}: {p}" for p in problems]
        turns_out.append({"say": turn["say"], "answer": r["answer"], "searches": r["searches"],
                          "grounding": r["grounding"], "latency_ms": r["latency_ms"],
                          "first_token_ms": r["first_token_ms"]})
    return {"id": case["id"], "passed": not failures, "failures": failures, "turns": turns_out}


def pct(values: list[int], p: float) -> int:
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def summarise(results: list[dict]) -> dict:
    turns = [t for r in results for t in r["turns"] if t["latency_ms"] is not None]
    with_search = [t["latency_ms"] for t in turns if t["searches"]]
    without = [t["latency_ms"] for t in turns if not t["searches"]]
    firsts = [t["first_token_ms"] for t in turns if t["first_token_ms"] is not None]
    summary = {
        "cases_passed": sum(r["passed"] for r in results),
        "cases_total": len(results),
        "first_token_ms_p50": pct(firsts, 0.5) if firsts else None,
        "latency_ms_p50_no_search": pct(without, 0.5) if without else None,
        "latency_ms_p50_with_search": pct(with_search, 0.5) if with_search else None,
        "latency_ms_p95_all": pct([t["latency_ms"] for t in turns], 0.95) if turns else None,
    }
    return summary


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judged rubrics")
    args = ap.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    llm = AsyncAnthropic(api_key=settings.anthropic_api_key)

    async with httpx.AsyncClient(timeout=90) as http:
        try:
            await http.get(f"{args.base}/health")
        except httpx.HTTPError:
            print(f"Cannot reach {args.base}. Start the server first.")
            return 1
        results = []
        for case in cases:   # sequential on purpose: fair latency, no rate-limit surprises
            res = await run_case(http, llm, args.base, case, not args.no_judge)
            results.append(res)
            print(("PASS  " if res["passed"] else "FAIL  ") + res["id"])
            for f in res["failures"]:
                print("        - " + f)

    summary = summarise(results)
    print("\n=== Report card ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    RESULTS.write_text(json.dumps({"summary": summary, "results": results}, indent=2),
                       encoding="utf-8")
    print(f"\nFull details: {RESULTS}")
    return 0 if summary["cases_passed"] == summary["cases_total"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))