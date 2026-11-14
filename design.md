# Design: minimi chat agent

## Goal
A chat agent with one tool (web search) that answers fast, gets facts right,
and remembers what the user asked and when.

## Architecture (request flow)
User -> FastAPI -> Agent loop -> (LLM <-> web_search tool) -> Memory (SQLite) -> Response

## Key decisions
| Decision | Choice | Why | Alternative rejected |
|---|---|---|---|
| Backend | FastAPI | async, streaming, typed | Flask (no native async) |
| Memory | SQLite | zero-setup, durable, queryable by time | Vector DB (overkill) |
| Agent | single tool-calling loop | simple, debuggable | multi-agent framework |

## Speed / Accuracy / Memory
(to fill in Phases 3-4)

## Tradeoffs and what I'd do with more time
(to fill at the end)

## Agent loop (Phase 2)
- Single tool-calling loop; the model decides when to search.
- Safety: max 4 steps, 20s timeouts, forced final answer when budget is spent.
- Tool errors are returned to the model as data (is_error), never raised, so the
  agent can recover or be honest about failure.
- Parallel tool calls via asyncio.gather for speed.
- LLM client and search function are injected, so the loop is unit-tested with fakes.
- Date/time injected into the system prompt so "latest/today" questions work.

## Memory (Phase 3)
- SQLite with two tables (conversations, messages). Queries are exact and
  time-ordered, so a vector DB would add cost and fuzziness for no benefit.
- Timestamps stored in UTC, shown to the model and user in IST. Each user
  message is stamped "[Sat 14 Nov 2026, 15:30 IST]" so the model can answer
  "what did I ask, and when" without a separate memory tool (the challenge
  allows only web search).
- Short-term memory: last 20 messages of the conversation are replayed each turn.
  Long-term memory: last 15 questions from other conversations go into the
  system prompt as a digest.
- Only user text and final answers are stored (not raw search results) to keep
  context small and fast; searches/sources are kept in a `meta` JSON column.
- Turn saved atomically after a successful answer; no orphan messages, and a
  failed DB write never blocks the user's answer.
- Blocking sqlite3 calls run via asyncio.to_thread to keep the event loop free.
- Known limits: single-user (no auth), concurrent requests in one conversation
  can interleave, and old history beyond the window is not summarised.