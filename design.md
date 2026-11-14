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

  ## Speed and accuracy (Phase 4)
**Speed**
- Streaming (SSE) via an async-generator agent loop: user sees words in ~1s
  instead of waiting for the whole answer. Time-to-first-token is measured
  and returned (`first_token_ms`).
- Search results cached in memory (5 min TTL, normalised key, size-capped,
  errors never cached). Tradeoff: answers can be up to 5 minutes stale.
- Parallel tool calls (asyncio.gather) and short snippets keep LLM input small.
- Search timeout 8s per attempt so retries fit within the request budget.

**Reliability**
- Search: up to 3 attempts with exponential backoff, only for timeouts / 429 / 5xx.
  Auth and bad-request errors fail fast.
- LLM client: SDK retries (max_retries=2) on 429/5xx.
- Mid-stream failures are sent as an SSE `error` event (status is already 200).

**Accuracy (grounding)**
- Every search result gets a stable numeric id; the model must cite [id].
- Code verifies citations: nonexistent ids are stripped and a `grounding` label
  (cited / uncited / search_failed / not_needed) is returned with each answer.
- Rejected: a second "judge" LLM call per answer (doubles latency and cost).
  Used offline in evals instead.
- Known limit: this proves citations exist, not that a snippet fully supports
  the claim. Phase 5 evals measure that.
- Known limit: the cache is per-process, so it is lost on restart and not shared
  across workers. At scale I would use Redis.

  ## Frontend (Phase 5)
- Single static HTML file served by FastAPI at `/`: no build step, same origin
  (no CORS), one deployable unit.
- Reads the POST SSE stream with fetch + ReadableStream (EventSource is GET-only).
- `searching` event clears the model's preamble; `done` re-renders the verified
  answer, so the screen always matches what was saved.
- All model/web text is inserted via textContent (no XSS); citation links only
  render for ids the server verified and only for http(s) URLs.
- UI shows trust signals: grounding badge, source chips, latency and first-word time.
- Conversation id kept in localStorage; history restored from the API on reload.

## Evals (Phase 5)
- `evals/cases.json` + `evals/run_evals.py` run 10 cases against the real server.
- Deterministic checks: search-or-not, expected content, grounding label,
  minimum sources, no leaked internal timestamp format.
- LLM judge (offline only) for fuzzy cases: time recall and not fabricating memories.
- Multi-turn cases cover short-term memory and cross-conversation recall.
- Reports pass rate and latency (p50 with/without search, p95, first token).
- Results (fill in from `evals/results.json`): X/10 pass, first token ~Xms,
  no-search p50 ~Xms, search p50 ~Xms.
- Limits: small suite, one run (model output varies, so results can flicker),
  judge is itself a model, and live-news cases can only check behaviour and
  citations, not truth.