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