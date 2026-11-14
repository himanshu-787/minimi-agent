# minimi chat agent

A fast, accurate, production-grade chat agent with one tool: **web search**. Built for the **minimi** Software Engineering Intern challenge.

- **Fast Streaming**: Token-by-token Server-Sent Events (SSE) with ~820ms time-to-first-token.
- **Accurate Grounding**: In-code citation verification of search results with trust badges (`[1]`, `[2]`).
- **Persistent Temporal Memory**: SQLite database storing UTC timestamps, allowing the agent to answer "what did I ask and when" across sessions.
- **Client Isolation**: Memory digests and cross-conversation recall scoped per client ID.
- **Production Guardrails**: In-memory search TTL caching, exponential backoff retries, sliding window rate limiter, and max step limits.
- **Comprehensive Evals**: 10-case automated test suite testing correctness, citations, memory recall, and refusal of fabricated memory.

**Architecture & Decisions:** See [design.md](design.md)  
**Evaluation Report:** See [evals/results.json](evals/results.json)

---

## Architecture

```
User (Browser) ──► FastAPI (/chat/stream, SSE) ──► stream_agent() ──► Claude (anthropic)
                          │                              │                    ▲
                     Rate Limiter                        ▼                    │
                   (sliding window)               web_search (Tavily) ────────┘
                          │                      (TTL cache + retries)
                          ▼
                  Memory (SQLite, WAL)
                (client_id scoped digest)
```

---

## Directory Layout

| Path | Purpose |
|---|---|
| `app/agent/` | ReAct tool-calling loop, prompts, and in-code citation validator |
| `app/tools/` | Web search integration (Tavily), TTL caching, and exponential backoff |
| `app/memory/` | SQLite store (WAL mode), migrations, timestamps, and client-scoped digest |
| `app/ratelimit.py` | Sliding window per-IP rate limiter |
| `app/timeutil.py` | Centralized timezone utilities (UTC storage, local display) |
| `frontend/` | Zero-build, single-file chat UI with dark/light theme, streaming, and citations |
| `tests/` | 21 fast unit tests covering agent loop, memory, search caching, and API routes |
| `evals/` | 10 end-to-end evaluation cases, runner (`run_evals.py`), and results |
| `design.md` | Full engineering design, architectural decisions, and tradeoffs |
| `Dockerfile` | Multi-stage production container setup |

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/minimi-agent.git
cd minimi-agent

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and add your API keys:
```bash
# On Windows (PowerShell):
Copy-Item .env.example .env
# On Linux / macOS:
cp .env.example .env
```

Edit `.env`:
```env
ANTHROPIC_API_KEY=your-anthropic-api-key
MODEL_NAME=claude-sonnet-5-5
SEARCH_API_KEY=your-tavily-search-api-key
```

### 4. Run the Server
```bash
uvicorn app.main:app --reload
```
Open your browser at **http://127.0.0.1:8000/**.

---

## Running Tests & Evals

### Unit Tests
The test suite consists of 21 tests using mocked clients and in-memory databases (no network calls, fast execution):
```bash
pytest -q
```
Expected output:
```text
.....................                                                    [100%]
21 passed in 4.97s
```

### End-to-End Evaluation Suite
The evaluation suite runs 10 multi-turn test cases against the live server, testing search triggers, citation binding, short-term memory, duration estimation, cross-conversation recall, and memory hallucination resistance:
```bash
# In terminal 1 (with server running):
uvicorn app.main:app

# In terminal 2:
python -m evals.run_evals
```
Add `--no-judge` to run deterministic checks without offline LLM grading.

---

## Deployment (Docker)

The project includes a production `Dockerfile`:
```bash
# Build the Docker image
docker build -t minimi-agent .

# Run container
docker run -p 8000:8000 --env-file .env minimi-agent
```

You can deploy directly to platforms like **Render**, **Railway**, or **Fly.io** by connecting the GitHub repository and adding `ANTHROPIC_API_KEY` and `SEARCH_API_KEY` in the platform's environment settings.

---

## API Endpoints

- `GET /` — Serves the chat web interface.
- `GET /health` — Health check returning service status and model configuration.
- `POST /chat/stream` — Primary streaming endpoint (Server-Sent Events) yielding `start`, `token`, `searching`, `done`, and `error` frames.
- `POST /chat` — Non-streaming fallback endpoint returning `ChatResponse` with latency metrics and sources.
- `GET /conversations/{id}/messages` — Retrieves time-ordered chat history for a given conversation ID.
