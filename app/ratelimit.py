import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from app.config import settings

_hits: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    # Behind a host's proxy, the real client is the first X-Forwarded-For entry.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Sliding window: at most N requests per IP in the last 60 seconds."""
    ip, now = client_ip(request), time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_limit_per_min:
        raise HTTPException(429, "Too many requests. Please wait a minute.")
    q.append(now)
