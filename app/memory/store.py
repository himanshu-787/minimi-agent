import json
import sqlite3
from contextlib import closing
from pathlib import Path

from app.timeutil import format_ts, now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    meta            TEXT,              -- JSON: searches + sources used
    created_at      TEXT NOT NULL      -- UTC ISO-8601
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id, id);
"""


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
            conn.executescript(SCHEMA)

    def save_turn(self, conversation_id: str, user_text: str, user_ts: str,
                  assistant_text: str, meta: dict | None = None) -> None:
        """Save the user message and the answer atomically: both or neither."""
        with closing(self._connect()) as conn:
            with conn:  # one transaction
                conn.execute(
                    "INSERT OR IGNORE INTO conversations (id, created_at) VALUES (?, ?)",
                    (conversation_id, user_ts),
                )
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
                    "VALUES (?, 'user', ?, NULL, ?)",
                    (conversation_id, user_text, user_ts),
                )
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, meta, created_at) "
                    "VALUES (?, 'assistant', ?, ?, ?)",
                    (conversation_id, assistant_text,
                     json.dumps(meta) if meta else None, now_iso()),
                )

    def get_history(self, conversation_id: str, limit: int) -> list[dict]:
        """Last `limit` messages of one conversation, oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
        history = [dict(r) for r in reversed(rows)]
        # The API wants the first message to be from the user. Cutting the
        # window may leave an assistant message first, so trim it.
        while history and history[0]["role"] != "user":
            history.pop(0)
        return history

    def recent_user_questions(self, limit: int, exclude_conversation: str | None = None) -> list[dict]:
        """Newest-first questions the user asked, across conversations."""
        query = "SELECT content, created_at FROM messages WHERE role = 'user'"
        args: list = []
        if exclude_conversation:
            query += " AND conversation_id != ?"
            args.append(exclude_conversation)
        query += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with closing(self._connect()) as conn:
            return [dict(r) for r in conn.execute(query, args).fetchall()]


def format_digest(rows: list[dict]) -> str:
    """Turn recent questions into a short block for the system prompt."""
    return "\n".join(f"- {format_ts(r['created_at'])}: {r['content'][:120]}" for r in rows)