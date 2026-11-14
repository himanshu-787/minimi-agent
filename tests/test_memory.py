from app.memory.store import MemoryStore
from app.timeutil import format_ts


def make_store(tmp_path):
    store = MemoryStore(str(tmp_path / "test.db"))
    store.init()
    return store


def test_roundtrip_keeps_order(tmp_path):
    s = make_store(tmp_path)
    s.save_turn("c1", "first q", "2026-11-14T10:00:00+00:00", "first a")
    s.save_turn("c1", "second q", "2026-11-14T10:05:00+00:00", "second a")
    history = s.get_history("c1", 10)
    assert [m["content"] for m in history] == ["first q", "first a", "second q", "second a"]


def test_conversations_are_isolated(tmp_path):
    s = make_store(tmp_path)
    s.save_turn("c1", "q in c1", "2026-11-14T10:00:00+00:00", "a1")
    s.save_turn("c2", "q in c2", "2026-11-14T10:01:00+00:00", "a2")
    assert [m["content"] for m in s.get_history("c2", 10)] == ["q in c2", "a2"]


def test_window_always_starts_with_user(tmp_path):
    s = make_store(tmp_path)
    s.save_turn("c1", "first q", "2026-11-14T10:00:00+00:00", "first a")
    s.save_turn("c1", "second q", "2026-11-14T10:05:00+00:00", "second a")
    # A window of 3 would begin with an assistant message; it must be trimmed.
    history = s.get_history("c1", 3)
    assert history[0]["role"] == "user"
    assert [m["content"] for m in history] == ["second q", "second a"]


def test_recent_questions_across_chats_and_exclusion(tmp_path):
    s = make_store(tmp_path)
    s.save_turn("c1", "old question", "2026-11-13T09:00:00+00:00", "a")
    s.save_turn("c2", "newer question", "2026-11-14T09:00:00+00:00", "a")
    rows = s.recent_user_questions(10)
    assert [r["content"] for r in rows] == ["newer question", "old question"]
    rows = s.recent_user_questions(10, exclude_conversation="c2")
    assert [r["content"] for r in rows] == ["old question"]


def test_memory_survives_restart(tmp_path):
    path = str(tmp_path / "test.db")
    first = MemoryStore(path)
    first.init()
    first.save_turn("c1", "remember me", "2026-11-14T10:00:00+00:00", "ok")
    second = MemoryStore(path)  # simulates a server restart
    second.init()
    assert second.get_history("c1", 10)[0]["content"] == "remember me"


def test_timestamp_shown_in_local_time():
    shown = format_ts("2026-11-14T00:00:00+00:00")
    assert "05:30" in shown and "IST" in shown


def test_digest_is_scoped_to_client(tmp_path):
    s = make_store(tmp_path)
    s.save_turn("c1", "alice question", "2026-11-14T09:00:00+00:00", "a", client_id="alice")
    s.save_turn("c2", "bob question", "2026-11-14T09:01:00+00:00", "a", client_id="bob")
    rows = s.recent_user_questions(10, client_id="alice")
    assert [r["content"] for r in rows] == ["alice question"]
