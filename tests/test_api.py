from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_index_page_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "minimi" in r.text


def test_health_ok():
    assert client.get("/health").json()["status"] == "ok"


def test_empty_message_is_rejected():
    assert client.post("/chat", json={"message": ""}).status_code == 422