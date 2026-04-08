from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validation():
    response = client.post("/api/policy-chatbot/ask", json={"user_query": ""})
    assert response.status_code == 422