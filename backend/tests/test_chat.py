from fastapi.testclient import TestClient
from app.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

def test_chat_endpoint():
    with TestClient(app) as client:
        payload = {
            "message": "What products do you offer?",
            "top_k": 2
        }
        response = client.post("/api/chat", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
