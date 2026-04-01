from fastapi.testclient import TestClient

from src.app import app


def test_health_endpoint_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "SAT Vocabulary AI"
    assert "timestamp" in payload


def test_home_page_loads():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
