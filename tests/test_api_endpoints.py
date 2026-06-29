from fastapi.testclient import TestClient


def test_root_endpoint(app_client: TestClient):
    response = app_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "health_url" in data


def test_health_endpoint(app_client: TestClient):
    # The health endpoint is mounted at api_prefix /health
    # Usually /api/v1/health
    response = app_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
