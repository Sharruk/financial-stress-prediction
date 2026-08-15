from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_project_status():
    response = client.get("/api/project/status")
    # Even if missing, it should return a 404 cleanly
    assert response.status_code in [200, 404]

def test_experiments_endpoint():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    assert "experiments" in response.json()
