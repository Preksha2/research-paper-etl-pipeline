"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "pinecone_connected" in data
        assert "openai_configured" in data
