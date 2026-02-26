"""Tests for the FastAPI inference server."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.inference import app


@pytest.fixture
def client():
    """Return a TestClient for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRootEndpoint:
    def test_root_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Article Summarizer" in response.text


class TestSummarizeEndpoint:
    @patch("src.inference.get_model")
    def test_summarize_returns_summary(self, mock_get_model, client):
        mock_model = MagicMock()
        mock_model.summarize.return_value = "This is a generated summary."
        mock_get_model.return_value = mock_model

        response = client.post("/summarize", json={
            "article": "This is a long article that needs summarization. " * 10,
        })
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["summary"] == "This is a generated summary."
        assert data["article_length"] > 0
        assert data["summary_length"] > 0

    def test_summarize_rejects_short_article(self, client):
        response = client.post("/summarize", json={
            "article": "Short",
        })
        assert response.status_code == 422  # Validation error

    def test_summarize_rejects_empty_body(self, client):
        response = client.post("/summarize", json={})
        assert response.status_code == 422
