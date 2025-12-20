import pytest
import json
from unittest.mock import patch
from flask import Flask
from dreamsApp.app.utils.sentiment import bp  # Adjust if Blueprint is registered differently

@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    app.config["TESTING"] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

# Mock the AI response for all tests 
@pytest.fixture
def mock_sentiment():
    with patch('dreamsApp.app.utils.sentiment.get_image_caption_and_sentiment') as mock:
        mock.return_value = {
            "imgcaption": "A mock caption",
            "sentiment": {"label": "POSITIVE", "score": 0.99}
        }
        yield mock

def test_valid_caption(client, mock_sentiment):
    """Refactored to include image and correct URL"""
    payload = {
        "caption": "This is a wonderful day!",
        "image_path_or_url": "http://mock.url/img.jpg" # ADDED: Required by App
    }
    response = client.post(
        "/sentiment/analyze", 
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == mock_sentiment.return_value

def test_empty_caption(client, mock_sentiment):
    """
    CHANGED: The app logic allows empty captions (defaults to ""),
    so we now expect 200 OK, not 400.
    """
    payload = {
        "caption": "",
        "image_path_or_url": "http://mock.url/img.jpg"
    }
    response = client.post(
        "/sentiment/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200 # App allows empty caption

def test_missing_caption_key(client, mock_sentiment):
    """
    CHANGED: App defaults caption to "" if missing, so this should pass (200).
    """
    payload = {
        "image_path_or_url": "http://mock.url/img.jpg"
    }
    response = client.post(
        "/sentiment/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200 # App handles missing key gracefully

def test_missing_image_key(client, mock_sentiment):
    """
    NEW: This replaces the old checks. This is the REAL required field.
    """
    payload = {
        "caption": "I have text but no image"
    }
    response = client.post(
        "/sentiment/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    assert "image_path_or_url" in response.get_json()["error"]

def test_invalid_content_type(client):
    response = client.post(
        "/sentiment/analyze",
        data="caption=I love this!",
        content_type="text/plain"
    )
    # This remains correct
    assert response.status_code in [400, 415]

def test_long_caption(client, mock_sentiment):
    """Updated to include image"""
    payload = {
        "caption": "This is a very long caption " * 100,
        "image_path_or_url": "http://mock.url/img.jpg"
    }
    response = client.post(
        "/sentiment/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200

def test_special_characters(client, mock_sentiment):
    """Updated to include image"""
    payload = {
        "caption": "Hello! @#$%^&*()_+ 世界",
        "image_path_or_url": "http://mock.url/img.jpg"
    }
    response = client.post(
        "/sentiment/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200

def test_numeric_caption(client, mock_sentiment):
    """Updated to include image"""
    payload = {
        "caption": "123 456 789",
        "image_path_or_url": "http://mock.url/img.jpg"
    }
    response = client.post(
        "/sentiment/analyze",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200

def test_malformed_json(client):
    response = client.post(
        "/sentiment/analyze",
        data="{invalid json",
        content_type="application/json"
    )
    assert response.status_code == 400