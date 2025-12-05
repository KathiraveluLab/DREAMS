import pytest
import json
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

def test_valid_caption(client):
    payload = {
        "caption": "This is a wonderful day!"
    }
    response = client.post(
        "/sentiments/caption",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)
    assert "label" in json_data[0]
    assert "score" in json_data[0]
    assert json_data[0]["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0 <= json_data[0]["score"] <= 1

def test_empty_caption(client):
    payload = {
        "caption": ""
    }
    response = client.post(
        "/sentiments/caption",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "No caption provided"}

def test_missing_caption_key(client):
    payload = {}
    response = client.post(
        "/sentiments/caption",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "No caption provided"}

def test_invalid_content_type(client):
    response = client.post(
        "/sentiments/caption",
        data="caption=I love this!",
        content_type="text/plain"
    )
    # Still passes, but likely returns a 400 due to parsing failure
    assert response.status_code in [400, 415]
def test_long_caption(client):
    payload = {
        "caption": "This is a very long caption " * 100
    }
    response = client.post(
        "/sentiments/caption",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)
    assert "label" in json_data[0]
    assert "score" in json_data[0]

def test_special_characters(client):
    payload = {
        "caption": "Hello! @#$%^&*()_+ 世界"
    }
    response = client.post(
        "/sentiments/caption",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)

def test_numeric_caption(client):
    payload = {
        "caption": "123 456 789"
    }
    response = client.post(
        "/sentiments/caption",
        data=json.dumps(payload),
        content_type="application/json"
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)

def test_malformed_json(client):
    response = client.post(
        "/sentiments/caption",
        data="{invalid json",
        content_type="application/json"
    )
    assert response.status_code == 400

