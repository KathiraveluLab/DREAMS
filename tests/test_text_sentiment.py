"""Tests for ml.text_sentiment and the /api/text-emotion endpoint."""
import sys
import types

import pytest

from ml import text_sentiment as ts

pytest.importorskip("flask")

import analytics_server  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_pipeline_state(monkeypatch):
    monkeypatch.setattr(ts, "_sentiment_pipeline", None)
    monkeypatch.setattr(ts, "_pipeline_loaded", False)
    monkeypatch.setattr(ts, "_pipeline_error", None)
    monkeypatch.setattr(ts, "_pipeline_retry_at", 0.0)


def _install_transformers(monkeypatch, factory):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=factory))


def _joy_pipeline(*args, **kwargs):
    return lambda text: [[{"label": "joy", "score": 0.9}, {"label": "sadness", "score": 0.1}]]


# --- a failed model load is retried ---------------------------------------------------------


def test_model_is_used_when_it_loads(monkeypatch):
    _install_transformers(monkeypatch, _joy_pipeline)

    result = ts.analyze_text_sentiment("What a day")

    assert result["model"] == "j-hartmann/emotion-english-distilroberta-base"
    assert result["positive"] == pytest.approx(0.9)


def test_failed_load_is_retried_after_the_cooldown(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(ts.time, "monotonic", lambda: clock["now"])
    attempts = []

    def flaky(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("model hub unreachable")
        return _joy_pipeline()

    _install_transformers(monkeypatch, flaky)

    assert ts.analyze_text_sentiment("I feel good")["model"] == "keyword-fallback"

    clock["now"] += ts.PIPELINE_RETRY_S + 1
    assert ts.analyze_text_sentiment("I feel good")["model"].startswith("j-hartmann")
    assert len(attempts) == 2


def test_failed_load_is_not_retried_on_every_request(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(ts.time, "monotonic", lambda: clock["now"])
    attempts = []

    def down(*args, **kwargs):
        attempts.append(1)
        raise OSError("model hub unreachable")

    _install_transformers(monkeypatch, down)

    for _ in range(5):
        clock["now"] += 10
        assert ts.analyze_text_sentiment("hello")["model"] == "keyword-fallback"

    assert len(attempts) == 1


def test_loaded_model_is_kept(monkeypatch):
    attempts = []

    def factory(*args, **kwargs):
        attempts.append(1)
        return _joy_pipeline()

    _install_transformers(monkeypatch, factory)

    ts.analyze_text_sentiment("one")
    ts.analyze_text_sentiment("two")

    assert len(attempts) == 1


# --- the keyword fallback sees words next to punctuation -------------------------------------


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("sad: really", "negative"),
        ("I am happy;", "positive"),
        ("(happy)", "positive"),
        ("happy-go-lucky", "positive"),
        ("so stressed... and lonely", "negative"),
        ("a daily routine", "neutral"),
    ],
)
def test_keyword_fallback_matches_words_next_to_punctuation(text, label):
    result = ts._keyword_fallback(text)

    assert max(result, key=result.get) == label
    assert result[label] == 1.0


def test_keyword_fallback_without_known_words_is_even():
    assert ts._keyword_fallback("zzz qqq") == {"positive": 0.33, "negative": 0.33, "neutral": 0.34}


# --- /api/text-emotion input handling --------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(analytics_server, "_TEXT_SENTIMENT_AVAILABLE", True, raising=False)
    monkeypatch.setattr(analytics_server, "analyze_text_sentiment", ts.analyze_text_sentiment, raising=False)
    return analytics_server.app.test_client()


@pytest.mark.parametrize("text", [123, None, ["a"], {"a": 1}, True, "", "   \n"])
def test_non_string_or_blank_text_gets_400(client, text):
    response = client.post("/api/text-emotion", json={"text": text})

    assert response.status_code == 400
    assert response.get_json()["error"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data": "not json", "content_type": "application/json"},
        {"data": "[1, 2]", "content_type": "application/json"},
        {"data": "null", "content_type": "application/json"},
        {},
    ],
    ids=["invalid-json", "list-body", "null-body", "no-body"],
)
def test_bad_bodies_get_json_400(client, kwargs):
    response = client.post("/api/text-emotion", **kwargs)

    assert response.status_code == 400
    assert response.get_json()["error"]


def test_valid_text_is_analysed(client):
    response = client.post("/api/text-emotion", json={"text": "I feel so happy and grateful"})

    assert response.status_code == 200
    assert response.get_json()["positive"] == 1.0


def test_internal_errors_do_not_leak_exception_text(client, monkeypatch):
    def boom(text):
        raise RuntimeError("secret path /srv/models/token-abc123")

    monkeypatch.setattr(analytics_server, "analyze_text_sentiment", boom)

    response = client.post("/api/text-emotion", json={"text": "hello"})

    assert response.status_code == 500
    assert "abc123" not in response.get_data(as_text=True)
