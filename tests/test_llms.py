import importlib
import json
import sys
import types
from unittest.mock import MagicMock

import pytest


class _Chunk:
    def __init__(self, text):
        self.text = text


@pytest.fixture
def llms(monkeypatch):
    """Import llms with a stand-in for google.genai, which may not be installed."""
    genai = types.ModuleType("google.genai")
    genai.types = types.SimpleNamespace(
        Content=MagicMock(),
        Part=MagicMock(),
        GenerateContentConfig=MagicMock(),
        ThinkingConfig=MagicMock(),
        Tool=MagicMock(),
        GoogleSearch=MagicMock(),
    )
    genai.Client = MagicMock()
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.delitem(sys.modules, "dreams_app.core.extra.llms", raising=False)
    module = importlib.import_module("dreams_app.core.extra.llms")
    yield module, genai
    sys.modules.pop("dreams_app.core.extra.llms", None)


def test_generate_skips_stream_chunks_without_text(llms):
    module, genai = llms
    payload = {"positive": [{"theme": "Family", "meaning": "Close ties."}], "negative": []}
    genai.Client.return_value.models.generate_content_stream.return_value = iter([
        _Chunk(json.dumps(payload)[:20]),
        _Chunk(None),
        _Chunk(json.dumps(payload)[20:]),
        _Chunk(None),
    ])
    collection = MagicMock()

    data = module.generate("u1", ["family"], ["loss"], collection)

    assert data == payload
    collection.update_one.assert_called_once_with(
        {"user_id": "u1"}, {"$set": {"data": payload}}, upsert=True
    )
