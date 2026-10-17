import logging
import pytest

pytest.importorskip("chromadb")
from dreams_app.core.vector_store import VectorStore


def test_location_collection_is_created(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    assert "layer_2_semantic" in store.collections


def test_location_vector_is_stored(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    assert store.store_vector(
        "layer_2_semantic", "post-1", [0.0] * 384, {"location_test": "church"}
    ) is True


def test_unknown_collection_warn(tmp_path, caplog):
    store = VectorStore(persist_dir=str(tmp_path))
    
    with caplog.at_level(logging.WARNING):
        assert store.store_vector("no_such_collection", "x", [0.0] * 384) is False
    assert "no_such_collection" in caplog.text