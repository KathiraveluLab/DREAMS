"""Tests for keyword clustering (dreams_app/core/extra/clustering.py)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dreams_app.core.extra import clustering

# Keywords as upload_post stores them in Mongo: the embedding is removed before saving.
POSITIVE = ['park', 'friend', 'garden', 'walk']
NEGATIVE = ['hospital', 'night', 'alone', 'pain']


def _stored(words):
    return [{'keyword': w, 'timestamp': '2026-10-18T10:00:00'} for w in words]


def _doc():
    return {
        'user_id': 'alice',
        'positive_keywords': _stored(POSITIVE),
        'negative_keywords': _stored(NEGATIVE),
    }


def _fake_embed(keywords):
    """Two well-separated groups: the positive words and the negative words."""
    rng = np.random.default_rng(0)
    base = {w: [1.0, 0.0, 0.0] for w in POSITIVE}
    base.update({w: [0.0, 1.0, 0.0] for w in NEGATIVE})
    return np.array([np.array(base[k]) + rng.normal(0, 0.1, 3) for k in keywords])


def test_stored_embeddings_are_used_when_present():
    doc = {'positive_keywords': [{'keyword': 'park', 'embedding': [0.1, 0.2]}]}

    vectors, metadata = clustering.get_vectors_and_metadata(doc)

    assert vectors.tolist() == [[0.1, 0.2]]
    assert metadata == [{'keyword': 'park', 'sentiment': 'positive_keywords'}]


def test_keywords_without_stored_embedding_are_embedded_on_demand():
    vectors, metadata = clustering.get_vectors_and_metadata(_doc(), embed=_fake_embed)

    assert vectors.shape == (8, 3)
    assert [m['keyword'] for m in metadata] == POSITIVE + NEGATIVE
    assert {m['sentiment'] for m in metadata} == {'positive_keywords', 'negative_keywords'}


def test_keywords_without_embedding_give_no_vectors_when_nothing_can_embed():
    vectors, metadata = clustering.get_vectors_and_metadata(_doc())

    assert len(vectors) == 0
    assert metadata == []


def test_clustering_runs_for_keywords_stored_without_embeddings(monkeypatch):
    pytest.importorskip('hdbscan')
    monkeypatch.setattr(clustering, '_embed_keywords', _fake_embed)
    collection = MagicMock()
    collection.find.return_value = [_doc()]

    clustering.cluster_keywords_for_all_users(collection)

    collection.update_one.assert_called_once()
    query, update = collection.update_one.call_args[0]
    assert query == {'user_id': 'alice'}
    result = update['$set']['clustered_keywords']
    assert {r['keyword'] for r in result} == set(POSITIVE + NEGATIVE)
    labels = {r['keyword']: r['cluster'] for r in result}
    assert len({labels[w] for w in POSITIVE}) == 1
    assert len({labels[w] for w in NEGATIVE}) == 1
    assert labels['park'] != labels['hospital']


def test_user_is_skipped_when_keywords_cannot_be_embedded(monkeypatch):
    pytest.importorskip('hdbscan')
    monkeypatch.setattr(clustering, '_embed_keywords', lambda keywords: None)
    collection = MagicMock()
    collection.find.return_value = [_doc()]

    clustering.cluster_keywords_for_all_users(collection)

    collection.update_one.assert_not_called()
