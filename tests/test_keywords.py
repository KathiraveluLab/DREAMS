"""Tests for keyword extraction post-processing."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from dreams_app.core.extra import keywords


class _Doc:
    def __init__(self, ents, chunks):
        self.ents = ents
        self.noun_chunks = chunks

    def __iter__(self):
        return iter([])


class _Chunk(list):
    def __init__(self, text):
        super().__init__([SimpleNamespace(is_stop=False, is_punct=False)
                          for _ in text.split()])
        self.text = text
        self.root = SimpleNamespace(pos_="NOUN", lemma_=text)


def _extract(*chunk_texts):
    doc = _Doc([], [_Chunk(t) for t in chunk_texts])
    model = MagicMock()
    model.encode.side_effect = lambda items: np.zeros((len(items), 3))
    with patch.object(keywords, "_get_nlp", return_value=lambda s: doc), \
            patch.object(keywords, "_get_keyword_model", return_value=model):
        result = keywords.extract_keywords_and_vectors("text", False)
    return {r["keyword"] for r in result}


def test_keyword_kept_when_only_a_letter_substring_of_longer_phrase():
    assert _extract("tea", "steam engine") == {"tea", "steam engine"}


def test_keyword_dropped_when_a_whole_word_of_longer_phrase():
    assert _extract("garden", "community garden") == {"community garden"}
