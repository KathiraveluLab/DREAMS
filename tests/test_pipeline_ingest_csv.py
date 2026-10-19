"""Tests for DreamsPipeline.ingest_csv with blank cells."""

from unittest.mock import patch

import pytest

from dreams_app.core.pipeline import DreamsPipeline


def _ingest(tmp_path, csv_text):
    csv_path = tmp_path / "posts.csv"
    csv_path.write_text(csv_text)
    seen = []

    def fake_process(self, user_id, image_path, caption, timestamp_iso=None):
        seen.append((user_id, image_path, caption, timestamp_iso))
        return {
            "post_doc": {"timestamp": None, "sentiment": {"label": "neutral", "score": 1.0}},
            "caption_embedding": None,
            "image_embedding": None,
        }

    with patch.object(DreamsPipeline, "process_new_post", fake_process), \
         patch("dreams_app.core.database.db_manager") as db, \
         patch("dreams_app.core.vector_store.vector_store"):
        db.insert_post.return_value = 1
        DreamsPipeline().ingest_csv(str(csv_path))
    return seen, db


def test_blank_caption_is_passed_on_as_empty_text(tmp_path):
    seen, _ = _ingest(tmp_path, "user_id,image_path,caption\nu1,a.jpg,\n")

    assert seen == [("u1", "a.jpg", "", None)]


def test_rows_without_a_user_or_an_image_are_skipped(tmp_path):
    seen, db = _ingest(
        tmp_path,
        "user_id,image_path,caption\n"
        ",a.jpg,no user\n"
        "u1,,no image\n"
        "u2,b.jpg,complete\n",
    )

    assert seen == [("u2", "b.jpg", "complete", None)]
    assert db.insert_post.call_count == 1


def test_a_real_caption_is_unchanged(tmp_path):
    seen, _ = _ingest(
        tmp_path, "user_id,image_path,caption,timestamp\nu1,a.jpg,hello there,2026-01-01T10:00:00\n"
    )

    assert seen == [("u1", "a.jpg", "hello there", "2026-01-01T10:00:00")]
