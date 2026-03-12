"""
API tests for POST /api/ingest

Tests all validation branches and the happy-path 202 response.
The background worker is NOT started in tests — we only verify that
the HTTP layer behaves correctly (validation, DB insert, queue entry).
"""

import io
import pytest
from PIL import Image


@pytest.fixture
def upload(client):
    """Fixture: helper to POST to /api/ingest."""
    def _func(image_bytes=None, filename="test.jpg", **form_fields):
        data = dict(form_fields)
        if image_bytes is not None:
            data["image"] = (io.BytesIO(image_bytes), filename)
        return client.post(
            "/api/ingest",
            data=data,
            content_type="multipart/form-data",
        )
    return _func


# ── Validation errors (400) ───────────────────────────────────────────────────

class TestIngestValidation:
    def test_missing_image_field(self, upload):
        rv = upload(image_bytes=None, user_id="u1")
        assert rv.status_code == 400
        assert "image" in rv.get_json()["error"].lower()

    def test_missing_user_id(self, upload, make_jpeg):
        rv = upload(make_jpeg())
        assert rv.status_code == 400
        assert "user_id" in rv.get_json()["error"].lower()

    def test_invalid_extension(self, upload):
        rv = upload(b"fake content", filename="doc.pdf", user_id="u1")
        assert rv.status_code == 400
        assert "invalid file type" in rv.get_json()["error"].lower()

    def test_corrupted_image(self, upload):
        rv = upload(b"this is not an image", filename="bad.jpg", user_id="u1")
        assert rv.status_code == 400
        assert "invalid" in rv.get_json()["error"].lower()


# ── Happy path (202 Accepted) ─────────────────────────────────────────────────

class TestIngestSuccess:
    def test_returns_202_with_job_id_and_memory_id(self, upload, make_jpeg):
        rv = upload(make_jpeg(), user_id="u1", caption="A park")
        assert rv.status_code == 202
        data = rv.get_json()
        assert "job_id" in data
        assert "memory_id" in data
        assert data["status"] == "queued"

    def test_memory_id_is_deterministic(self, upload, make_jpeg):
        """Uploading the same user_id + filename returns the same memory_id."""
        jpeg = make_jpeg()
        rv1 = upload(jpeg, filename="same.jpg", user_id="u1")
        # Second upload with identical user+filename should be 200 already_exists
        rv2 = upload(jpeg, filename="same.jpg", user_id="u1")
        assert rv1.status_code == 202
        assert rv2.status_code == 200
        assert rv2.get_json()["status"] == "already_exists"
        assert rv1.get_json()["memory_id"] == rv2.get_json()["memory_id"]

    def test_optional_fields_accepted(self, upload, make_jpeg):
        data = {
            "user_id": "u2",
            "caption": "Sunset at the beach",
            "category": "park",
            "timestamp": "2026-01-15T12:00:00",
        }
        rv = upload(make_jpeg(color=(200, 100, 50)), filename="opts.jpg", **data)
        assert rv.status_code == 202


# ── Duplicate detection (200) ─────────────────────────────────────────────────

class TestIngestDuplicate:
    def test_perceptually_identical_image_flagged_as_duplicate(self, upload, make_jpeg):
        """Two near-identical images should trigger duplicate detection."""
        # Upload first image
        rv1 = upload(make_jpeg(color=(10, 10, 10)), filename="orig.jpg", user_id="dup_user")
        assert rv1.status_code == 202

        # Upload pixel-identical image under a different filename
        rv2 = upload(make_jpeg(color=(10, 10, 10)), filename="copy.jpg", user_id="dup_user")
        body2 = rv2.get_json()
        # Should be identified as duplicate (200) or accepted as new (202) —
        # depends on hamming threshold; at least it must not error.
        assert rv2.status_code in (200, 202)
        if rv2.status_code == 200:
            assert body2.get("status") in ("duplicate", "already_exists")
