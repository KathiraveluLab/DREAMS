"""
API tests for:
  GET /api/status/<job_id>
  GET /api/analysis/<memory_id>
  GET /api/analysis  (paginated list)
  GET /             (health check)

These tests use the Flask test client.  The background worker is NOT
started, so jobs remain in "queued" state throughout — that's fine,
we're testing the HTTP layer and DB-read logic, not the ML steps.
"""

import io
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def ingest(client, make_jpeg):
    """Fixture: upload one image and return the parsed JSON."""
    def _func(jpeg_bytes=None, filename="img.jpg", user_id="u1", **extra):
        data = {"user_id": user_id, **extra}
        if jpeg_bytes is None:
            jpeg_bytes = make_jpeg()
        data["image"] = (io.BytesIO(jpeg_bytes), filename)
        rv = client.post("/api/ingest", data=data, content_type="multipart/form-data")
        return rv.get_json()
    return _func


# ── GET / (health check) ──────────────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_200(self, client):
        rv = client.get("/")
        assert rv.status_code == 200

    def test_contains_service_name(self, client):
        data = client.get("/").get_json()
        assert "service" in data
        assert "DREAMS" in data["service"]

    def test_worker_active_field_present(self, client):
        data = client.get("/").get_json()
        assert "worker_active" in data


# ── GET /api/status/<job_id> ──────────────────────────────────────────────────

class TestStatus:
    def test_unknown_job_returns_404(self, client):
        rv = client.get("/api/status/nonexistent-job-id")
        assert rv.status_code == 404

    def test_queued_job_returns_200(self, client, ingest):
        result = ingest(filename="status_test.jpg", user_id="s1")
        assert result["status"] == "queued"
        job_id = result["job_id"]

        rv = client.get(f"/api/status/{job_id}")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["job_id"] == job_id
        assert "status" in data


# ── GET /api/analysis/<memory_id> ─────────────────────────────────────────────

class TestAnalysisSingle:
    def test_unknown_memory_id_returns_404(self, client):
        rv = client.get("/api/analysis/totally_unknown_id")
        assert rv.status_code == 404

    def test_known_memory_returns_200(self, client, ingest):
        result = ingest(filename="analysis_test.jpg", user_id="a1")
        mid = result["memory_id"]

        rv = client.get(f"/api/analysis/{mid}")
        assert rv.status_code == 200

    def test_response_has_required_fields(self, client, ingest):
        result = ingest(filename="fields_test.jpg", user_id="f1")
        mid = result["memory_id"]

        data = client.get(f"/api/analysis/{mid}").get_json()
        required = ["memory_id", "user_id", "image_path",
                    "caption", "generated_caption", "category",
                    "captured_at", "is_duplicate", "processing_status",
                    "emotions", "location", "temporal", "embeddings"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_embeddings_included_by_default(self, client, ingest):
        """After our fix, embeddings must appear without ?include_embeddings=true."""
        result = ingest(filename="emb_default.jpg", user_id="e1")
        mid = result["memory_id"]

        data = client.get(f"/api/analysis/{mid}").get_json()
        assert "embeddings" in data
        # The job hasn't been processed yet (worker not running in tests),
        # so ChromaDB may return empty — but the key must be present.
        assert isinstance(data["embeddings"], dict)

    def test_embeddings_suppressed_with_false_param(self, client, ingest):
        """?include_embeddings=false should return placeholder metadata only."""
        result = ingest(filename="emb_suppress.jpg", user_id="e2")
        mid = result["memory_id"]

        data = client.get(f"/api/analysis/{mid}?include_embeddings=false").get_json()
        emb = data["embeddings"]
        # placeholder: {image: {collection, dimensions}, caption: {collection, dimensions}}
        assert "image" in emb
        assert "collection" in emb["image"]
        assert "dimensions" in emb["image"]

    def test_processing_status_pending_before_worker_runs(self, client, ingest):
        result = ingest(filename="pending_test.jpg", user_id="p1")
        mid = result["memory_id"]

        data = client.get(f"/api/analysis/{mid}").get_json()
        assert data["processing_status"] == "pending"

    def test_caption_source_field(self, client, ingest):
        result = ingest(filename="captioned.jpg", user_id="c1",
                         caption="My holiday photo")
        mid = result["memory_id"]

        data = client.get(f"/api/analysis/{mid}").get_json()
        assert data["caption_source"] == "user"

    def test_no_caption_source_is_generated(self, client, ingest):
        result = ingest(filename="nocaption.jpg", user_id="nc1")
        mid = result["memory_id"]

        data = client.get(f"/api/analysis/{mid}").get_json()
        assert data["caption_source"] == "generated"


# ── GET /api/analysis (list) ──────────────────────────────────────────────────

class TestAnalysisList:
    def test_returns_paginated_structure(self, client):
        rv = client.get("/api/analysis")
        assert rv.status_code == 200
        data = rv.get_json()
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert "records" in data
        assert isinstance(data["records"], list)

    def test_default_page_is_1(self, client):
        data = client.get("/api/analysis").get_json()
        assert data["page"] == 1

    def test_per_page_capped_at_100(self, client):
        data = client.get("/api/analysis?per_page=9999").get_json()
        assert data["per_page"] <= 100

    def test_user_id_filter(self, client, ingest):
        # Upload one memory for a unique user
        ingest(filename="list_filter.jpg", user_id="unique_list_user")

        data = client.get("/api/analysis?user_id=unique_list_user").get_json()
        assert data["total"] >= 1
        for record in data["records"]:
            assert record["user_id"] == "unique_list_user"

    def test_records_do_not_contain_vectors_in_list(self, client, ingest):
        """List endpoint always uses include_embeddings=False (no vectors)."""
        ingest(filename="list_novec.jpg", user_id="lv1")
        data = client.get("/api/analysis?user_id=lv1").get_json()
        if data["records"]:
            emb = data["records"][0].get("embeddings", {})
            # Should be placeholder metadata, not actual vector list
            img_emb = emb.get("image", {})
            assert "collection" in img_emb
            assert "vector" not in img_emb
