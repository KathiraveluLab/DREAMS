"""
API tests for POST /api/ingest/batch and GET /api/batch/<batch_id>/status
"""
from PIL import Image
import random
import io
import zipfile
import pytest

@pytest.fixture
def make_zip():
    def _make(num_images=3):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for i in range(num_images):
                img = Image.new('RGB', (64, 64))
                pixels = img.load()
                for x in range(64):
                    for y in range(64):
                        pixels[x, y] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                img_buf = io.BytesIO()
                img.save(img_buf, format="JPEG")
                img_buf.seek(0)
                zf.writestr(f"image_{i}.jpg", img_buf.read())
        buf.seek(0)
        return buf.read()
    return _make

@pytest.fixture
def make_csv():
    def _make(num_images=3):
        lines = ["id,user_id,image_filename,category,date,caption"]
        for i in range(num_images):
            lines.append(f"{i},user_batch_test,image_{i}.jpg,vacation,2026-01-01T00:00:00Z,Caption {i}")
        return "\n".join(lines).encode('utf-8')
    return _make

class TestBatchIngest:
    def test_missing_files(self, client):
        rv = client.post("/api/ingest/batch", data={})
        assert rv.status_code == 400
        assert "Missing" in rv.get_json()["error"]

    def test_invalid_zip(self, client, make_csv):
        data = {
            "csv": (io.BytesIO(make_csv()), "test.csv"),
            "images": (io.BytesIO(b"not a zip file"), "test.zip")
        }
        rv = client.post("/api/ingest/batch", data=data, content_type="multipart/form-data")
        assert rv.status_code == 400
        assert "Invalid zip file" in rv.get_json()["error"]

    def test_successful_batch_upload_and_status(self, client, make_zip, make_csv):
        data = {
            "csv": (io.BytesIO(make_csv(3)), "test.csv"),
            "images": (io.BytesIO(make_zip(3)), "test.zip")
        }
        rv = client.post("/api/ingest/batch", data=data, content_type="multipart/form-data")
        assert rv.status_code == 202
        json_data = rv.get_json()
        assert "batch_id" in json_data
        assert json_data["enqueued_count"] == 3
        
        # Test status endpoint
        batch_id = json_data["batch_id"]
        rv2 = client.get(f"/api/batch/{batch_id}/status")
        assert rv2.status_code == 200
        status_data = rv2.get_json()
        assert status_data["batch_id"] == batch_id
        assert status_data["status"]["queued"] == 3
        assert status_data["status"]["total"] == 3
