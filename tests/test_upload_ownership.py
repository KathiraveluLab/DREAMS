import io
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

ROUTES = "dreams_app.app.ingestion.routes"  # actual package path


@pytest.fixture
def user_a():
    """Pretend user A is the authenticated session user."""
    fake_user = MagicMock()
    fake_user.get_id.return_value = "user-a"
    with patch(f"{ROUTES}.current_user", fake_user, create=True):
        yield fake_user


@pytest.fixture
def auth_client(app, user_a):
    # Lets @login_required pass; identity comes from the patched current_user.
    app.config["LOGIN_DISABLED"] = True
    return app.test_client()


@pytest.fixture
def stub_upload_deps(app, mock_mongo):
    """Replace the slow/AI parts so the route runs fast, with or without the fix."""
    pipeline = MagicMock()
    pipeline.process_new_post.return_value = {
        "post_doc": {
            "timestamp": datetime.now(timezone.utc),
            "sentiment": {"label": "neutral"},
        }
    }
    app.dreams_pipeline = pipeline

    insert_result = mock_mongo["posts"].insert_one.return_value
    insert_result.acknowledged = True
    insert_result.inserted_id = "abc123"

    with patch(f"{ROUTES}.get_image_caption", return_value="a caption"), \
         patch(f"{ROUTES}.get_chime_category", return_value=None), \
         patch(f"{ROUTES}.extract_gps_from_image", return_value=None):
        yield pipeline


def make_png():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_upload_rejects_other_users_id(auth_client, mock_mongo, stub_upload_deps):
    resp = auth_client.post(
        "/api/upload",
        data={"caption": "hi", "user_id": "user-b", "image": (make_png(), "x.png")},
        content_type="multipart/form-data",
    )
    print(resp.status_code, stub_upload_deps.process_new_post.call_args)
    assert resp.status_code == 403
    mock_mongo["posts"].insert_one.assert_not_called()


def test_upload_uses_session_user_id(auth_client, stub_upload_deps):
    resp = auth_client.post(
        "/api/upload",
        data={"caption": "hi", "image": (make_png(), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    assert resp.get_json()["user_id"] == "user-a"
    assert stub_upload_deps.process_new_post.call_args.args[0] == "user-a"