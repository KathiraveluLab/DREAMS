"""Tests for oversized images in POST /api/upload."""

import io
import os

from PIL import Image

from tests.test_upload_timestamp import _log_in


def _huge_blank_png():
    # A blank 1-bit PNG compresses to a few KB but declares more than twice
    # Pillow's own decompression-bomb limit (2 * 89,478,485 pixels).
    side = 14_000
    buffer = io.BytesIO()
    Image.new('1', (side, side)).save(buffer, 'PNG')
    buffer.seek(0)
    return buffer


def test_upload_of_decompression_bomb_is_rejected_with_400(client, app, mock_mongo):
    _log_in(client, mock_mongo)

    response = client.post(
        '/api/upload',
        data={
            'user_id': 'alice',
            'caption': 'a walk in the park',
            'image': (_huge_blank_png(), 'photo.png'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert 'not a valid image' in response.get_json()['error']
    assert os.listdir(app.config['UPLOAD_FOLDER']) == []
    app.dreams_pipeline.process_new_post.assert_not_called()
