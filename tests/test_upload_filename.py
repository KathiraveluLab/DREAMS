"""Tests for filename handling in POST /api/upload."""

import io
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from PIL import Image

USER_ID = '5f0000000000000000000001'
ROUTES = 'dreams_app.app.ingestion.routes'


def _png():
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8)).save(buffer, 'PNG')
    buffer.seek(0)
    return buffer


def _log_in(client, mock_mongo):
    mock_mongo.users.find_one.return_value = {
        '_id': USER_ID,
        'username': 'alice',
        'email': 'alice@example.com',
        'password_hash': 'unused',
    }
    with client.session_transaction() as session:
        session['_user_id'] = USER_ID


def _upload(client, app, filename):
    app.dreams_pipeline.process_new_post.return_value = {
        'post_doc': {
            'timestamp': datetime.now(timezone.utc),
            'sentiment': {'label': 'neutral'},
        }
    }
    with patch(f'{ROUTES}.get_image_caption', return_value='a park'), \
         patch(f'{ROUTES}.get_chime_category', return_value=None), \
         patch(f'{ROUTES}.extract_gps_from_image', return_value=None):
        return client.post(
            '/api/upload',
            data={
                'user_id': 'alice',
                'caption': 'a walk in the park',
                'image': (_png(), filename),
            },
            content_type='multipart/form-data',
        )


@pytest.mark.parametrize('filename, suffix', [
    ('фото.png', '.png'),
    ('照片.JPG', '.JPG'),
])
def test_upload_accepts_valid_image_with_non_ascii_filename(
        client, app, mock_mongo, filename, suffix):
    _log_in(client, mock_mongo)

    response = _upload(client, app, filename)

    assert response.status_code == 201, response.get_json()
    saved = os.listdir(app.config['UPLOAD_FOLDER'])
    assert len(saved) == 1
    assert saved[0].endswith(suffix)


def test_upload_still_rejects_unsupported_extension(client, app, mock_mongo):
    _log_in(client, mock_mongo)

    response = _upload(client, app, 'фото.exe')

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Unsupported file extension'
    assert os.listdir(app.config['UPLOAD_FOLDER']) == []


def test_upload_keeps_ascii_filename(client, app, mock_mongo):
    _log_in(client, mock_mongo)

    response = _upload(client, app, '../my photo.png')

    assert response.status_code == 201
    saved = os.listdir(app.config['UPLOAD_FOLDER'])
    assert len(saved) == 1
    assert saved[0].endswith('_my_photo.png')
