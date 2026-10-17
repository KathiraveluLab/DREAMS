"""Tests for timestamp handling in POST /api/upload."""

import io
import os

from PIL import Image

USER_ID = '5f0000000000000000000001'


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


def test_upload_with_invalid_timestamp_is_rejected_before_saving(client, app, mock_mongo):
    _log_in(client, mock_mongo)

    response = client.post(
        '/api/upload',
        data={
            'user_id': 'alice',
            'caption': 'a walk in the park',
            'timestamp': 'yesterday',
            'image': (_png(), 'photo.png'),
        },
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert 'timestamp' in response.get_json()['error'].lower()
    # Nothing was written to disk and the pipeline never ran.
    assert os.listdir(app.config['UPLOAD_FOLDER']) == []
    app.dreams_pipeline.process_new_post.assert_not_called()
