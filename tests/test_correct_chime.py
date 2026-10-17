"""Tests for POST /dashboard/correct_chime."""

from unittest.mock import MagicMock, patch

from bson.objectid import ObjectId

USER_ID = '5f0000000000000000000001'
POST_ID = '5f0000000000000000000002'
FL_TRIGGER = 'dreams_app.app.dashboard.main._maybe_trigger_fl_training'


def _log_in(client, mock_mongo):
    mock_mongo.users.find_one.return_value = {
        '_id': USER_ID,
        'username': 'alice',
        'email': 'alice@example.com',
        'password_hash': 'unused',
    }
    with client.session_transaction() as session:
        session['_user_id'] = USER_ID


def _posts(mock_mongo):
    """The mocked 'posts' collection used by the route."""
    posts = mock_mongo.__getitem__.return_value
    posts.count_documents.return_value = 0
    return posts


def _correct(client, **overrides):
    payload = {'post_id': POST_ID, 'corrected_label': 'Hope'}
    payload.update(overrides)
    return client.post('/dashboard/correct_chime', json=payload)


def test_correct_chime_requires_login(client):
    response = _correct(client)

    assert response.status_code == 302
    assert '/auth/login' in response.headers['Location']


def test_correct_chime_rejects_missing_fields(client, mock_mongo):
    _log_in(client, mock_mongo)

    response = client.post('/dashboard/correct_chime', json={'post_id': POST_ID})

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Missing fields'


def test_correct_chime_rejects_invalid_post_id(client, mock_mongo):
    _log_in(client, mock_mongo)

    response = _correct(client, post_id='not-an-object-id')

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Invalid post ID format'


def test_correct_chime_rejects_unknown_label(client, mock_mongo):
    _log_in(client, mock_mongo)

    response = _correct(client, corrected_label='Happiness')

    assert response.status_code == 400
    assert response.get_json()['error'] == 'Invalid label value'


def test_correct_chime_rate_limit(client, mock_mongo):
    _log_in(client, mock_mongo)
    posts = _posts(mock_mongo)
    posts.count_documents.return_value = 10

    response = _correct(client)

    assert response.status_code == 429
    posts.update_one.assert_not_called()


def test_correct_chime_saves_correction_for_own_post(client, mock_mongo):
    _log_in(client, mock_mongo)
    posts = _posts(mock_mongo)
    posts.update_one.return_value = MagicMock(modified_count=1)

    with patch(FL_TRIGGER) as trigger:
        response = _correct(client)

    assert response.status_code == 200
    assert response.get_json() == {'success': True}

    posts.update_one.assert_called_once()
    query, update = posts.update_one.call_args[0]
    # The update is limited to the logged-in user's own post.
    assert query == {'_id': ObjectId(POST_ID), 'user_id': USER_ID}
    assert update['$set']['corrected_label'] == 'Hope'
    assert update['$set']['is_fl_processed'] is False
    assert update['$push']['correction_history']['label'] == 'Hope'
    trigger.assert_called_once()


def test_correct_chime_returns_404_when_nothing_changed(client, mock_mongo):
    _log_in(client, mock_mongo)
    posts = _posts(mock_mongo)
    posts.update_one.return_value = MagicMock(modified_count=0)

    with patch(FL_TRIGGER) as trigger:
        response = _correct(client)

    assert response.status_code == 404
    trigger.assert_not_called()
