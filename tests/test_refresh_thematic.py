"""Tests for POST /dashboard/refresh_thematic/<user_id>.

profile.html shows `data.success` / `data.message` from the response, so every
response has to carry both.
"""

from unittest.mock import patch

USER_ID = '5f0000000000000000000001'
GENERATE = 'dreams_app.app.dashboard.main.generate'


def _log_in(client, mock_mongo):
    mock_mongo.users.find_one.return_value = {
        '_id': USER_ID,
        'username': 'alice',
        'email': 'alice@example.com',
        'password_hash': 'unused',
    }
    with client.session_transaction() as session:
        session['_user_id'] = USER_ID


def _refresh(client):
    return client.post(f'/dashboard/refresh_thematic/{USER_ID}')


def test_refresh_thematic_requires_login(client):
    response = _refresh(client)

    assert response.status_code == 302
    assert '/auth/login' in response.headers['Location']


def test_refresh_thematic_success_reports_success(client, mock_mongo):
    _log_in(client, mock_mongo)
    mock_mongo.__getitem__.return_value.find_one.return_value = {
        'positive_keywords': [{'keyword': 'garden'}],
        'negative_keywords': [{'keyword': 'rain'}],
    }

    with patch(GENERATE, return_value={'themes': ['nature']}) as generate:
        response = _refresh(client)

    assert response.status_code == 200
    body = response.get_json()
    assert body['success'] is True
    assert body['thematic_data'] == {'themes': ['nature']}
    assert generate.call_args[0][1:3] == (['garden'], ['rain'])


def test_refresh_thematic_generate_failure_reports_failure(client, mock_mongo):
    _log_in(client, mock_mongo)
    mock_mongo.__getitem__.return_value.find_one.return_value = None

    with patch(GENERATE, side_effect=RuntimeError('llm down')):
        response = _refresh(client)

    assert response.status_code == 500
    body = response.get_json()
    assert body['success'] is False
    assert body['message'] == 'Failed to refresh thematics'


def test_refresh_thematic_does_not_leak_internal_errors(client, mock_mongo):
    _log_in(client, mock_mongo)
    secret = 'mongodb://admin:hunter2@db.internal:27017 refused'
    mock_mongo.__getitem__.return_value.find_one.side_effect = RuntimeError(secret)

    with patch(GENERATE, return_value={}):
        response = _refresh(client)

    assert response.status_code == 500
    body = response.get_json()
    assert body['success'] is False
    assert 'hunter2' not in response.get_data(as_text=True)


def test_refresh_thematic_without_llm_reports_failure(client, mock_mongo):
    _log_in(client, mock_mongo)

    with patch(GENERATE, None):
        response = _refresh(client)

    assert response.status_code == 500
    body = response.get_json()
    assert body['success'] is False
    assert body['message'] == 'LLM dependency is not available'
