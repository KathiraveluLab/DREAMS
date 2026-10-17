"""Tests for the dashboard profile page."""

USER_ID = '5f0000000000000000000001'


def _log_in(client, mock_mongo):
    mock_mongo.users.find_one.return_value = {
        '_id': USER_ID,
        'username': 'alice',
        'email': 'alice@example.com',
        'password_hash': 'unused',
    }
    with client.session_transaction() as session:
        session['_user_id'] = USER_ID


def test_profile_for_user_without_posts_returns_404(client, mock_mongo):
    _log_in(client, mock_mongo)
    posts = mock_mongo.__getitem__.return_value
    posts.find.return_value.sort.return_value = []

    response = client.get('/dashboard/user/nobody')

    assert response.status_code == 404
    assert 'No posts found' in response.get_data(as_text=True)
