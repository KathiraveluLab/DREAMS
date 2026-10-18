"""Access control tests for GET /api/run_clustering."""

from unittest.mock import patch

CLUSTER_FN = 'dreams_app.app.ingestion.routes.cluster_keywords_for_all_users'


def test_run_clustering_requires_login(client):
    with patch(CLUSTER_FN) as cluster:
        response = client.get('/api/run_clustering')

    assert response.status_code == 302
    assert '/auth/login' in response.headers['Location']
    cluster.assert_not_called()


def test_run_clustering_runs_for_logged_in_user(client, mock_mongo):
    user_id = '5f0000000000000000000001'
    mock_mongo.users.find_one.return_value = {
        '_id': user_id,
        'username': 'alice',
        'email': 'alice@example.com',
        'password_hash': 'unused',
    }
    with client.session_transaction() as session:
        session['_user_id'] = user_id

    with patch(CLUSTER_FN) as cluster:
        response = client.get('/api/run_clustering')

    assert response.status_code == 200
    assert response.get_data(as_text=True) == 'Clustering done'
    cluster.assert_called_once()
