"""Tests for the authentication routes in dreams_app/app/auth.py."""

from werkzeug.security import check_password_hash, generate_password_hash


def _flashed_messages(client):
    """Return the flash messages currently stored in the client's session."""
    with client.session_transaction() as session:
        return [message for _, message in session.get('_flashes', [])]


def _register_form(**overrides):
    form = {
        'username': 'alice',
        'email': 'alice@example.com',
        'password': 'correct-horse',
    }
    form.update(overrides)
    return form


# --- /auth/register ---------------------------------------------------------

def test_register_with_missing_fields_redirects_back(client, mock_mongo):
    # The password is left out on purpose.
    response = client.post(
        '/auth/register',
        data={'username': 'alice', 'email': 'alice@example.com'},
    )

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/auth/register')
    assert 'Please fill in all fields.' in _flashed_messages(client)
    mock_mongo.users.insert_one.assert_not_called()


def test_register_with_existing_username_is_rejected(client, mock_mongo):
    # The first lookup (by username) finds an existing user.
    mock_mongo.users.find_one.return_value = {'username': 'alice'}

    response = client.post('/auth/register', data=_register_form())

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/auth/register')
    assert 'Please use a different username.' in _flashed_messages(client)
    mock_mongo.users.insert_one.assert_not_called()


def test_register_with_existing_email_is_rejected(client, mock_mongo):
    # Username lookup finds nobody, email lookup finds an existing user.
    mock_mongo.users.find_one.side_effect = [None, {'email': 'alice@example.com'}]

    response = client.post('/auth/register', data=_register_form())

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/auth/register')
    assert 'Please use a different email address.' in _flashed_messages(client)
    mock_mongo.users.insert_one.assert_not_called()


def test_register_success_stores_hashed_password(client, mock_mongo):
    mock_mongo.users.find_one.return_value = None

    response = client.post('/auth/register', data=_register_form())

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/auth/login')

    mock_mongo.users.insert_one.assert_called_once()
    stored = mock_mongo.users.insert_one.call_args[0][0]
    assert stored['username'] == 'alice'
    assert stored['email'] == 'alice@example.com'
    # The plain password must never be stored.
    assert stored['password_hash'] != 'correct-horse'
    assert check_password_hash(stored['password_hash'], 'correct-horse')


# --- /auth/login ------------------------------------------------------------

def test_login_with_unknown_username_is_rejected(client, mock_mongo):
    mock_mongo.users.find_one.return_value = None

    response = client.post(
        '/auth/login',
        data={'username': 'nobody', 'password': 'whatever'},
    )

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/auth/login')
    assert 'Invalid username or password' in _flashed_messages(client)


def test_login_with_wrong_password_is_rejected(client, mock_mongo):
    mock_mongo.users.find_one.return_value = {
        '_id': 'abc123',
        'username': 'alice',
        'email': 'alice@example.com',
        'password_hash': generate_password_hash('correct-horse'),
    }

    response = client.post(
        '/auth/login',
        data={'username': 'alice', 'password': 'wrong-password'},
    )

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/auth/login')
    assert 'Invalid username or password' in _flashed_messages(client)
    with client.session_transaction() as session:
        assert '_user_id' not in session


def test_login_success_logs_user_in_and_redirects_to_dashboard(client, mock_mongo):
    mock_mongo.users.find_one.return_value = {
        '_id': 'abc123',
        'username': 'alice',
        'email': 'alice@example.com',
        'password_hash': generate_password_hash('correct-horse'),
    }

    response = client.post(
        '/auth/login',
        data={'username': 'alice', 'password': 'correct-horse'},
    )

    assert response.status_code == 302
    assert response.headers['Location'].endswith('/dashboard/')
    with client.session_transaction() as session:
        assert session['_user_id'] == 'abc123'


# --- /auth/logout -----------------------------------------------------------

def test_logout_requires_login(client):
    response = client.get('/auth/logout')

    assert response.status_code == 302
    # login_required redirects with ?next=..., a plain logout redirect would not.
    assert '/auth/login?next=' in response.headers['Location']
