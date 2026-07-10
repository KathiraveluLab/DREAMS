# tests/test_graph_metrics_api.py

"""
Integration tests for the ``GET /api/analytics/graph-metrics/<user_id>`` endpoint.

These tests require a running MongoDB instance (or mongomock).  They are marked
with ``@pytest.mark.integration`` so they can be excluded in CI environments
that lack a database:

    pytest -m "not integration"

The unit tests in ``test_graph_analysis.py`` are the primary quality gate for the
analysis logic itself and have zero external dependencies.
"""

import sys
import types
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class _StubModule(types.ModuleType):
    """Module stub that returns MagicMock for any attribute access.

    This allows ``from heavy_lib import SomeClass`` to succeed without
    the actual library installed.
    """

    def __getattr__(self, name: str):
        return MagicMock()


def _ensure_stub_modules():
    """Inject lightweight stubs for heavy dependencies that are not
    installed in the current environment.  Only truly-missing modules
    are stubbed — installed ones are left untouched."""
    import importlib
    candidates = [
        'hdbscan', 'setfit', 'wordcloud',
        'torch', 'transformers', 'sentence_transformers',
        'spacy', 'scipy', 'scipy.special',
        'google.genai', 'google.genai.types',
        'PIL', 'PIL.Image', 'PIL.ExifTags',
    ]
    for name in candidates:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            sys.modules[name] = _StubModule(name)


_ensure_stub_modules()

from dreams_app.app import create_app  # noqa: E402  (after stubs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path):
    """Create a Flask app configured for testing with mocked MongoDB."""
    app = create_app(test_config={
        'TESTING': True,
        'SECRET_KEY': 'test-secret',
        'MONGO_URI': 'mongodb://localhost:27017',
        'MONGO_DB_NAME': 'dreams_test',
        'UPLOAD_FOLDER': str(tmp_path / 'uploads'),
    })
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def _make_mock_posts(user_id: str, count: int = 5):
    """Generate a list of fake post documents mimicking MongoDB schema."""
    base = datetime(2024, 6, 1, 8, 0, 0)
    labels = ["positive", "negative", "neutral"]
    posts = []
    for i in range(count):
        from datetime import timedelta
        posts.append({
            '_id': f'fake_id_{i}',
            'user_id': user_id,
            'caption': f'test caption {i}',
            'timestamp': base + timedelta(hours=i),
            'sentiment': {
                'label': labels[i % len(labels)],
                'score': 0.85,
            },
        })
    return posts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGraphMetricsEndpoint:
    """Tests that exercise the full endpoint through Flask's test client.

    MongoDB calls are mocked to keep the tests self-contained.
    """

    def _login(self, client):
        """Skip login_required by disabling it in the test."""
        # We patch flask_login.utils._get_user so @login_required passes.
        pass  # handled via the login_disabled context below

    def _get_with_auth(self, client, url):
        """Issue a GET request with login_required bypassed."""
        return client.get(url)

    def test_missing_user_returns_404(self, app, client):
        mock_collection = MagicMock()
        mock_find = MagicMock()
        mock_find.sort = MagicMock(return_value=[])
        mock_collection.find = MagicMock(return_value=mock_find)

        with app.app_context():
            app.mongo = {'posts': mock_collection}
            # Bypass login_required
            app.config['LOGIN_DISABLED'] = True

            resp = client.get('/api/analytics/graph-metrics/nonexistent_user')
            assert resp.status_code == 404
            data = resp.get_json()
            assert data['error'] == 'No posts found for user'
            assert data['user_id'] == 'nonexistent_user'

    def test_valid_user_returns_200(self, app, client):
        user_id = 'test_user_001'
        posts = _make_mock_posts(user_id, count=5)

        mock_find = MagicMock()
        mock_find.sort = MagicMock(return_value=posts)
        mock_collection = MagicMock()
        mock_collection.find = MagicMock(return_value=mock_find)

        with app.app_context():
            app.mongo = {'posts': mock_collection}
            app.config['LOGIN_DISABLED'] = True

            resp = client.get(f'/api/analytics/graph-metrics/{user_id}')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['user_id'] == user_id
            assert 'metrics' in data

    def test_response_has_all_sections(self, app, client):
        user_id = 'test_user_002'
        posts = _make_mock_posts(user_id, count=6)

        mock_find = MagicMock()
        mock_find.sort = MagicMock(return_value=posts)
        mock_collection = MagicMock()
        mock_collection.find = MagicMock(return_value=mock_find)

        with app.app_context():
            app.mongo = {'posts': mock_collection}
            app.config['LOGIN_DISABLED'] = True

            resp = client.get(f'/api/analytics/graph-metrics/{user_id}')
            assert resp.status_code == 200
            metrics = resp.get_json()['metrics']

            assert 'graph_summary' in metrics
            assert 'node_metrics' in metrics
            assert 'pattern_analysis' in metrics

            summary = metrics['graph_summary']
            assert 'node_count' in summary
            assert 'edge_count' in summary
            assert 'density' in summary
            assert 'connected_components' in summary
            assert 'is_dag' in summary

            pattern = metrics['pattern_analysis']
            assert 'common_transitions' in pattern
            assert 'emotional_cycles' in pattern
            assert 'label_distribution' in pattern

            # Verify the edges list is returned
            assert 'edges' in metrics
            assert isinstance(metrics['edges'], list)

    def test_unauthenticated_redirects_to_login(self, app, client):
        """API endpoint requires login — unauthenticated requests are redirected."""
        with app.app_context():
            app.config['LOGIN_DISABLED'] = False

            resp = client.get('/api/analytics/graph-metrics/some_user')
            # @login_required redirects to the login page
            assert resp.status_code == 302

    def test_invalid_user_id_returns_400(self, app, client):
        """User IDs with special characters or excessive length are rejected."""
        with app.app_context():
            app.config['LOGIN_DISABLED'] = True
            # Script injection attempt
            resp = client.get('/api/analytics/graph-metrics/<script>alert(1)</script>')
            assert resp.status_code in (400, 404)  # 404 from Flask if / in path

        with app.app_context():
            app.config['LOGIN_DISABLED'] = True
            # Excessively long user_id
            long_id = 'a' * 100
            resp = client.get(f'/api/analytics/graph-metrics/{long_id}')
            assert resp.status_code == 400
            data = resp.get_json()
            assert data['error'] == 'Invalid user_id format'


@pytest.mark.integration
class TestNarrativeDashboardRoute:
    """Tests for the narrative visualization page route."""

    def test_narrative_page_returns_200(self, app, client):
        with app.app_context():
            app.config['LOGIN_DISABLED'] = True
            resp = client.get('/dashboard/user/test_user/narrative')
            assert resp.status_code == 200

    def test_narrative_page_contains_d3(self, app, client):
        with app.app_context():
            app.config['LOGIN_DISABLED'] = True
            resp = client.get('/dashboard/user/test_user/narrative')
            html = resp.data.decode('utf-8')
            assert 'd3.min.js' in html
            assert 'chart.js' in html or 'chart.umd.min.js' in html

    def test_narrative_page_contains_graph_container(self, app, client):
        with app.app_context():
            app.config['LOGIN_DISABLED'] = True
            resp = client.get('/dashboard/user/test_user/narrative')
            html = resp.data.decode('utf-8')
            assert 'narrative-graph' in html
            assert 'centralityChart' in html
            assert 'distributionChart' in html

    def test_narrative_page_has_user_id(self, app, client):
        with app.app_context():
            app.config['LOGIN_DISABLED'] = True
            resp = client.get('/dashboard/user/my_user_42/narrative')
            html = resp.data.decode('utf-8')
            assert 'my_user_42' in html
