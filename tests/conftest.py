"""
Shared pytest fixtures for DREAMS test suite.

This module provides reusable fixtures for Flask app testing,
MongoDB mocking, and common test utilities.
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from dreamsApp.app import create_app


@pytest.fixture
def app():
    """
    Create and configure a Flask application instance for testing.
    
    Returns a Flask app with:
    - TESTING mode enabled
    - Temporary upload folder
    - Mocked MongoDB connection
    - All blueprints registered
    
    Usage:
        def test_something(app):
            with app.app_context():
                # test code here
    """
    # Create temporary directory for uploads during tests
    temp_dir = tempfile.mkdtemp()
    
    test_config = {
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key',
        'UPLOAD_FOLDER': temp_dir,
        'MONGO_URI': 'mongodb://localhost:27017',
        'MONGO_DB_NAME': 'dreams_test',
    }
    
    with patch('dreamsApp.app.DreamsPipeline'), \
         patch('dreamsApp.app.MongoClient'):
        app = create_app(test_config=test_config)
        
        # Mock MongoDB to avoid requiring a running MongoDB instance
        # Individual tests can override this if they need real DB access
        mock_mongo = MagicMock()
        app.mongo = mock_mongo
        
        yield app
    
    # Cleanup: remove temporary upload directory
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def client(app):
    """
    Create a test client for making HTTP requests to the Flask app.
    
    Usage:
        def test_endpoint(client):
            response = client.get('/some/endpoint')
            assert response.status_code == 200
    """
    return app.test_client()


@pytest.fixture
def app_context(app):
    """
    Provide an application context for tests that need it.
    
    Some operations (like accessing current_app or g) require
    an active application context. This fixture provides that.
    
    Usage:
        def test_with_context(app_context):
            from flask import current_app
            # current_app is now available
    """
    with app.app_context():
        yield app


@pytest.fixture
def runner(app):
    """
    Create a CLI test runner for testing Flask CLI commands.
    
    Usage:
        def test_cli_command(runner):
            result = runner.invoke(args=['some-command'])
            assert result.exit_code == 0
    """
    return app.test_cli_runner()


@pytest.fixture
def mock_mongo(app):
    """
    Provide access to the mocked MongoDB instance.
    
    Useful for setting up test data or verifying database calls.
    
    Usage:
        def test_db_operation(app, mock_mongo):
            mock_mongo['posts'].find_one.return_value = {'_id': '123'}
            # test code that uses the database
    """
    return app.mongo
