# DREAMS Test Suite

## Overview

This directory contains the test suite for the DREAMS platform. Tests are organized by module and use pytest as the test framework.

## Setup

### Install Test Dependencies

```bash
# Install all test dependencies
pip install -r requirements-dev.txt

# Or install just the core requirements plus test tools
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock
```

### Verify Installation

```bash
# Check that pytest can discover tests
pytest --collect-only

# Should show ~92 tests collected
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage Report

```bash
# Terminal output
pytest tests/ -v --cov=dreamsApp --cov-report=term-missing

# HTML report (opens in browser)
pytest tests/ -v --cov=dreamsApp --cov-report=html
python -m webbrowser htmlcov/index.html
```

### Run Specific Test Files

```bash
# Single file
pytest tests/test_sentiment.py -v

# Multiple files
pytest tests/test_sentiment.py tests/test_clustering.py -v

# Specific test function
pytest tests/test_sentiment.py::test_valid_caption -v
```

### Run Tests by Marker

```bash
# Run only unit tests (if markers are added)
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures (Flask app, MongoDB mocks)
├── README.md                # This file
├── data/                    # Test data files
│   ├── expected_results.json
│   ├── locations.json
│   └── sentiments.csv
├── test_*.py                # Test modules (one per feature)
└── integration/             # Integration tests (future)
```

## Test Categories

### Unit Tests
- `test_sentiment.py` - Sentiment analysis API
- `test_clustering.py` - Keyword clustering
- `test_location_enrichment.py` - Location geocoding and embedding
- `test_temporal_narrative_graph.py` - Temporal graph construction
- `test_graph_analysis.py` - Graph metrics computation
- `test_timeline.py` - Emotion timeline utilities

### Integration Tests
- `test_graph_metrics_api.py` - Full API endpoint testing
- `test_fl.py` - Federated learning workflow
- `test_chime.py` - CHIME model integration

## Writing Tests

### Using Fixtures

The `conftest.py` file provides shared fixtures:

```python
def test_with_flask_app(app):
    """Test that needs Flask app instance."""
    assert app.config['TESTING'] is True

def test_with_client(client):
    """Test that makes HTTP requests."""
    response = client.get('/api/some-endpoint')
    assert response.status_code == 200

def test_with_context(app_context):
    """Test that needs application context."""
    from flask import current_app
    assert current_app.config['TESTING'] is True

def test_with_mock_db(mock_mongo):
    """Test that uses mocked MongoDB."""
    mock_mongo['posts'].find_one.return_value = {'_id': '123'}
    # test code here
```

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test functions: `test_<what_it_tests>()`
- Test classes: `Test<FeatureName>`

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

@patch('dreamsApp.app.utils.sentiment.pipeline')
def test_with_mocked_model(mock_pipeline):
    """Mock HuggingFace pipeline to avoid loading models."""
    mock_pipeline.return_value = MagicMock()
    # test code here
```

## Coverage Goals

| Module | Current | Target |
|--------|---------|--------|
| `app/utils/sentiment.py` | 78% | 90% |
| `app/utils/clustering.py` | 65% | 85% |
| `analytics/graph_analysis.py` | 95% | 95% |
| `analytics/temporal_narrative_graph.py` | 98% | 98% |
| **Overall** | 78% | 85% |

## Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions.

See `.github/workflows/test.yml` for the CI configuration.

### CI Workflow

1. Checkout code
2. Set up Python 3.10
3. Install dependencies (`requirements.txt` + `requirements-dev.txt`)
4. Run pytest with coverage
5. Run flake8 for code quality checks

## Troubleshooting

### ModuleNotFoundError: No module named 'flask'

```bash
# Install test dependencies
pip install -r requirements-dev.txt
```

### ModuleNotFoundError: No module named 'bson'

```bash
# Install pymongo with all extras
pip install 'pymongo[srv]>=4.6.0'
```

### Tests fail with MongoDB connection errors

The test suite uses mocked MongoDB by default. If you see connection errors:

1. Check that `conftest.py` is in the `tests/` directory
2. Verify the `mock_mongo` fixture is being used
3. For tests that need real MongoDB, use `pytest.mark.integration` and skip in CI

### Import errors in test files

Make sure you're running pytest from the project root:

```bash
# Correct (from project root)
cd /path/to/DREAMS
pytest tests/

# Incorrect (from tests directory)
cd tests
pytest .  # This will fail with import errors
```

## Test Data

Test data files are located in `tests/data/`:

- `locations.json` - Sample location data for proximity tests
- `sentiments.csv` - Sample sentiment data for timeline tests
- `expected_results.json` - Expected outputs for validation

## Adding New Tests

1. Create a new file: `tests/test_<feature>.py`
2. Import fixtures from `conftest.py`
3. Write test functions with descriptive names
4. Use mocks for external dependencies (APIs, models, databases)
5. Run tests locally before committing
6. Ensure coverage doesn't decrease

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Flask testing documentation](https://flask.palletsprojects.com/en/latest/testing/)
- [DREAMS TEST_PLAN.md](../docs/TEST_PLAN.md) - Comprehensive test strategy

## Questions?

For questions about the test suite, see:
- `docs/TEST_PLAN.md` - Overall testing strategy
- GitHub Discussions - Ask the community
- Open an issue - Report test failures or suggest improvements
