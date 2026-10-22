"""The CHIME classifier must follow the federated-learning model on disk.

fl_worker promotes a retrained model by swapping the `production_chime_model` folder.
Predictions have to pick that model up without restarting the server.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from dreams_app.core.sentiment import HF_MODEL_ID, SentimentAnalyzer

PIPELINE = 'dreams_app.core.sentiment.pipeline'


@pytest.fixture
def app(tmp_path):
    return Flask(__name__, root_path=str(tmp_path))


def _production_dir(tmp_path):
    return tmp_path / 'models' / 'production_chime_model'


def _make_pipeline():
    """A fake transformers.pipeline that records the model path of every classifier it builds."""
    loaded = []

    def factory(task, model=None, tokenizer=None, top_k=None):
        loaded.append(model)
        classifier = MagicMock()
        classifier.return_value = [[{'label': 'Hope', 'score': 0.9}]]
        return classifier

    return factory, loaded


def _swap_in_new_model(tmp_path):
    """Promote a new model the way fl_worker does: build it in a temp folder, then rename it."""
    temp = tmp_path / 'models' / 'temp_training_artifact'
    temp.mkdir(parents=True)
    production = _production_dir(tmp_path)
    if production.exists():
        production.rename(str(production) + '_backup')
    os.rename(temp, production)


def test_uses_the_base_model_until_a_federated_model_exists(app):
    factory, loaded = _make_pipeline()
    with patch(PIPELINE, side_effect=factory), app.app_context():
        analyzer = SentimentAnalyzer()
        analyzer.analyze_chime('I feel hopeful.')
        analyzer.analyze_chime('Still hopeful.')

    assert loaded == [HF_MODEL_ID]


def test_loads_the_federated_model_once_it_is_promoted(app, tmp_path):
    factory, loaded = _make_pipeline()
    with patch(PIPELINE, side_effect=factory), app.app_context():
        analyzer = SentimentAnalyzer()
        analyzer.analyze_chime('I feel hopeful.')

        _swap_in_new_model(tmp_path)
        analyzer.analyze_chime('I feel hopeful again.')

    assert loaded == [HF_MODEL_ID, str(_production_dir(tmp_path))]


def test_does_not_reload_while_the_model_is_unchanged(app, tmp_path):
    _swap_in_new_model(tmp_path)
    factory, loaded = _make_pipeline()
    with patch(PIPELINE, side_effect=factory), app.app_context():
        analyzer = SentimentAnalyzer()
        for _ in range(3):
            analyzer.analyze_chime('I feel hopeful.')

    assert loaded == [str(_production_dir(tmp_path))]


def test_loads_each_new_promotion(app, tmp_path):
    _swap_in_new_model(tmp_path)
    factory, loaded = _make_pipeline()
    with patch(PIPELINE, side_effect=factory), app.app_context():
        analyzer = SentimentAnalyzer()
        analyzer.analyze_chime('first')

        _swap_in_new_model(tmp_path)
        analyzer.analyze_chime('second')

    production = str(_production_dir(tmp_path))
    assert loaded == [production, production]


def test_falls_back_to_the_base_model_if_the_federated_model_is_removed(app, tmp_path):
    _swap_in_new_model(tmp_path)
    factory, loaded = _make_pipeline()
    with patch(PIPELINE, side_effect=factory), app.app_context():
        analyzer = SentimentAnalyzer()
        analyzer.analyze_chime('first')

        _production_dir(tmp_path).rmdir()
        analyzer.analyze_chime('second')

    assert loaded == [str(_production_dir(tmp_path)), HF_MODEL_ID]
