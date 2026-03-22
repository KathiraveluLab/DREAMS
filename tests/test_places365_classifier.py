"""
Tests for Places365 scene classification module.

Uses mocked model outputs so tests run without downloading the
Places365 weights (~100 MB). Run with:
    python -m pytest tests/test_places365_classifier.py -v -s
"""

import pytest
import os
import sys
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import tempfile
import torch

# Import the classifier directly to avoid triggering the full Flask app init
# chain (which requires flask_login, MongoDB, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamsApp", "app", "utils"))
import places365_classifier  # noqa: E402


# --- Fixtures ---

@pytest.fixture
def sample_image_path():
    """Create a temporary test image and return its path."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(f, format="JPEG")
        yield f.name
    os.unlink(f.name)


def _make_mock_output(top_label_index, confidence, labels_list):
    """
    Build a fake model output tensor where `top_label_index` gets
    `confidence` and the rest is spread uniformly.

    Args:
        top_label_index: Index of the label that should have highest confidence.
        confidence: Desired softmax probability for the top label.
        labels_list: Full list of label strings (length 365).

    Returns:
        torch.Tensor: A 1x365 logit tensor whose softmax peaks at top_label_index.
    """
    n = len(labels_list)
    # Start with uniform small logits
    logits = torch.zeros(1, n)
    # Set one logit high enough to dominate after softmax
    logits[0, top_label_index] = 10.0 if confidence >= 0.4 else 1.0
    return logits


def _build_mock_labels():
    """Build a list of 365 fake labels, including several known DREAMS-mapped ones."""
    labels = [f"unlabeled_scene_{i}" for i in range(365)]
    # Place known labels at specific indices for testing
    labels[0] = "hospital_room"         # clinical_or_institutional
    labels[1] = "church/indoor"         # faith_community
    labels[2] = "park"                  # outdoor_or_wilderness
    labels[3] = "bedroom"              # residential_or_transitional
    labels[4] = "dormitory"            # shelter_or_dropin
    labels[5] = "community_center"     # recovery_support
    return labels


MOCK_LABELS = _build_mock_labels()


# --- Test Cases ---

@pytest.mark.parametrize("top_idx,expected_scene_type", [
    (0, "clinical_or_institutional"),   # hospital_room
    (1, "faith_community"),             # church/indoor
    (2, "outdoor_or_wilderness"),       # park
    (3, "residential_or_transitional"), # bedroom
    (4, "shelter_or_dropin"),           # dormitory
    (5, "recovery_support"),            # community_center
])
def test_classify_scene_categories(sample_image_path, top_idx, expected_scene_type):
    """
    Test that classify_scene correctly maps raw Places365 labels to
    DREAMS categories when confidence is above threshold.
    """
    mock_model = MagicMock()
    mock_output = _make_mock_output(top_idx, 0.9, MOCK_LABELS)
    mock_model.return_value = mock_output

    with patch.object(places365_classifier, "_load_model") as mock_load:
        mock_load.return_value = (mock_model, MOCK_LABELS)
        result = places365_classifier.classify_scene(sample_image_path)

    print(f"\n  Top label index: {top_idx}")
    print(f"  Result: {result}")

    assert result["scene_type"] == expected_scene_type
    assert result["scene_type"] in places365_classifier.VALID_SCENE_TYPES
    assert 0 <= result["scene_confidence"] <= 1
    assert len(result["scene_raw_top3"]) == 3


def test_classify_scene_low_confidence_returns_unknown(sample_image_path):
    """
    Test that classify_scene returns 'unknown' when the model's
    top confidence is below the 0.4 threshold.
    """
    mock_model = MagicMock()
    # All logits near zero → softmax spreads probability → max < 0.4
    mock_output = torch.zeros(1, 365)
    mock_model.return_value = mock_output

    with patch.object(places365_classifier, "_load_model") as mock_load:
        mock_load.return_value = (mock_model, MOCK_LABELS)
        result = places365_classifier.classify_scene(sample_image_path)

    print(f"\n  Low-confidence result: {result}")

    assert result["scene_type"] == "unknown"
    assert 0 <= result["scene_confidence"] <= 1
    assert len(result["scene_raw_top3"]) == 3


def test_classify_scene_unmapped_label_returns_unknown(sample_image_path):
    """
    Test that classify_scene returns 'unknown' when the top prediction
    is a Places365 label that is not in the DREAMS category mapping,
    even if confidence is high.
    """
    # Use a labels list where ALL labels are unmapped
    unmapped_labels = [f"unlabeled_scene_{i}" for i in range(365)]

    mock_model = MagicMock()
    mock_output = _make_mock_output(100, 0.9, unmapped_labels)
    mock_model.return_value = mock_output

    with patch.object(places365_classifier, "_load_model") as mock_load:
        mock_load.return_value = (mock_model, unmapped_labels)
        result = places365_classifier.classify_scene(sample_image_path)

    print(f"\n  Unmapped-label result: {result}")

    # scene_type should be "unknown" since none of top3 map to DREAMS categories
    assert result["scene_type"] == "unknown"
    assert 0 <= result["scene_confidence"] <= 1
    assert len(result["scene_raw_top3"]) == 3


def test_classify_scene_result_structure(sample_image_path):
    """
    Test that the returned dictionary always has the expected keys
    and value types regardless of classification outcome.
    """
    mock_model = MagicMock()
    mock_output = _make_mock_output(2, 0.8, MOCK_LABELS)  # park
    mock_model.return_value = mock_output

    with patch.object(places365_classifier, "_load_model") as mock_load:
        mock_load.return_value = (mock_model, MOCK_LABELS)
        result = places365_classifier.classify_scene(sample_image_path)

    print(f"\n  Structure check result: {result}")

    # Check keys exist
    assert "scene_type" in result
    assert "scene_confidence" in result
    assert "scene_raw_top3" in result

    # Check types
    assert isinstance(result["scene_type"], str)
    assert isinstance(result["scene_confidence"], float)
    assert isinstance(result["scene_raw_top3"], list)

    # Check scene_type validity
    assert result["scene_type"] in (places365_classifier.VALID_SCENE_TYPES | {"unknown"})

    # Check scene_raw_top3 entries
    for entry in result["scene_raw_top3"]:
        assert "label" in entry
        assert "confidence" in entry
        assert isinstance(entry["label"], str)
        assert isinstance(entry["confidence"], float)
        assert 0 <= entry["confidence"] <= 1
