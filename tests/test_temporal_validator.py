"""Tests for data_integrity.temporal_validator."""
import json
from pathlib import Path

import pytest

from data_integrity.reporter import Severity
from data_integrity.temporal_validator import validate_temporal

EXAMPLES = Path(__file__).resolve().parent.parent / "data_integrity" / "examples"


def _errors(data, **kwargs):
    return [i for i in validate_temporal(data, **kwargs) if i.severity == Severity.ERROR]


# --- ordering is checked within one field across the items of one array -------------------


def test_samples_out_of_order_are_reported():
    data = {"samples": [
        {"timestamp": "2024-01-01T10:00:00Z"},
        {"timestamp": "2024-01-03T10:00:00Z"},
        {"timestamp": "2024-01-02T10:00:00Z"},
    ]}

    errors = _errors(data)

    assert len(errors) == 1
    assert "samples[2].timestamp" in errors[0].location
    assert "samples[1].timestamp" in errors[0].location


def test_ordered_samples_pass():
    data = {"samples": [{"timestamp": f"2024-01-0{i}T10:00:00Z"} for i in range(1, 5)]}

    assert _errors(data) == []


def test_strict_mode_reports_duplicate_timestamps():
    data = {"samples": [
        {"timestamp": "2024-01-01T10:00:00Z"},
        {"timestamp": "2024-01-01T10:00:00Z"},
    ]}

    assert _errors(data) == []
    assert len(_errors(data, strict_monotonic=True)) == 1


@pytest.mark.parametrize("first", ["created_at", "capture_time"])
def test_different_fields_of_one_record_are_not_compared(first):
    """A photo is taken before it is uploaded, whichever key comes first in the JSON."""
    taken, uploaded = "2024-01-01T10:00:00Z", "2024-01-05T10:00:00Z"
    record = (
        {"created_at": uploaded, "capture_time": taken}
        if first == "created_at"
        else {"capture_time": taken, "created_at": uploaded}
    )

    assert _errors(record) == []


def test_different_fields_of_each_array_item_are_not_compared():
    data = {"posts": [
        {"created_at": "2024-01-05T10:00:00Z", "capture_time": "2024-01-01T10:00:00Z"},
        {"created_at": "2024-01-06T10:00:00Z", "capture_time": "2024-01-02T10:00:00Z"},
    ]}

    assert _errors(data) == []


def test_each_field_is_still_checked_across_items():
    data = {"posts": [
        {"created_at": "2024-01-06T10:00:00Z", "capture_time": "2024-01-01T10:00:00Z"},
        {"created_at": "2024-01-05T10:00:00Z", "capture_time": "2024-01-02T10:00:00Z"},
    ]}

    errors = _errors(data)

    assert len(errors) == 1
    assert "posts[1].created_at" in errors[0].location


def test_nested_readings_are_not_compared_with_their_session_timestamp():
    data = {"sessions": [{
        "timestamp": "2024-01-02T00:00:00Z",
        "readings": [
            {"timestamp": "2024-01-01T10:00:00Z"},
            {"timestamp": "2024-01-01T11:00:00Z"},
        ],
    }]}

    assert _errors(data) == []


def test_readings_out_of_order_inside_one_session_are_reported():
    data = {"sessions": [{"readings": [
        {"timestamp": "2024-01-01T11:00:00Z"},
        {"timestamp": "2024-01-01T10:00:00Z"},
    ]}]}

    errors = _errors(data)

    assert len(errors) == 1
    assert "sessions[0].readings[1].timestamp" in errors[0].location


def test_readings_of_different_sessions_are_ordered_separately():
    data = {"sessions": [
        {"readings": [{"timestamp": "2024-01-02T10:00:00Z"}]},
        {"readings": [{"timestamp": "2024-01-01T10:00:00Z"}]},
    ]}

    assert _errors(data) == []


# --- values that are not usable timestamps ------------------------------------------------


@pytest.mark.parametrize("value", [1e300, float("inf"), float("-inf"), float("nan"), 10**30])
def test_out_of_range_numbers_are_ignored_not_fatal(value):
    issues = validate_temporal({"samples": [{"timestamp": value}]})

    assert [i.message for i in issues] == ["No timestamps found in data"]


@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_not_timestamps(value):
    issues = validate_temporal({"timestamp": value})

    assert [i.message for i in issues] == ["No timestamps found in data"]


def test_a_bad_value_does_not_hide_the_other_timestamps():
    data = {"samples": [
        {"timestamp": 1e300},
        {"timestamp": "2024-01-02T10:00:00Z"},
        {"timestamp": "2024-01-01T10:00:00Z"},
    ]}

    assert len(_errors(data)) == 1


# --- the shipped examples keep their results ----------------------------------------------


def _example(name):
    return json.loads((EXAMPLES / name).read_text())


def test_sample_data_still_reports_its_out_of_order_sample():
    errors = _errors(_example("sample_data.json"))

    assert any(i.message.startswith("Out-of-order") for i in errors)


@pytest.mark.parametrize("name", ["valid_data.json", "millisecond_timestamps.json"])
def test_clean_examples_stay_clean(name):
    assert validate_temporal(_example(name)) == []
