"""Tests for data_integrity.schema_validator."""

import json

import pytest

pytest.importorskip("jsonschema")

from data_integrity.reporter import Severity
from data_integrity.schema_validator import validate_schema


def _write_schema(tmp_path, schema):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    return path


def test_valid_schema_reports_data_errors(tmp_path):
    schema_path = _write_schema(tmp_path, {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
    })

    issues = validate_schema({"age": "ten"}, schema_path)

    assert len(issues) == 1
    assert issues[0].location == "age"


@pytest.mark.parametrize("schema", [
    {"type": "objekt"},
    {"type": "object", "required": "name"},
    [1, 2],
])
def test_invalid_schema_is_reported_not_raised(tmp_path, schema):
    schema_path = _write_schema(tmp_path, schema)

    issues = validate_schema({"name": "x"}, schema_path)

    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].category == "schema"
    assert issues[0].message.startswith("Invalid schema")
    assert issues[0].location == str(schema_path)
