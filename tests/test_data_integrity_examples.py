import json
from pathlib import Path

from data_integrity.reporter import Severity
from data_integrity.temporal_validator import validate_temporal

EXAMPLES = Path(__file__).resolve().parent.parent / "data_integrity" / "examples"


def test_sample_data_reports_future_timestamp():
    with open(EXAMPLES / "sample_data.json", encoding="utf-8") as f:
        data = json.load(f)

    issues = validate_temporal(data)
    future = [i for i in issues if "Future timestamp" in i.message]

    assert [i.location for i in future] == ["samples[4].timestamp"]
    assert future[0].severity == Severity.ERROR


def test_sample_data_reports_out_of_order_timestamp():
    with open(EXAMPLES / "sample_data.json", encoding="utf-8") as f:
        data = json.load(f)

    issues = validate_temporal(data)
    out_of_order = [i for i in issues if "Out-of-order" in i.message]

    assert len(out_of_order) == 1
    assert out_of_order[0].location.startswith("samples[3].timestamp")
