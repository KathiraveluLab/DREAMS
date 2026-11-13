# tests/test_validator_cli.py

import json

from data_integrity import validator


def _run(monkeypatch, capsys, tmp_path, data, *extra_args):
    input_path = tmp_path / "data.json"
    input_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["validator", "--input", str(input_path), "--base-dir", str(tmp_path), *extra_args],
    )
    exit_code = validator.main()
    return exit_code, capsys.readouterr()


def test_json_output_is_machine_readable_when_clean(monkeypatch, capsys, tmp_path):
    exit_code, captured = _run(
        monkeypatch, capsys, tmp_path,
        {"samples": [{"timestamp": "2024-01-01T00:00:00"}]},
        "--json",
    )

    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["summary"]["has_errors"] is False


def test_json_output_is_machine_readable_with_errors(monkeypatch, capsys, tmp_path):
    exit_code, captured = _run(
        monkeypatch, capsys, tmp_path,
        {"samples": [{"image": "missing.jpg", "timestamp": "2024-01-01T00:00:00"}]},
        "--json",
    )

    report = json.loads(captured.out)
    assert exit_code == 1
    assert report["summary"]["has_errors"] is True
    assert any(issue["category"] == "path" for issue in report["issues"])


def test_progress_messages_go_to_stderr(monkeypatch, capsys, tmp_path):
    _, captured = _run(
        monkeypatch, capsys, tmp_path,
        {"samples": [{"timestamp": "2024-01-01T00:00:00"}]},
        "--json",
    )

    assert "Validating:" in captured.err
    assert "Validating:" not in captured.out


def test_text_report_still_printed_to_stdout(monkeypatch, capsys, tmp_path):
    _, captured = _run(
        monkeypatch, capsys, tmp_path,
        {"samples": [{"image": "missing.jpg"}]},
    )

    assert "Media file not found" in captured.out
