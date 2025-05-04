import json
import pytest
from analytics_eda.core.reporting.write_json_report import write_json_report


def test_write_json_report_sanitizes_and_writes_file(tmp_path):
    # Prepare test data
    original = {'foo': 'bar'}
    clean = {'foo': 'bar'}

    report_dir = tmp_path / 'nested' / 'dir'
    report_path = report_dir / 'out.json'
    # Ensure directory does not exist initially
    assert not report_dir.exists()

    # Call function
    result = write_json_report(original, str(report_path))

    # Directory should be created
    assert report_dir.exists() and report_dir.is_dir()
    # Function should return the sanitized report
    assert result == clean

    # File should exist and contain clean JSON
    assert report_path.exists()
    text = report_path.read_text(encoding='utf-8')
    data = json.loads(text)
    assert data == clean
    # Check formatting: indent of 4 spaces on second line
    lines = text.splitlines()
    assert lines[1].startswith('    ')


def test_existing_directory_is_not_error(tmp_path):
    # Create directory beforehand
    report_dir = tmp_path / 'preexists'
    report_dir.mkdir(parents=True)
    report_path = report_dir / 'r.json'

    result = write_json_report({'a': 1}, str(report_path))
    assert result == {'a': 1}
    assert report_path.exists()
