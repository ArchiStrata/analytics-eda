from pathlib import Path
import pytest
from tests.analytics_eda.utils_internal.load_and_validate_report import load_and_validate_report as _load_and_validate_report

@pytest.fixture
def load_and_validate_report():
    return _load_and_validate_report

@pytest.fixture
def assert_plot_metadata():
    """
    Assert a plot payload that looks like:
      {
        "chart_metadata": {...},
        "descriptive_stats": {...}
      }

    Usage:
        expect = {
            "chart_metadata": { "xlabel": "Frequency", "data_source": "UnitTest" },
            "descriptive_stats": { "n": lambda v: v > 0 }
        }
        assert_plot_metadata(payload, expect, tmp_path)

    Notes:
      • Expected values may be literals (==) or callables (predicate → True).
      • If 'file_name' is included in expected chart_metadata:
          - when None → assert no file was saved
          - otherwise → assert the PNG exists and has a valid signature
      • If 'file_name' is NOT specified in expectations, we still verify the file
        if the payload provides a non-empty file_name.
    """
    def _assert(payload: dict, expect: dict, tmp_path: Path):
        assert isinstance(payload, dict), "payload must be a dict"
        assert "chart_metadata" in payload and "descriptive_stats" in payload, \
            "payload must contain 'chart_metadata' and 'descriptive_stats'"

        cm = payload["chart_metadata"]
        ds = payload["descriptive_stats"]

        def _assert_mapping(actual: dict, expected: dict, label: str):
            for k, v in (expected or {}).items():
                assert k in actual, f"{label} missing key: {k!r}"
                if callable(v):
                    assert v(actual[k]), f"{label}[{k!r}] predicate failed; got {actual[k]!r}"
                else:
                    assert actual[k] == v, f"{label}[{k!r}] expected {v!r}, got {actual[k]!r}"

        # Field-by-field checks
        _assert_mapping(cm, expect.get("chart_metadata", {}), "chart_metadata")
        _assert_mapping(ds, expect.get("descriptive_stats", {}), "descriptive_stats")

        if "inferential_stats" in expect:
            assert "inferential_stats" in payload, "payload must contain 'inferential_stats'"
            inferential_stats = payload["inferential_stats"]
            _assert_mapping(inferential_stats, expect.get("inferential_stats", {}), "inferential_stats")

        # File check logic
        exp_file_in_expect = "file_name" in (expect.get("chart_metadata") or {})
        file_name = cm.get("file_name")

        if exp_file_in_expect:
            exp_file = expect["chart_metadata"]["file_name"]
            if exp_file is None:
                assert file_name is None, "Expected no file to be saved"
                return
            # explicit file name required
            assert file_name == exp_file, "file_name mismatch"
            saved = tmp_path / exp_file
        else:
            # not specified → if present, verify it; if absent, do nothing
            if not file_name:
                return
            saved = tmp_path / file_name

        # Verify PNG exists and signature
        assert saved.exists() and saved.is_file(), f"Missing saved file: {saved}"
        assert saved.stat().st_size > 0, "Saved file is empty"
        with open(saved, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n", "Saved file is not a PNG"

    return _assert
