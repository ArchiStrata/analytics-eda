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

        def _handle_callable(actual_value, func, path_label: str):
            res = func(actual_value)
            import numpy as _np
            if isinstance(res, (bool, _np.bool_)) or res is None:
                assert bool(res), f"{path_label} predicate failed; got {actual_value!r}"
            elif isinstance(res, dict):
                # Allow returning a nested expected dict
                _assert_mapping(actual_value, res, path_label)
            elif isinstance(res, (list, tuple)):
                # Allow returning a sequence to compare to
                assert actual_value == res, f"{path_label} expected {res!r}, got {actual_value!r}"
            else:
                # e.g. pytest.approx(...) or any other comparator-like object
                assert actual_value == res, f"{path_label} expected {res!r}, got {actual_value!r}"

        def _assert_sequence(actual_seq, expected_seq, path_label: str):
            assert isinstance(actual_seq, (list, tuple)), f"{path_label} should be a sequence"
            assert len(actual_seq) == len(expected_seq), f"{path_label} length mismatch"
            for i, (ai, ei) in enumerate(zip(actual_seq, expected_seq)):
                item_label = f"{path_label}[{i}]"
                if isinstance(ei, dict):
                    assert isinstance(ai, dict), f"{item_label} should be a dict"
                    _assert_mapping(ai, ei, item_label)
                elif callable(ei):
                    _handle_callable(ai, ei, item_label)
                elif isinstance(ei, (list, tuple)):
                    _assert_sequence(ai, ei, item_label)
                else:
                    assert ai == ei, f"{item_label} expected {ei!r}, got {ai!r}"

        def _assert_mapping(actual: dict, expected: dict, label: str):
            for k, v in (expected or {}).items():
                assert k in actual, f"{label} missing key: {k!r}"
                av = actual[k]
                key_label = f"{label}[{k!r}]"
                if isinstance(v, dict):
                    assert isinstance(av, dict), f"{key_label} should be a dict"
                    _assert_mapping(av, v, key_label)
                elif callable(v):
                    _handle_callable(av, v, key_label)
                elif isinstance(v, (list, tuple)):
                    _assert_sequence(av, v, key_label)
                else:
                    assert av == v, f"{key_label} expected {v!r}, got {av!r}"

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
