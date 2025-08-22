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


@pytest.fixture
def assert_report_data(load_and_validate_report, assert_plot_metadata):
    """
    Validate a (sub)section of a JSON report with minimal boilerplate.

    Usage patterns:

        # 1) Start from a response that has "report_file_path", validate whole "data"
        assert_report_data(out, expected_distribution, tmp_path)

        # 2) Start from an already-loaded report dict (or any nested dict),
        #    and validate a nested section
        report = load_and_validate_report(out, tmp_path)
        assert_report_data(report["data"]["shape"], expected_shape, tmp_path, root_key=None)

    Notes:
      • For each (key -> expected) in the expected mapping:
          - Asserts `key` is present in the actual mapping.
          - If the actual node has "report_file_path": calls load_and_validate_report(node, tmp_path) and stops there.
          - Else if the actual node looks like a plot (has "chart_metadata" & "descriptive_stats"):
              calls assert_plot_metadata(actual_node, expected, tmp_path).
          - Else if both expected & actual are dicts: recurses into that subsection.
          - Else if expected is callable: uses it as a predicate on the actual node.
          - Else if expected is a sequence: does elementwise checks (supports nested dicts/callables).
          - Else: equality check.
      • Empty dict `{}` in expectations simply asserts presence (and, if the actual node is a sub‑report,
        it will still verify that the report file exists).
    """
    def _assert_sequence(actual_seq, expected_seq, tmp_path: Path, path_label: str):
        assert isinstance(actual_seq, (list, tuple)), f"{path_label} should be a sequence"
        assert len(actual_seq) == len(expected_seq), f"{path_label} length mismatch"
        for i, (ai, ei) in enumerate(zip(actual_seq, expected_seq)):
            item_label = f"{path_label}[{i}]"
            if isinstance(ei, dict):
                assert isinstance(ai, dict), f"{item_label} should be a dict"
                _assert_mapping(ai, ei, tmp_path, item_label)
            elif callable(ei):
                _handle_callable(ai, ei, tmp_path, item_label)
            elif isinstance(ei, (list, tuple)):
                _assert_sequence(ai, ei, tmp_path, item_label)
            else:
                assert ai == ei, f"{item_label} expected {ei!r}, got {ai!r}"

    def _handle_callable(actual_value, func, tmp_path: Path, path_label: str):
        res = func(actual_value)
        import numpy as _np
        if isinstance(res, (bool, _np.bool_)) or res is None:
            assert bool(res), f"{path_label} predicate failed; got {actual_value!r}"
        elif isinstance(res, dict):
            _assert_mapping(actual_value, res, tmp_path, path_label)
        elif isinstance(res, (list, tuple)):
            _assert_sequence(actual_value, res, tmp_path, path_label)
        else:
            assert actual_value == res, f"{path_label} expected {res!r}, got {actual_value!r}"

    def _looks_like_plot(node: dict) -> bool:
        return isinstance(node, dict) and ("chart_metadata" in node and "descriptive_stats" in node)

    def _assert_mapping(actual: dict, expected: dict, tmp_path: Path, label: str):
        assert isinstance(actual, dict), f"{label} should be a dict"
        for k, vexp in (expected or {}).items():
            assert k in actual, f"{label} missing key: {k!r}"
            aval = actual[k]
            key_label = f"{label}.{k}"

            # 1) If this node links to a nested report, only verify it exists.
            if isinstance(aval, dict) and "report_file_path" in aval:
                load_and_validate_report(aval, tmp_path)  # do not recurse or assert plot metadata
                continue

            # 2) Plot payloads → use existing plot validator
            if isinstance(aval, dict) and _looks_like_plot(aval):
                assert isinstance(vexp, dict), f"{key_label} expectations must be a dict for plot nodes"
                assert_plot_metadata(aval, vexp, tmp_path)
                continue

            # 3) Sections → recurse
            if isinstance(vexp, dict) and isinstance(aval, dict):
                _assert_mapping(aval, vexp, tmp_path, key_label)
                continue

            # 4) Callable predicate on the whole node
            if callable(vexp):
                _handle_callable(aval, vexp, tmp_path, key_label)
                continue

            # 5) Sequences
            if isinstance(vexp, (list, tuple)):
                _assert_sequence(aval, vexp, tmp_path, key_label)
                continue

            # 6) Fallback: equality
            assert aval == vexp, f"{key_label} expected {vexp!r}, got {aval!r}"

        # If expected is empty {}, we still want to verify a direct report link if present.
        # (This covers cases like transforms: {"yeo-johnson": {}, ...} where actual holds a report link.)
        if (expected == {} or expected is None) and isinstance(actual, dict) and "report_file_path" in actual:
            load_and_validate_report(actual, tmp_path)

    def _entry_point(response_or_mapping, expected: dict, tmp_path: Path, root_key: str | None):
        # If we were handed a response with a file pointer, load it.
        if isinstance(response_or_mapping, dict) and "report_file_path" in response_or_mapping:
            loaded = load_and_validate_report(response_or_mapping, tmp_path)
            root = loaded.get(root_key, loaded) if root_key else loaded
        else:
            root = response_or_mapping.get(root_key, response_or_mapping) if isinstance(response_or_mapping, dict) else response_or_mapping

        assert isinstance(root, dict), f"Root to validate must be a dict; got {type(root).__name__}"
        _assert_mapping(root, expected, tmp_path, root_key or "<root>")

    def _fixture(response_or_mapping, expected: dict, tmp_path: Path, root_key: str = "data"):
        _entry_point(response_or_mapping, expected, tmp_path, root_key)

    return _fixture
