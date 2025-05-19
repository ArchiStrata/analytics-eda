import numpy as np
from analytics_eda.core.reporting.sanitize_json import sanitize_json


def test_nan_float_converted_to_none():
    assert sanitize_json(float('nan')) is None


def test_numpy_scalars_converted_to_native():
    assert sanitize_json(np.int32(10)) == 10
    assert isinstance(sanitize_json(np.int32(10)), int)

    assert sanitize_json(np.float64(3.14)) == 3.14
    assert isinstance(sanitize_json(np.float64(3.14)), float)

    assert sanitize_json(np.bool_(True)) is True
    assert isinstance(sanitize_json(np.bool_(True)), bool)


def test_nested_structure_sanitization():
    data = {
        "a": np.int64(5),
        "b": [np.float32(2.5), float('nan'), {"x": np.bool_(False)}]
    }
    expected = {
        "a": 5,
        "b": [2.5, None, {"x": False}]
    }
    assert sanitize_json(data) == expected


def test_builtin_types_unchanged():
    assert sanitize_json("string") == "string"
    assert sanitize_json(42) == 42
    assert sanitize_json(3.14) == 3.14
    assert sanitize_json(True) is True
