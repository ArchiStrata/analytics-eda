from pathlib import Path

import pandas as pd

from analytics_eda.core.numeric.evaluate_transforms import evaluate_transforms
from analytics_eda.core.numeric.select_transforms     import select_transforms


def test_evaluate_transforms_invokes_analysis_per_candidate(monkeypatch, tmp_path):
    # 1) Prepare a simple series and stats/normality inputs
    series = pd.Series([1.0, 2.0, 3.0], name="x")
    statistics = {'min': 1.0, 'skewness': 0.0, 'kurtosis': 0.0}
    normality_tests = {'reject_normality': False}

    # 2) Stub‐out numeric_distribution_analysis to capture calls and return dummy meta
    calls = []
    def fake_numeric_distribution_analysis(transformed_series, report_path, **kwargs):
        # record the exact arguments
        calls.append((transformed_series.copy(), Path(report_path)))
        return {'dummy_meta': str(report_path)}
    monkeypatch.setattr(
        'analytics_eda.core.numeric.evaluate_transforms.numeric_distribution_analysis',
        fake_numeric_distribution_analysis
    )

    is_discrete = True

    # 3) Run evaluate_transforms
    out = evaluate_transforms(
        series,
        is_discrete,
        statistics,
        normality_tests,
        report_path=tmp_path
    )

    # 4) Expectations on returned structure
    assert 'transforms' in out, "Must return a 'transforms' mapping"
    transforms = out['transforms']

    # 5) Determine expected candidates from select_transforms
    expected = select_transforms(statistics, normality_tests)
    assert set(transforms.keys()) == set(expected), "Should include exactly the selected transforms"

    # 6) Ensure analysis stub was called once per transform
    assert len(calls) == len(expected)

    # 7) For each candidate, check that:
    for transform_name, (transformed_series, call_path) in zip(expected, calls):
        # a) numeric_distribution_analysis was called with the correct subdirectory
        expected_dir = tmp_path / transform_name
        assert call_path == expected_dir

        # b) That subdirectory was actually created
        assert expected_dir.exists() and expected_dir.is_dir()

        # c) The returned metadata is preserved
        assert transforms[transform_name] == {'dummy_meta': str(expected_dir)}
