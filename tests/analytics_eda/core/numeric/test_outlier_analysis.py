import pytest
import pandas as pd
from pathlib import Path
from analytics_eda.core.numeric.outlier_analysis import outlier_analysis


def test_empty_series_returns_empty_and_no_files(tmp_path):
    s = pd.Series([], dtype=float, name="nums")
    result = outlier_analysis(s, tmp_path)
    assert result == {}
    # No files should be created
    assert not any(tmp_path.iterdir())


def test_non_series_input_raises_type_error(tmp_path):
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        outlier_analysis([1, 2, 3], tmp_path)


def test_missing_name_raises_value_error(tmp_path):
    s = pd.Series([1, 2, 3])  # name is None
    with pytest.raises(ValueError, match="Series must have a non-empty 'name'"):
        outlier_analysis(s, tmp_path)


def test_outlier_analysis_creates_expected_files_and_summary(tmp_path):
    # Use a simple series to force IQR outliers at extremes
    s = pd.Series([1, 2, 3, 4], name="nums")
    # Set iqr_multiplier=0 to flag 1 and 4 as outliers; high z_thresh to disable zscore flags
    summary = outlier_analysis(s, tmp_path, iqr_multiplier=0, z_thresh=100)

    # IQR summary
    iqr = summary['iqr']
    assert iqr['count'] == 2
    assert iqr['pct'] == pytest.approx(0.5)
    file_iqr = Path(iqr['outliers_file'])
    assert file_iqr.exists()
    df_iqr = pd.read_csv(file_iqr)
    assert list(df_iqr['nums']) == [1, 4]

    # Z-score summary should have no outliers
    z = summary['zscore']
    assert z['count'] == 0
    file_z = tmp_path / 'nums_zscore_outliers.csv'
    assert file_z.exists()
    df_z = pd.read_csv(file_z)
    assert df_z.empty

    # Robust Z-score summary should have no outliers
    rz = summary['robust_zscore']
    assert rz['count'] == 0
    file_rz = tmp_path / 'nums_robust_zscore_outliers.csv'
    assert file_rz.exists()
    df_rz = pd.read_csv(file_rz)
    assert df_rz.empty
