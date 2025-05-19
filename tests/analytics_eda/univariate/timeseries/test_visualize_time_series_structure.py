import pytest
from pathlib import Path
import pandas as pd
from analytics_eda.univariate.timeseries.visualize_time_series_structure import visualize_time_series_structure

def test_visualize_time_series_structure(tmp_path):
    # 1. Create sample time-series DataFrame
    dates = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    values = range(1, 25)
    df = pd.DataFrame({'date': dates, 'value': values})

    # 2. Define report directory under tmp_path
    report_dir = tmp_path / "reports" / "ts_structure"
    
    # 3. Call the function
    visuals = visualize_time_series_structure(
        df,
        value_col='value',
        time_col='date',
        report_dir=report_dir,
        rolling_window=6
    )

    # 4. Assertions
    assert isinstance(visuals, dict), "Function should return a dict"

    expected_keys = [
        'raw_line_plot',
        'rolling_statistics',
        'acf',
        'pacf',
        'stl_decomposition'
    ]
    expected_filenames = {
        'raw_line_plot': 'raw_line_plot.png',
        'rolling_statistics': 'rolling_statistics.png',
        'acf': 'acf.png',
        'pacf': 'pacf.png',
        'stl_decomposition': 'stl_decomposition.png'
    }

    for key in expected_keys:
        # key must exist
        assert key in visuals, f"Missing visual key: {key}"

        path_str = visuals[key]
        # path should not be None
        assert path_str is not None, f"File path for {key} is None"

        path = Path(path_str)
        # file must exist and be a file
        assert path.exists(), f"File for {key} does not exist at {path}"
        assert path.is_file(), f"Path for {key} is not a file: {path}"
        # file should not be empty
        assert path.stat().st_size > 0, f"File for {key} is empty: {path}"

        # path must match expected filename under report_dir
        expected_path = report_dir / expected_filenames[key]
        assert path == expected_path, (
            f"Expected path for {key} to be {expected_path}, got {path}"
        )

def test_non_datetime_time_col_raises_type_error(tmp_path):
    # Create a DataFrame where 'date' column is present but not datetime64 dtype
    df = pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01'],
        'value': [10, 20, 30]
    })
    report_dir = tmp_path / "reports"

    # Expect a TypeError when time_col exists but isn't datetime64
    with pytest.raises(TypeError) as excinfo:
        visualize_time_series_structure(
            df,
            value_col='value',
            time_col='date',
            report_dir=report_dir,
            rolling_window=3
        )

    # Assert the exception message mentions the column name and dtype requirement
    msg = str(excinfo.value)
    assert "'date' must be datetime64 dtype" in msg
