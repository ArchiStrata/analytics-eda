# Copyright 2025 ArchiStrata, LLC and Andrew Dabrowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
import logging
import uuid
import pandas as pd

from ...core.reporting import write_json_report

from .visualize_time_series_structure import visualize_time_series_structure

logger = logging.getLogger(__name__)

def univariate_timeseries_analysis(df: pd.DataFrame,
                                   numeric_col: str,
                                   time_col: str,
                                   report_root: str = 'reports/eda/univariate/timeseries',
                                   rolling_window: int = 12,
                                   report_log_id = str(uuid.uuid4())) -> Path:
    """
    Run full univariate time-series analysis for a single numeric series.

    Validates input types, drops missing data, sets the datetime index,
    generates structure visualizations, and writes a JSON report with paths
    to all generated visuals.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the time and value columns.
    - numeric_col (str): Name of the numeric series column (must be numeric dtype).
    - time_col (str): Name of the datetime column (must be datetime64[ns] dtype).
    - report_root (str): Base directory under which a subfolder for this analysis
      will be created.
    - rolling_window (int): Window size for rolling statistics (default: 12).
    - report_log_id (str): report log id.

    Returns:
    - pathlib.Path: Path to the JSON report summarizing the analysis.
    """
    logger.info(
        "Starting univariate_timeseries_analysis",
        extra={
            'time_col': time_col,
            'numeric_col': numeric_col,
            'report_log_id': report_log_id
        }
    )

    # Work on a copy to preserve original data
    df_copy = df.copy()

    # 1. Validate that time_col is datetime64
    if not pd.api.types.is_datetime64_any_dtype(df_copy[time_col]):
        raise TypeError(
            f"Column '{time_col}' is of type {df_copy[time_col].dtype}; "
            "expected datetime64 dtype. "
            "Please convert it to datetime in your cleaning pipeline before analysis."
        )

    # 2. Validate that numeric_col is numeric
    if not pd.api.types.is_numeric_dtype(df_copy[numeric_col]):
        raise TypeError(
            f"Column '{numeric_col}' is of type {df_copy[numeric_col].dtype}; "
            "expected a numeric dtype. "
            "Please convert it to numeric in your cleaning pipeline before analysis."
        )

    # 3. Drop rows with missing values in essential columns
    df_copy = df_copy.dropna(subset=[time_col, numeric_col])

    # 4. Set datetime index and sort chronologically
    df_copy = df_copy.set_index(time_col).sort_index()

    # 5. Prepare report directory
    report_dir = Path(report_root) / f"{time_col.replace(' ', '_')}_{numeric_col.replace(' ', '_')}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 6. Visualize Time Series Structure
    visuals = visualize_time_series_structure(df, numeric_col, time_col, report_dir, rolling_window, report_log_id=report_log_id)

    # Generate report
    eda_report = {
        'visuals': visuals
    }

    full_report = {
        'metadata': {
            'version': '0.1.0',
            'report_name': 'univariate_categorical_analysis',
            'parameters': {
                'numeric_col': numeric_col,
                'time_col': time_col
            }
        },
        'eda': eda_report
    }

    report_path = report_dir / f"{time_col.replace(' ', '_')}_{numeric_col.replace(' ', '_')}_univariate_analysis_report.json"
    write_json_report(full_report, report_path)

    logger.info(
        "Completed univariate_timeseries_analysis",
        extra={
            'time_col': time_col,
            'numeric_col': numeric_col,
            'report_log_id': report_log_id
        }
    )

    return report_path
