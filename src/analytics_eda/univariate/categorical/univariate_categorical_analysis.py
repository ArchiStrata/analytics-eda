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

from ...core import write_json_report, missing_data_analysis, validate_categorical_named_series, categorical_inferential_analysis, categorical_distribution_analysis

logger = logging.getLogger(__name__)

def univariate_categorical_analysis(
    series: pd.Series,
    top_n: int = 10,
    report_root: str = 'reports/eda/univariate/categorical',
    rare_threshold: float = 0.01,
    alpha: float = 0.05,
    report_log_id = str(uuid.uuid4())
) -> Path:
    """
    Run a full univariate analysis on a named categorical pandas Series and save results.

    This function will:
      1. Validate that `series` is a named categorical Series.
      2. Compute and save missing-data statistics using `missing_data_analysis`.
      3. Generate frequency distribution and a top-N bar plot via `categorical_distribution_analysis`.
      4. Identify rare categories below `rare_threshold`.
      5. Perform a chi-square goodness-of-fit test (uniform) via `categorical_inferential_analysis`.
      6. Compile all outputs and write a JSON report with `write_json_report`.

    Args:
        series (pd.Series): Named categorical Series (dtype 'category' or 'object').
        top_n (int, optional): Number of leading categories in the bar plot.
            Adjusted if fewer unique values exist. Defaults to 10.
        report_root (str, optional): Directory path for saving plots and report.
            Defaults to 'reports/eda/univariate/categorical'.
        rare_threshold (float, optional): Proportion threshold for rare-category detection.
            Defaults to 0.01.
        alpha (float, optional): Significance level for inferential testing. Defaults to 0.05.
        report_log_id (str): report log id.

    Returns:
        Path: File path to the saved JSON report as written by `write_json_report`.

    JSON report structure:
        {
            'metadata': { ... } # Report metadata
            'eda': {
                'missing_data': {'total': int, 'missing': int, 'pct_missing': float},
                'distribution': {...},  # output from categorical_distribution_analysis
                'outliers': {'rare_categories': List[str]},
                'inferential': {
                    'goodness_of_fit': {
                        'chi2_statistic': float,
                        'p_value': float,
                        'alpha': float,
                        'reject_null_uniform': bool
                    }
                }
            }
        }
    """
    # 1. Validation
    validate_categorical_named_series(series)

    logger.info(
        "Starting univariate_categorical_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    # Prepare save directory
    save_dir = Path(report_root) / series.name.replace(' ', '_')
    save_dir.mkdir(parents=True, exist_ok=True)

    total = int(len(series))

    # 2. Missing Data Analysis
    missing_data = missing_data_analysis(series, save_dir, report_log_id=report_log_id)

    # 3. Distribution Analysis
    distribution_result = categorical_distribution_analysis(series, save_dir, top_n, report_log_id=report_log_id)
    freq_tbl = distribution_result['report']['frequency_report']['frequency_table']

    # 4. Outlier Analysis
    # Identify rare categories
    rare_categories = [
        cat
        for cat, stats in freq_tbl.items()
        if stats['proportion'] < rare_threshold
    ]

    outliers = {
        'rare_categories': rare_categories,
    }

    # 5. Inferential Analysis
    inferential = categorical_inferential_analysis(freq_tbl, total, alpha)

    # 6. Generate report
    eda_report = {
        'missing_data': missing_data,
        'distribution': distribution_result['report'],
        'outliers': outliers,
        'inferential': inferential
    }

    full_report = {
        'metadata': {
            'version': '0.1.0',
            'report_name': 'univariate_categorical_analysis',
            'parameters': {
                'series': series.name
            }
        },
        'eda': eda_report
    }

    report_path = save_dir / f"{series.name.replace(' ', '_')}_univariate_analysis_report.json"
    write_json_report(full_report, report_path)

    logger.info(
        "Completed univariate_categorical_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    return report_path
