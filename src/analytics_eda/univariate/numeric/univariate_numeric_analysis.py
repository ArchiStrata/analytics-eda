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

from ...core import write_json_report, missing_data_analysis, validate_numeric_named_series, numeric_distribution_analysis, numeric_outlier_analysis, numeric_inferential_analysis

logger = logging.getLogger(__name__)

def univariate_numeric_analysis(
    series: pd.Series,
    report_root: str = 'reports/eda/univariate/numeric',
    iqr_multiplier: float = 1.5,
    z_thresh: float = 3.0,
    alpha: float = 0.05,
    popmean: float | None = None,
    popmedian: float | None = None,
    popvariance: float | None = None,
    bootstrap_samples: int = 1_000,
    report_log_id = str(uuid.uuid4())
) -> Path:
    """
    Conduct a full univariate analysis on a numeric series.

    Steps:
      1. Validate numeric series.
      2. Missing data analysis (counts, percentage, plot).
      3. Distribution analysis (descriptive stats, normality tests, visualizations).
      4. Outlier analysis (IQR, Z-score, robust Z-score with CSV exports).
      5. Inferential analysis (confidence intervals, goodness-of-fit, variance tests, effect size, bootstrap inference).
      6. Aggregation and saving of all results into a single JSON report.

    Args:
        series (pd.Series): Series to analyze.
        report_root (str): Base directory where report files will be saved.
        iqr_multiplier (float): IQR multiplier for outlier detection.
        z_thresh (float): Z-score threshold for outlier detection.
        alpha (float): Significance level for inferential tests and confidence intervals.
        popmean (float|None): Hypothesized population mean for inferential tests.
        popmedian (float|None): Hypothesized population median for inferential tests.
        popvariance (float|None): Hypothesized population variance (σ²) for inferential tests.
        bootstrap_samples (int): Number of bootstrap resamples for CI estimation.
        report_log_id (str): report log id.
    
    Returns:
        Path: File path to the saved JSON report as written by `write_json_report`.

    JSON report structure:
        {
            'metadata': { ... } # Report metadata
            'eda': {
                'missing_data': Summary of missing data analysis,
                'distribution': Summary of distribution analysis,
                'outliers': Summary of outlier analysis,
                'inferential': Summary of inferential analysis.
            }
        }
    """
    # 1. Validate Numeric
    validate_numeric_named_series(series)

    logger.info(
        "Starting univariate_numeric_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    # Always work from a copy
    s = series.copy()

    # Prepare directory
    save_dir = Path(report_root) / series.name.replace(' ', '_')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Missing Data Analysis
    missing_data = missing_data_analysis(s, save_dir, report_log_id=report_log_id)

    # 3. Distribution Analysis
    distribution_result = numeric_distribution_analysis(s, save_dir, alpha=alpha, report_log_id=report_log_id)
    series = distribution_result['series']

    # 4. Outlier Analysis
    outliers = numeric_outlier_analysis(series, save_dir, iqr_multiplier, z_thresh, report_log_id=report_log_id)

    # 5. Inferential Analysis
    inferential = numeric_inferential_analysis(
        series,
        alpha=alpha,
        popmean=popmean,
        popmedian=popmedian,
        popvariance=popvariance,
        bootstrap_samples=bootstrap_samples,
        report_log_id=report_log_id
    )

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
            'report_name': 'univariate_numeric_analysis',
            'parameters': {
                'series': series.name
            }
        },
        'eda': eda_report
    }

    report_path = save_dir / f"{series.name.replace(' ', '_')}_univariate_analysis_report.json"
    write_json_report(full_report, report_path)

    logger.info(
        "Completed univariate_numeric_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    return report_path
