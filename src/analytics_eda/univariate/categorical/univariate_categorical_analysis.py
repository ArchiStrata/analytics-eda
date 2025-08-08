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
from typing import Any, Dict, Optional
import uuid
import pandas as pd

from ...core import write_json_report, missing_data_analysis, validate_categorical_named_series, categorical_distribution_analysis

logger = logging.getLogger(__name__)

def univariate_categorical_analysis(
    series: pd.Series,
    report_root: str = 'reports/eda/univariate/categorical',
    report_log_id = str(uuid.uuid4()),
    data_source: Optional[str] = None,
    plot_frequency_pareto_overrides: Optional[Dict[str, Any]] = None,
    plot_distribution_density_overrides: Optional[Dict[str, Any]] = None,
    plot_dispersion_boxplot_overrides: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Run a full univariate analysis on a named categorical pandas Series and save results.

    This function will:
      1. Validate that `series` is a named categorical Series.
      2. Compute and save missing-data statistics using `missing_data_analysis`.
      3. Generate frequency distribution and a top-N bar plot via `categorical_distribution_analysis`.
      4. Compile all outputs and write a JSON report with `write_json_report`.

    Args:
        series (pd.Series): Named categorical Series (dtype 'category' or 'object').
        report_root (str, optional): Directory path for saving plots and report.
            Defaults to 'reports/eda/univariate/categorical'.
        report_log_id (str): report log id.

    Returns:
        {
            'report_file_path': <report_file_path> # File path to the saved JSON report as written by `write_json_report`.
        }

    JSON report structure:
        {
            'metadata': { ... } # Report metadata
            'data': {
                'missing_data': {'total': int, 'missing': int, 'pct_missing': float},
                'distribution': {...}  # output from categorical_distribution_analysis
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

    # 1. Data Quality & Standardization
    # Missing Data Analysis
    missing_data = missing_data_analysis(series, save_dir, report_log_id=report_log_id)

    # TODO: Label consistency (spelling/casing/abbreviations)

    # 2. Distribution Analysis
    distribution_result = categorical_distribution_analysis(
        series,
        save_dir,
        report_log_id=report_log_id,
        data_source=data_source,
        plot_frequency_pareto_overrides=plot_frequency_pareto_overrides,
        plot_distribution_density_overrides=plot_distribution_density_overrides,
        plot_dispersion_boxplot_overrides=plot_dispersion_boxplot_overrides,
    )

    # Generate report
    eda_report = {
        'missing_data': missing_data,
        'distribution': distribution_result
    }

    full_report = {
        'metadata': {
            'version': '0.1.0',
            'report_name': 'univariate_categorical_analysis',
            'parameters': {
                'series': series.name
            }
        },
        'data': eda_report
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

    return {
        'report_file_path': report_path
    }
