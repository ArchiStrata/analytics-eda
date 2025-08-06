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
from typing import Optional
import uuid

import pandas as pd

from ...core import write_json_report, missing_data_analysis, validate_numeric_named_series, numeric_distribution_analysis, plot_cardinality_barchart

logger = logging.getLogger(__name__)

def univariate_numeric_analysis(
    series: pd.Series,
    report_root: str = 'reports/eda/univariate/numeric',
    report_log_id = str(uuid.uuid4()),
    data_source: Optional[str] = None
) -> Path:
    """
    Conduct a full univariate analysis on a numeric series.

    Steps:
      1. Validate numeric series.
      2. Missing data analysis (counts, percentage, plot).
      3. Cardinality analysis.
      4. Distribution analysis (descriptive stats, normality tests, visualizations).
      5. Aggregation and saving of all results into a single JSON report.

    Args:
        series (pd.Series): Series to analyze.
        report_root (str): Base directory where report files will be saved.
        report_log_id (str): report log id.
    
    Returns:
        Path: File path to the saved JSON report as written by `write_json_report`.

    JSON report structure:
        {
            'metadata': { ... } # Report metadata
            'data': {
                'missing_data': Summary of missing data analysis,
                'distribution': Summary of distribution analysis,
            }
        }
    """
    validate_numeric_named_series(series)

    logger.info(
        "Starting univariate_numeric_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    # Always work from a copy
    series_copy = series.copy()

    # Prepare directory
    report_path = Path(report_root) / series_copy.name.replace(' ', '_')
    report_path.mkdir(parents=True, exist_ok=True)

    # Missing Data Analysis
    missing_data = missing_data_analysis(series_copy, report_path, report_log_id=report_log_id)

    # Cardinality Analysis
    top_k = 10
    plot_cardinality_barchart_file_name = f"Value Counts (Top {top_k}) of {series_copy.name} for Cardinality.png"
    plot_cardinality_barchart_meta = plot_cardinality_barchart(series_copy, top_k=top_k, data_source=data_source, save_path=report_path, file_name=plot_cardinality_barchart_file_name)

    cardinality = {
        'plot_cardinality_barchart': plot_cardinality_barchart_meta
    }

    # Distribution Analysis
    distribution_result = numeric_distribution_analysis(series_copy, report_path, report_log_id=report_log_id)

    # Generate report
    eda_report = {
        'missing_data': missing_data,
        'cardinality': cardinality,
        'distribution': distribution_result
    }

    full_report = {
        'metadata': {
            'version': '1.0.0',
            'report_name': 'univariate_numeric_analysis',
            'parameters': {
                'series': series.name
            }
        },
        'data': eda_report
    }

    report_path = report_path / f"{series.name.replace(' ', '_')}_univariate_analysis_report.json"
    write_json_report(full_report, report_path)

    logger.info(
        "Completed univariate_numeric_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    return report_path
