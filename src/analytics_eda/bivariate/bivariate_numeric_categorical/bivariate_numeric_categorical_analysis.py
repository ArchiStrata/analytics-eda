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
from pandas.api.types import is_numeric_dtype, is_object_dtype

from ...univariate import univariate_numeric_analysis
from ...core import write_json_report
from .bivariate_numeric_categorical_tests import bivariate_numeric_categorical_tests

logger = logging.getLogger(__name__)

def bivariate_numeric_categorical_analysis(
    df: pd.DataFrame,
    numeric_col: str,
    categorical_col: str,
    report_root: str = 'reports/eda/bivariate/numeric_categorical',
    report_log_id = str(uuid.uuid4()),
    **kwargs
) -> str:
    """
    Run univariate numeric analysis on segments defined by a categorical column.

    Args:
        df (pd.DataFrame): The dataset.
        numeric_col (str): Numeric column to analyze.
        categorical_col (str): Column to segment by.
        report_root (str): Root directory for saving reports.
        report_log_id (str): report log id.
        **kwargs: Additional arguments passed to univariate_numeric_analysis (e.g., alpha, iqr_multiplier).
    
    Returns:
     str: File path to the saved JSON report as written by `write_json_report`.

    Report structure:
        - metadata # Report metadata
        - eda report with statistical test results and per-segment univariate reports.
    """
    logger.info(
        "Starting bivariate_numeric_categorical_analysis",
        extra={
            'numeric_col': numeric_col,
            'categorical_col': categorical_col,
            'report_root': report_root,
            'report_log_id': report_log_id
        }
    )

    if categorical_col not in df.columns:
        raise KeyError(f"Categorical column '{categorical_col}' not found.")
    if numeric_col not in df.columns:
        raise KeyError(f"Numeric column '{numeric_col}' not found.")
    
    if not (isinstance(df[categorical_col].dtype, pd.CategoricalDtype) or is_object_dtype(df[categorical_col])):
        raise TypeError(f"Column '{categorical_col}' must be categorical or object.")
    if not is_numeric_dtype(df[numeric_col]):
        raise TypeError(f"Column '{numeric_col}' must be numeric.")

    report_dir = Path(report_root) / f"{numeric_col}_by_{categorical_col}"
    report_dir.mkdir(parents=True, exist_ok=True)

    statistical_tests = bivariate_numeric_categorical_tests(df, numeric_col, categorical_col, report_log_id=report_log_id)

    segment_reports = {}
    for segment_value, group_df in df.groupby(categorical_col, observed=True):
        segment_name = str(segment_value).replace(" ", "_")
        segment_report_root = report_dir / f"{categorical_col}_{segment_name}"
        logger.debug("Running univariate analysis for segment",
                extra={
                    'segment': segment_value,
                    'numeric_col': numeric_col,
                    'categorical_col': categorical_col,
                    'report_log_id': report_log_id
                })

        try:
            report = univariate_numeric_analysis(
                group_df[numeric_col],
                report_root=segment_report_root,
                report_log_id=report_log_id,
                **kwargs
            )
            segment_reports[segment_value] = report
        except Exception as e:
            # NOTE: If a segment analysis fails we still want to continue with the remaining segements.
            logger.exception(
                "univariate_numeric_analysis failed", 
                extra={
                    'segment': segment_value,
                    'numeric_col': numeric_col,
                    'categorical_col': categorical_col,
                    'report_log_id': report_log_id
                }
            )
            segment_reports[segment_value] = {
                'error': str(e),
                'report_log_id': report_log_id
            }

    eda_report = {
        'statistical_tests': statistical_tests,
        'segments_report': segment_reports
    }

    full_report = {
        'metadata': {
            'version': '0.1.0',
            'report_name': 'bivariate_numeric_categorical_analysis',
            'parameters': {
                'numeric_col': numeric_col,
                'categorical_col': categorical_col
            }
        },
        'eda': eda_report
    }

    report_path = report_dir / f"{numeric_col}_by_{categorical_col}_bivariate_analysis_report.json"
    full_report = write_json_report(full_report, report_path)

    logger.info(
        "Completed bivariate_numeric_categorical_analysis",
        extra={
            'numeric_col': numeric_col,
            'categorical_col': categorical_col,
            'report_log_id': report_log_id,
            'report_path': str(report_path)
        }
    )

    return report_path
