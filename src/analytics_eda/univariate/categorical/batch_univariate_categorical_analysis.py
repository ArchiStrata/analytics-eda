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
import pandas as pd
from .univariate_categorical_analysis import univariate_categorical_analysis

def batch_univariate_categorical_analysis(
    df: pd.DataFrame,
    columns: list[str],
    top_n: int = 10,
    report_root: str = 'reports/eda/univariate/categorical'
) -> dict[str, str]:
    """
    Runs univariate_categorical_analysis on each specified column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list[str]): List of categorical column names to analyze.
        top_n (int, optional): Number of top categories to display in each bar plot.
        report_root (str, optional): Root directory for saving report.

    Returns:
        dict[str, str]: Mapping from column name to the univariate_analysis_report.json file path.
    """
    summary = {}
    for col in columns:
        print(f"\n\n==== Univariate Categorical Analysis for '{col}' ====")
        report_file_path = univariate_categorical_analysis(df[col], top_n=top_n, report_root=report_root)
        summary[col] = report_file_path
    return summary
