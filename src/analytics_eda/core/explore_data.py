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
import os

import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..core.reporting import write_json_report


def explore_data(
        df: pd.DataFrame,
        report_path: str = None,
        file_name: str = "explore_data_summary.json") -> dict:
    """
    Performs a structured exploratory summary of the given DataFrame and saves it to a JSON file.

    Sections:
    - Overview:
        - Shape
        - Duplicate Rows
        - Memory Usage
    - Per Column Drill-Down:
        - dtype
        - missing values
        - is_constant
        - is_high_cardinality
        - numeric: descriptive stats + 4-sigma outlier detection
        - category: unique count and values
        - object: unique count
    """
    summary = {
        "overview": {},
        "columns": {}
    }

    # ----------- Overview -----------
    summary["overview"]["shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

    summary["overview"]["duplicate_rows"] = int(df.duplicated().sum())

    summary["overview"]["memory_usage_bytes"] = {
        col: int(mem) for col, mem in df.memory_usage(deep=True).items()
    }

    n_rows = df.shape[0]

    # ----------- Per Column Drill-Down -----------
    for col in df.columns:
        col_summary = {}
        col_data = df[col]

        col_summary["dtype"] = str(col_data.dtype)
        col_summary["missing_values"] = int(col_data.isna().sum())

        # Constant column
        col_summary["is_constant"] = (col_data.nunique(dropna=False) == 1)

        # High Cardinality column
        unique_ratio = col_data.nunique(dropna=True) / n_rows
        col_summary["is_high_cardinality"] = (unique_ratio > 0.9)

        if is_numeric_dtype(col_data):
            desc = col_data.describe().to_dict()
            mean = col_data.mean()
            std = col_data.std()
            threshold = 4 * std
            lower_bound = mean - threshold
            upper_bound = mean + threshold

            lower_outliers = col_data[col_data < lower_bound]
            upper_outliers = col_data[col_data > upper_bound]

            col_summary["numeric_analysis"] = {
                "descriptive_stats": desc,
                "extreme_outliers_4sigma": {
                    "total_count": int(lower_outliers.count() + upper_outliers.count()),
                    "lower": {
                        "count": int(lower_outliers.count()),
                        "min": lower_outliers.min() if not lower_outliers.empty else None,
                        "max": lower_outliers.max() if not lower_outliers.empty else None
                    },
                    "upper": {
                        "count": int(upper_outliers.count()),
                        "min": upper_outliers.min() if not upper_outliers.empty else None,
                        "max": upper_outliers.max() if not upper_outliers.empty else None
                    }
                }
            }

        elif col_data.dtype == "category":
            col_summary["category_analysis"] = {
                "unique_count": int(col_data.nunique()),
                "unique_values": col_data.dropna().unique().tolist()
            }

        elif col_data.dtype == "object":
            col_summary["object_analysis"] = {
                "unique_count": int(col_data.nunique())
            }

        summary["columns"][col] = col_summary

    # write JSON report
    if report_path is not None:
        full_report_path = os.path.join(report_path, file_name)
        return write_json_report(summary, full_report_path)

    return summary
