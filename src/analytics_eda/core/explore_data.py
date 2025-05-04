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
from ..core import write_json_report

def explore_data(
        df: pd.DataFrame,
        n_head: int = 5,
        report_path: str = "reports/eda/explore_data_summary.json") -> dict:
    """
    Performs a detailed exploratory summary of the given DataFrame
    and saves it to a JSON file.

    Args:
        df (pd.DataFrame): The input DataFrame to explore.
        n_head (int, optional): Number of rows to include in the sample. Defaults to 5.
        report_path (str, optional): Where to write the JSON summary. Defaults to 'reports/eda/explore_data_summary.json'.

    Summary:
        - DataFrame shape (rows × columns)
        - Column names and data types
        - Missing values summary
        - Duplicate row count
        - Constant columns (columns with a single unique value)
        - High cardinality columns (>90% unique values)
        - Statistical summary for numeric columns
        - Unique value counts for object and category columns
        - Unique values preview for each column
        - Memory usage by column
        - Sample data (head)
    
    Returns:
        dict: A nested dictionary containing summary.
    """
    # 1. Build the summary dict
    summary = {}

    # Shape
    summary["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}

    # Column names and dtypes
    summary["columns"] = list(df.columns)
    summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing = df.isna().sum()
    summary["missing_values"] = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}

    # Duplicate rows
    summary["duplicate_count"] = int(df.duplicated().sum())

    # Constant columns
    constant = [col for col in df.columns if df[col].nunique() == 1]
    summary["constant_columns"] = constant

    # High Cardinality (>90% unique)
    high_card = [col for col in df.columns if df[col].nunique() > 0.9 * df.shape[0]]
    summary["high_cardinality_columns"] = high_card

    # Statistical summary
    desc = df.describe(include="all").to_dict()
    summary["statistical_summary"] = desc

    # Object & category unique counts
    obj_cols = df.select_dtypes(include="object").columns
    summary["object_unique_counts"] = {col: int(df[col].nunique()) for col in obj_cols}

    cat_cols = df.select_dtypes(include="category").columns
    cat_cols = df.select_dtypes(include="category").columns
    summary["category_unique_details"] = {
        col: {
            "unique_count": int(df[col].nunique()),
            "unique_values": df[col].dropna().unique().tolist()
        }
        for col in cat_cols
    }

    # Unique-values preview
    preview = {}
    for col in df.columns:
        uniques = df[col].dropna().unique().tolist()
        preview[col] = {
            "count": int(len(uniques)),
            "preview": uniques[:5]
        }
    summary["unique_values_preview"] = preview

    # Memory usage
    summary["memory_usage_bytes"] = {col: int(mem) for col, mem in df.memory_usage(deep=True).items()}

    # Sample data
    summary["sample_data"] = df.head(n_head).to_dict(orient="records")

    # 2. write JSON report
    clean_summary = write_json_report(summary, report_path)

    return clean_summary
