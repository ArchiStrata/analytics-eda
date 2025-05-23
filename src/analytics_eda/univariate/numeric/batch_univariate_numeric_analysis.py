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
import logging
import uuid
import pandas as pd
from .univariate_numeric_analysis import univariate_numeric_analysis

logger = logging.getLogger(__name__)

def batch_univariate_numeric_analysis(
    df: pd.DataFrame,
    columns: list[str],
    report_root: str = 'reports/eda/univariate/numeric',
    report_log_id = str(uuid.uuid4()),
    **analysis_kwargs
) -> dict[str, str]:
    """
    Runs univariate_numeric_analysis on each column in `columns`.

    Returns:
        dict mapping column name → univariate_analysis_report.json file path.
    """
    summary = {}
    for col in columns:
        summary[col] = univariate_numeric_analysis(df[col], report_root, report_log_id=report_log_id, **analysis_kwargs)
    return summary
