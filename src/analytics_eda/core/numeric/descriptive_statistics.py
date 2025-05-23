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

from .is_discrete import is_discrete
from .validate_numeric_named_series import validate_numeric_named_series

logger = logging.getLogger(__name__)

def descriptive_statistics(
    s: pd.Series,
    include_type: bool = False,
    report_log_id: str = str(uuid.uuid4()),
    **kwargs_for_discrete
) -> dict:
    """
    Compute and return key descriptive statistics for a numeric Series.
    Returns an empty dict if there are no non-null values.

    NOTE: All calculations drop NaN values and that kurtosis is "excess kurtosis" per pandas' default.

    Args:
        s (pd.Series): Input Series.
        include_type (bool): if True, runs `is_discrete` on `s` and adds `'is_discrete'` to the output.
        report_log_id (str): report log id.
        **kwargs_for_discrete: passed through to `is_discrete`.

    Returns:
        dict: Descriptive statistics, including:
              - count, mean, median, mode, std, var, min, max, range
              - skewness, kurtosis
              - 10th, 25th, 75th, 90th percentiles
              - mean absolute deviation (mad)
              - coefficient of variation (cv = std/mean; None if mean == 0)
              - number of unique values (nunique)
              Returns {} if s.dropna().empty.
    """
    # 1. Input validation
    validate_numeric_named_series(s)

    logger.info(
        "Starting descriptive_statistics",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    # 2. Drop nulls and short-circuit if empty
    s_clean = s.dropna()
    if s_clean.empty:
        return {}

    # 3. Compute statistics
    count = int(s_clean.count())
    mean = float(s_clean.mean())

    mode_vals = s_clean.mode()
    mode = float(mode_vals.iloc[0]) if not mode_vals.empty else None

    std = float(s_clean.std())
    var = float(s_clean.var())

    mad = float((s_clean - mean).abs().mean())
    cv = float(std / mean) if mean != 0 else None
    nunique = int(s_clean.nunique())

    stats = {
        'count':    count,
        'mean':     mean,
        'median':   float(s_clean.median()),
        'mode':     mode,
        'std':      std,
        'var':      var,
        'min':      float(s_clean.min()),
        'max':      float(s_clean.max()),
        'range':    float(s_clean.max() - s_clean.min()),
        'skewness': float(s_clean.skew()),
        'kurtosis': float(s_clean.kurtosis()),
        'pct_10': float(s_clean.quantile(0.10)),
        'pct_25': float(s_clean.quantile(0.25)),
        'pct_75': float(s_clean.quantile(0.75)),
        'pct_90': float(s_clean.quantile(0.90)),
        'mad':      mad,
        'cv':       cv,
        'nunique':  nunique,
    }

    if include_type:
        stats['is_discrete'] = is_discrete(s, **kwargs_for_discrete)

    logger.info(
        "Completed descriptive_statistics",
        extra={
            'series_name': s.name,
            'report_log_id': report_log_id
        }
    )

    return stats
