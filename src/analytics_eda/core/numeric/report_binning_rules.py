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
from typing import Any, Dict, Sequence
import logging
import uuid
import math
import pandas as pd
import numpy as np

from .validate_numeric_named_series import validate_numeric_named_series
from ..clean_series import clean_series

logger = logging.getLogger(__name__)

def report_binning_rules(
    series: pd.Series,
    is_discrete: bool,
    rules: Sequence[str] = ('sturges', 'scott', 'freedman-diaconis', 'doane'),
    report_log_id: str = str(uuid.uuid4())
) -> Dict[str, Dict[str, Any]]:
    """
    Generate frequency/binning reports for a numeric Series.

    If `is_discrete` is True, returns raw value counts.
    Otherwise, compares multiple histogram binning rules.

    Args:
        series (pd.Series): Numeric data; may contain NaNs.
        is_discrete (bool): Flag indicating discrete vs. continuous data.
        rules (Sequence[str]): Binning rules to apply when continuous.
            Supported: 'sturges', 'scott', 'freedman-diaconis', 'doane'.
        report_log_id (str): report log id.

    Returns:
        Dict[str, Dict[str, Any]]:
            - If discrete: {'value_counts': {value: count, ...}}
            - If continuous: {
                rule_name: {
                    'n_bins': int,
                    'edges': List[float],
                    'counts': List[int]
                },
                ...
              }

    Raises:
        TypeError: If `series` is not a pandas Series or not numeric.
        ValueError: If the cleaned Series is empty after dropping NAs.
    """
    # 1. Validate input
    validate_numeric_named_series(series)

    logger.info(
        "Starting report_binning_rules",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    clean = clean_series(series)

    # 2. Discrete case: raw frequencies
    if is_discrete:
        counts = clean.value_counts().sort_index()
        return {'value_counts': counts.to_dict()}

    report: Dict[str, Dict[str, Any]] = {}

    # 3. Continuous case: compare binning rules
    report: Dict[str, Dict[str, Any]] = {}
    for rule in rules:
        try:
            n_bins, edges, counts = compute_bin_rule(clean, rule=rule)
            report[rule] = {
                'n_bins': n_bins,
                'edges': edges.tolist(),
                'counts': counts.tolist(),
            }
        except Exception as e:
            logger.exception(
                "report_binning_rules failed", 
                extra={
                    'series_name': series.name,
                    'report_log_id': report_log_id
                }
            )
            report[rule] = {
                'error': str(e),
                'report_log_id': report_log_id
            }

    logger.info(
        "Completed report_binning_rules",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )
    return report

def compute_bin_rule(
    series: pd.Series,
    rule: str = 'sturges'
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Compute histogram bin count (k), edges, and counts for a numeric Series.

    Args:
        series (pd.Series): Numeric data with possible NaNs.
        rule (str): Binning rule to apply:
            'sturges', 'scott', 'freedman-diaconis', or 'doane'.

    Returns:
        tuple:
            k (int): Number of bins.
            edges (np.ndarray): Bin edges.
            counts (np.ndarray): Frequency counts per bin.

    Raises:
        TypeError: If `series` is not a pandas Series or not numeric.
        ValueError: If the cleaned Series is empty or rule is unknown.
    """
    # 1. Validate input
    validate_numeric_named_series(series)

    clean = clean_series(series)

    # 2. Determine number of bins
    if rule == 'sturges':
        k = sturges_bins(clean)
    elif rule == 'scott':
        k = scott_bins(clean)
    elif rule == 'freedman-diaconis':
        k = freedman_diaconis_bins(clean)
    elif rule == 'doane':
        k = doane_bins(clean)
    else:
        raise ValueError(f"Unknown binning rule: {rule!r}")

    # 3. Compute edges and counts
    edges = np.histogram_bin_edges(clean, bins=k)
    counts, _ = np.histogram(clean, bins=edges)

    return k, edges, counts

def sturges_bins(series: pd.Series) -> int:
    """
    Compute number of histogram bins using Sturges' Rule.

    Args:
        series (pd.Series): Numeric data. NAs will be dropped.

    Returns:
        int: Number of bins, k = ceil(log2(n_obs) + 1).

    Raises:
        ValueError: If n_obs < 1.
    """
    # 1. Validate input
    validate_numeric_named_series(series)

    clean = clean_series(series)

    n_obs = len(clean)
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    return math.ceil(math.log2(n_obs) + 1)

def scott_bins(series: pd.Series) -> int:
    """
    Compute the number of histogram bins using Scott's Rule.

    Scott's Rule chooses bin width as:
        h = 3.5 * σ / n^(1/3)
    where σ is the sample standard deviation (ddof=1) and n is the number of observations.
    The number of bins k is then:
        k = ⌈(max - min) / h⌉

    Args:
        series (pd.Series): Numeric data. NAs will be dropped.

    Returns:
        int: Number of bins according to Scott's Rule.

    Raises:
        TypeError: If `series` is not a pandas Series or not numeric.
        ValueError: If `series` has fewer than 2 non-NA observations or zero variance.
    """
    # 1. Validate input
    validate_numeric_named_series(series)

    clean = clean_series(series)

    # 2. Validate length
    n = len(clean)
    if n < 2:
        raise ValueError("Series must contain at least two non-NA values.")

    # 3. Compute standard deviation (sample, ddof=1)
    sigma = clean.std(ddof=1)
    if sigma <= 0:
        raise ValueError("Series must have non-zero variance for Scott's Rule.")

    # 4. Compute bin width and count
    h = 3.5 * sigma / (n ** (1/3))
    data_range = clean.max() - clean.min()
    k = math.ceil(data_range / h)

    return k

def freedman_diaconis_bins(series: pd.Series) -> int:
    """
    Compute number of histogram bins using the Freedman–Diaconis rule.

    The rule sets bin width as:
        h = 2 * IQR / n^(1/3)
    where IQR = Q3 - Q1 and n is the number of observations.
    The number of bins k is then:
        k = ceil((max - min) / h)

    Args:
        series (pd.Series): Numeric data with NAs already dropped.

    Returns:
        int: Number of bins according to Freedman–Diaconis.

    Raises:
        TypeError: If `series` is not a pandas Series or not numeric.
        ValueError: If `series` has fewer than 2 values or IQR is zero.
    """
    # 1. Validate input
    validate_numeric_named_series(series)

    clean = clean_series(series)

    # 2. Validate length
    n = len(clean)
    if n < 2:
        raise ValueError("Series must contain at least two non-NA values.")

    # 3. Compute IQR
    q75, q25 = clean.quantile(0.75), clean.quantile(0.25)
    iqr = q75 - q25
    if iqr <= 0:
        raise ValueError("IQR must be positive for Freedman–Diaconis rule.")

    # 4. Calculate bin count
    h = 2 * iqr / (n ** (1/3))
    data_range = clean.max() - clean.min()
    k = math.ceil(data_range / h)

    return max(k, 1)

def doane_bins(series: pd.Series) -> int:
    """
    Compute number of histogram bins using Doane’s Rule.

    Doane’s Rule adjusts Sturges’ formula for non-normality:
        k = ceil(1 + log2(n) + log2(1 + |g1|/σ_g1))
    where:
        - n is the number of observations
        - g1 is the sample skewness
        - σ_g1 = sqrt(6*(n-2)/((n+1)*(n+3)))

    Args:
        series (pd.Series): Numeric data with NAs already dropped.

    Returns:
        int: Number of bins according to Doane’s Rule.

    Raises:
        TypeError: If `series` is not a pandas Series or not numeric.
        ValueError: If `series` has fewer than three values or zero variance affecting skewness.
    """
    # 1. Validate input
    validate_numeric_named_series(series)

    clean = clean_series(series)

    # 2. Validate length
    n = len(clean)
    if n < 3:
        raise ValueError("Series must contain at least three values for Doane’s rule.")

    # 3. Compute skewness and its standard error
    g1 = clean.skew()
    sigma_g1 = math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    # Guard against division by zero in extreme cases
    if sigma_g1 <= 0:
        raise ValueError("Insufficient data variability for Doane’s rule.")

    # 4. Calculate bin count
    k = math.ceil(1 + math.log2(n) + math.log2(1 + abs(g1) / sigma_g1))

    return max(k, 1)
