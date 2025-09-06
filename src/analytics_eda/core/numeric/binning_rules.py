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
import math
from typing import Tuple
import numpy as np
import pandas as pd

from analytics_eda.core.visualization.validation.validation import numeric_validator

logger = logging.getLogger(__name__)

def choose_bins(series: pd.Series) -> Tuple[int, str]:
    """
    Choose a robust histogram bin count based on data size, shape, and skew.
    Prioritizes Doane, Scott, or Freedman–Diaconis with safe fallbacks.

    Returns:
        (k, rule_name): 
            k = int, number of bins selected
            rule_name = str, name of the rule or heuristic applied
    """
    cleaned_series = numeric_validator().validate(series)
    n = len(cleaned_series)
    if n == 0:
        return 1, "empty"

    # 1) Discrete-integer heuristic
    is_int_like = pd.api.types.is_integer_dtype(cleaned_series) or np.all(np.isclose(cleaned_series, np.round(cleaned_series)))
    n_unique = cleaned_series.nunique()
    if is_int_like and n_unique <= 20:
        return int(max(1, n_unique)), "discrete_integer"

    # 2) Shape stats
    skew = float(cleaned_series.skew()) if n >= 3 else 0.0
    data_range = cleaned_series.max() - cleaned_series.min()
    if data_range <= 0:
        return 1, "constant"

    # 3) Rule selection
    rule_name = ""
    try:
        if n < 50:
            k = doane_bins(series)
            rule_name = "doane"
        elif n < 2000:
            if abs(skew) > 0.5:
                k = freedman_diaconis_bins(series)
                rule_name = "freedman_diaconis"
            else:
                k = scott_bins(series)
                rule_name = "scott"
        else:
            k = freedman_diaconis_bins(series)
            rule_name = "freedman_diaconis"
    except Exception:
        # 4) Fallbacks
        try:
            k = doane_bins(series)
            rule_name = "doane:fallback"
        except Exception:
            try:
                k = sturges_bins(series)
                rule_name = "sturges:fallback"
            except Exception:
                k = max(1, int(np.sqrt(n)))
                rule_name = "sqrt:fallback"

    # 5) Safety caps
    k = int(k)
    k = max(5, k)
    k = min(k, max(10, int(2 * np.sqrt(n)), 200))
    return k, rule_name


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
    cleaned_series = numeric_validator().validate(series)

    n_obs = len(cleaned_series)
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
    cleaned_series = numeric_validator().validate(series)

    # 2. Validate length
    n = len(cleaned_series)
    if n < 2:
        raise ValueError("Series must contain at least two non-NA values.")

    # 3. Compute standard deviation (sample, ddof=1)
    sigma = cleaned_series.std(ddof=1)
    if sigma <= 0:
        raise ValueError("Series must have non-zero variance for Scott's Rule.")

    # 4. Compute bin width and count
    h = 3.5 * sigma / (n ** (1/3))
    data_range = cleaned_series.max() - cleaned_series.min()
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
    cleaned_series = numeric_validator().validate(series)

    # 2. Validate length
    n = len(cleaned_series)
    if n < 2:
        raise ValueError("Series must contain at least two non-NA values.")

    # 3. Compute IQR
    q75, q25 = cleaned_series.quantile(0.75), cleaned_series.quantile(0.25)
    iqr = q75 - q25
    if iqr <= 0:
        raise ValueError("IQR must be positive for Freedman–Diaconis rule.")

    # 4. Calculate bin count
    h = 2 * iqr / (n ** (1/3))
    data_range = cleaned_series.max() - cleaned_series.min()
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
    cleaned_series = numeric_validator().validate(series)

    # 2. Validate length
    n = len(cleaned_series)
    if n < 3:
        raise ValueError("Series must contain at least three values for Doane’s rule.")

    # 3. Compute skewness and its standard error
    g1 = cleaned_series.skew()
    sigma_g1 = math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    # Guard against division by zero in extreme cases
    if sigma_g1 <= 0:
        raise ValueError("Insufficient data variability for Doane’s rule.")

    # 4. Calculate bin count
    k = math.ceil(1 + math.log2(n) + math.log2(1 + abs(g1) / sigma_g1))

    return max(k, 1)
