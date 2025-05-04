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
from pandas.api.types import is_integer_dtype, is_float_dtype

def is_discrete(series: pd.Series,
                max_unique_fraction: float = 0.05,
                integer_tolerance: bool = True) -> bool:
    """
    Heuristically determine if a numeric Series is discrete.
    
    Args:
        series: numeric pd.Series
        max_unique_fraction: if (n_unique / len) < threshold, treat as discrete
        integer_tolerance: for float series, if all values .is_integer(), treat as discrete
    
    Returns:
        True if likely discrete, False if likely continuous.
    
    Raises:
        ValueError: if the series is empty after dropping NAs.
        TypeError: if the series is not int or float dtype.
    """
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty after dropping NA")

    # 1. Integer dtype -> discrete
    if is_integer_dtype(s.dtype):
        return True

    # 2. Float dtype -> maybe discrete
    if is_float_dtype(s.dtype):
        # 2a. All values are whole numbers?
        if integer_tolerance and ((s % 1) == 0).all():
            return True
        # 2b. Very few unique values?
        frac_unique = s.nunique() / len(s)
        return frac_unique < max_unique_fraction

    # Non-numeric dtype: raise or decide separately
    raise TypeError("Series must be int or float dtype.")
