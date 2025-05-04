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
from pandas.api.types import is_numeric_dtype

def validate_numeric_named_series(
    series: pd.Series,
    require_name: bool = True
) -> pd.Series:
    """
    Validate that the input is a numeric pandas Series with a non-empty name.

    Args:
        series (pd.Series): The Series to validate.
        require_name (bool): If True, series.name must be non-empty.

    Returns:
        pd.Series: The validated Series.

    Raises:
        TypeError: If `series` is not a pandas Series or not numeric dtype.
        ValueError: If `require_name` is True and `series.name` is None or empty.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    if not is_numeric_dtype(series):
        raise TypeError("Series must be numeric (int or float dtype).")
    if require_name and (series.name is None or str(series.name).strip() == ""):
        raise ValueError("Series must have a non-empty 'name' attribute.")
    return series
