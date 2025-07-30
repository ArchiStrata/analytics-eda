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
import numpy as np
from scipy import stats

def transform_series(series: pd.Series, method: str) -> pd.Series:
    """
    Apply a specified transform to a numeric Series.

    Supported methods:
      - 'yeo-johnson': Yeo–Johnson (handles negatives)
      - 'arcsinh':     inverse hyperbolic sine
      - 'box-cox':     Box–Cox (x > 0)
      - 'log':         ln(x) (x > 0)
      - 'log1p':       ln(1 + x) (x ≥ –1)
      - 'sqrt':        √x (x ≥ 0)
      - 'reciprocal':  1/x (x ≠ 0)
    """
    method = method.lower()
    supported = {'yeo-johnson','arcsinh','box-cox','log','log1p','sqrt','reciprocal'}
    if method not in supported:
        raise ValueError(f"Unsupported transform '{method}'")

    result = series.copy().astype(float)
    mask   = result.notna()
    arr    = result.loc[mask].to_numpy()

    match method:
        case 'yeo-johnson':
            transformed, _ = stats.yeojohnson(arr)
        case 'arcsinh':
            transformed = np.arcsinh(arr)
        case 'box-cox':
            if (arr <= 0).any():
                raise ValueError("Box–Cox requires x > 0")
            transformed, _ = stats.boxcox(arr)
        case 'log':
            if (arr <= 0).any():
                raise ValueError("Log requires x > 0")
            transformed = np.log(arr)
        case 'log1p':
            if (arr < -1).any():
                raise ValueError("Log1p requires x ≥ -1")
            transformed = np.log1p(arr)
        case 'sqrt':
            if (arr < 0).any():
                raise ValueError("Sqrt requires x ≥ 0")
            transformed = np.sqrt(arr)
        case 'reciprocal':
            if (arr == 0).any():
                raise ValueError("Reciprocal requires x ≠ 0")
            transformed = 1.0 / arr

    result.loc[mask] = transformed
    return result
