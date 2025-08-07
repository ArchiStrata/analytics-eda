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

import inspect
from typing import Any, Callable, Dict, Optional

import pandas as pd


def call_plot_with_overrides(
    plot_func: Callable[..., Dict[str, Any]],
    series: pd.Series,
    overrides: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    data_source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic wrapper to call a plotting function with overrideable kwargs.

    Steps:
      1. Inspect the target func’s signature.
      2. Start with its default parameter values.
      3. Apply any other overrides (error on unknown keys).
      4. Invoke plot_func(series, **kwargs, save_path=save_path).
    """
    overrides = overrides.copy() if overrides else {}

    # 1. inspect signature
    sig = inspect.signature(plot_func)
    forbidden = {"series", "save_path", "data_source"}
    allowed = {p for p in sig.parameters if p not in forbidden}

    # 2. start with defaults
    plot_kwargs: Dict[str, Any] = {
        name: param.default
        for name, param in sig.parameters.items()
        if name in allowed
    }

    # 3. apply remaining overrides
    for key, val in overrides.items():
        if key not in allowed:
            raise KeyError(f"'{key}' is not a valid parameter for {plot_func.__name__}")
        plot_kwargs[key] = val

    # 4. call the plot function
    return plot_func(
        series,
        **plot_kwargs,
        save_path=save_path,
        data_source=data_source,
    )
