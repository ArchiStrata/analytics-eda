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
"""Utility functions for exploratory regression analysis (EDA).

This module contains small helpers for resolving input columns, cleaning
and aggregating series, and preparing labels/vectors for plotting.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def resolve_cat_col(df: pd.DataFrame, cols: Sequence[str] | None, role_map: Mapping[str, str] | None) -> str:
    """Resolve the categorical input column name.

    Chooses, in priority order: ``role_map['x']`` → ``cols[0]``. Raises
    ``KeyError`` if the resolved name is missing from *df*.
    """
    col = (role_map or {}).get("x") or (cols[0] if cols else None)
    if not col or col not in df.columns:
        raise KeyError(f"Categorical column '{col}' not found. Provide cols=['<cat>', ...] or role_map['x'].")
    return col


def resolve_num_col(df: pd.DataFrame, cols: Sequence[str] | None, role_map: Mapping[str, str] | None, role: str = "y") -> str:
    """Resolve the numeric column for a given role.

    Defaults to ``role='y'`` for backward compatibility. Chooses, in
    priority order, ``role_map[role]`` → the appropriate index in *cols*
    (0 for ``'x'``, 1 for ``'y'``). Raises ``KeyError`` if not present.
    """
    if role == "y":
        col = (role_map or {}).get("y") or (cols[1] if (cols and len(cols) >= 2) else None)
    elif role == "x":
        col = (role_map or {}).get("x") or (cols[0] if (cols and len(cols) >= 1) else None)
    else:
        raise ValueError(f"Unsupported role '{role}', expected 'x' or 'y'.")

    if not col or col not in df.columns:
        raise KeyError(f"Numeric column for role '{role}' not found. Provided cols={cols}, role_map={role_map}.")
    return col


def dropna_on(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a copy of *df* with rows dropped where *col* is NA."""
    return df.dropna(subset=[col])


def postprocess_series(s: pd.Series, *, min_value: float | None = None, sort_desc: bool = True, top_k: int | None = None) -> pd.Series:
    """Filter/sort/trim a numeric Series for charting.

    Parameters
    ----------
    s : pd.Series
        Input numeric series.
    min_value : float, optional
        Keep only values >= ``min_value``.
    sort_desc : bool, default True
        Sort descending if True; ascending otherwise.
    top_k : int, optional
        Keep only the first ``top_k`` values after sorting.

    Returns
    -------
    pd.Series
        The post-processed series.
    """
    if min_value is not None:
        s = s[s >= min_value]
    if sort_desc:
        s = s.sort_values(ascending=False)
    if top_k and top_k > 0:
        s = s.iloc[:top_k]
    return s


def truncate_labels(labels: list[str], max_len: int | None) -> list[str]:
    """Truncate labels to at most ``max_len`` characters with an ellipsis."""
    if not max_len:
        return labels
    return [lbl if len(lbl) <= max_len else lbl[: max(0, max_len - 1)] + "…" for lbl in labels]


def k_groups_from_series(s: pd.Series) -> int:
    """Return the number of non-null unique values in *s*."""
    return int(pd.Series(s).dropna().nunique())


def agg_count_rows(df: pd.DataFrame, cat: str) -> pd.Series:
    """Count rows per category (including NaN handling via groupby)."""
    return df.groupby(cat, observed=True).size().astype(int)


def agg_count_nonnull(df: pd.DataFrame, cat: str, num: str) -> pd.Series:
    """Count non-null entries of *num* per category *cat*."""
    return df.groupby(cat, observed=True)[num].count().astype(int)


def agg_sum(df: pd.DataFrame, cat: str, num: str) -> pd.Series:
    """Sum *num* per category *cat* (NaNs contribute 0)."""
    # pandas >= 1.1 has min_count; NaNs contribute 0
    return df.groupby(cat, observed=True)[num].sum(min_count=0).fillna(0)


def agg_mean(df: pd.DataFrame, cat: str, num: str) -> pd.Series:
    """Mean of *num* per category *cat*."""
    return df.groupby(cat, observed=True)[num].mean()


def agg_var(df: pd.DataFrame, cat: str, num: str, ddof: int = 1) -> pd.Series:
    """Variance of *num* per category *cat* (with configurable ``ddof``)."""
    return df.groupby(cat, observed=True)[num].var(ddof=ddof)


def grouped_arrays(df: pd.DataFrame, cat: str, num: str) -> list[np.ndarray]:
    """Return a list of numpy arrays of *num*, one per category in *cat*."""
    return [g[num].dropna().to_numpy() for _, g in df.groupby(cat, observed=True)]
