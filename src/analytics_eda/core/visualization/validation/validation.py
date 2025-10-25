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
"""Validation helpers for visualization inputs.

This module defines validators for Pandas Series and DataFrames that enforce
lightweight structural/dtype constraints used by plotting utilities.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype


class SeriesKind(str, Enum):
    """Kinds of Series accepted by visualization routines."""

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    NAMED = "named"          # no dtype check; just the name requirement


@dataclass(frozen=True)
class SeriesValidator:
    """Validator for a single Pandas Series.

    Attributes
    ----------
        kind: Expected logical kind of the Series.
        require_name: Whether a non-empty `name` is required.
        dropna: Drop NA values before returning.
        reset_index: Reset the index after filtering.
        throw_if_empty: Raise if the post-processed Series is empty.
        cast_str: If categorical/named, cast values to `str`.
        coerce_numeric: If numeric, coerce with `pd.to_numeric(errors="coerce")`.
    """

    # What kind of series this plot expects?
    kind: SeriesKind

    # Common knobs
    require_name: bool = True
    dropna: bool = True
    reset_index: bool = True
    throw_if_empty: bool = False

    # CATEGORICAL / NAMED specific
    cast_str: bool = True     # convert values to str (useful for hue/labels/ticks)

    # NUMERIC specific
    coerce_numeric: bool = False  # try to coerce to numeric with pd.to_numeric

    def validate(self, s: pd.Series) -> pd.Series:
        """Validate and (optionally) normalize a Series.

        Applies name checks, dtype checks by `kind`, optional coercions and
        basic cleanup (`dropna`, `reset_index`).

        Args:
            s: Input Series.

        Returns
        -------
            A validated copy of `s`, potentially coerced and cleaned.

        Raises
        ------
            TypeError: If `s` is not a Series or has an incompatible dtype.
            ValueError: If `require_name` is True and the name is empty,
                if `kind` is unsupported, or if `throw_if_empty` and result is empty.
        """
        # ---- common structural checks ----
        if not isinstance(s, pd.Series):
            raise TypeError("Input must be a pandas Series.")

        if self.require_name and (s.name is None or str(s.name).strip() == ""):
            raise ValueError("Series must have a non-empty 'name' attribute.")

        out = s.copy(deep=True)

        # ---- kind-specific checks & optional coercions ----
        if self.kind == SeriesKind.CATEGORICAL:
            if not (isinstance(out.dtype, pd.CategoricalDtype) or is_object_dtype(out)):
                raise TypeError(
                    f"Series '{out.name}' must be categorical (or object) for categorical analysis."
                )
            if self.cast_str:
                out = out.astype(str)

        elif self.kind == SeriesKind.NUMERIC:
            if self.coerce_numeric:
                out = pd.to_numeric(out, errors="coerce")
            if not is_numeric_dtype(out):
                raise TypeError("Series must be numeric (int or float dtype).")

        elif self.kind == SeriesKind.NAMED:
            # Only the name requirement is enforced; dtype is unrestricted.
            if self.cast_str:
                out = out.astype(str)

        else:
            raise ValueError(f"Unsupported SeriesKind: {self.kind}")

        if self.dropna:
            out = out.dropna()

        if self.reset_index:
            out = out.reset_index(drop=True)

        if self.throw_if_empty:
            if out.empty:
                raise ValueError("Series is empty.")

        return out


# Convenience factories (tiny wrappers for readability at call sites)
def categorical_validator(
    *,
    require_name: bool = True,
    dropna: bool = True,
    reset_index: bool = True,
    throw_if_empty: bool = False,
    cast_str: bool = True,
) -> SeriesValidator:
    """Create a `SeriesValidator` for categorical (or object) Series.

    Args:
        require_name: Enforce a non-empty `Series.name`.
        dropna: Drop NA values.
        reset_index: Reset index after filtering.
        throw_if_empty: Raise if result is empty.
        cast_str: Cast values to `str`.

    Returns
    -------
        Configured `SeriesValidator` for categorical inputs.
    """
    return SeriesValidator(
        kind=SeriesKind.CATEGORICAL,
        require_name=require_name,
        dropna=dropna,
        reset_index=reset_index,
        throw_if_empty=throw_if_empty,
        cast_str=cast_str,
    )


def numeric_validator(
    *,
    require_name: bool = True,
    dropna: bool = True,
    reset_index: bool = True,
    throw_if_empty: bool = False,
    coerce_numeric: bool = False,
) -> SeriesValidator:
    """Create a `SeriesValidator` for numeric Series.

    Args:
        require_name: Enforce a non-empty `Series.name`.
        dropna: Drop NA values.
        reset_index: Reset index after filtering.
        throw_if_empty: Raise if result is empty.
        coerce_numeric: Coerce with `pd.to_numeric(errors="coerce")` before checks.

    Returns
    -------
        Configured `SeriesValidator` for numeric inputs.
    """
    return SeriesValidator(
        kind=SeriesKind.NUMERIC,
        require_name=require_name,
        dropna=dropna,
        reset_index=reset_index,
        throw_if_empty=throw_if_empty,
        coerce_numeric=coerce_numeric,
    )


def named_only_validator(
    *,
    require_name: bool = True,
    dropna: bool = True,
    reset_index: bool = True,
    throw_if_empty: bool = False,
    cast_str: bool = True,
) -> SeriesValidator:
    """Create a `SeriesValidator` that only enforces a non-empty name.

    Args:
        require_name: Enforce a non-empty `Series.name`.
        dropna: Drop NA values.
        reset_index: Reset index after filtering.
        throw_if_empty: Raise if result is empty.
        cast_str: Cast values to `str`.

    Returns
    -------
        Configured `SeriesValidator` for name-only enforcement.
    """
    return SeriesValidator(
        kind=SeriesKind.NAMED,
        require_name=require_name,
        dropna=dropna,
        reset_index=reset_index,
        throw_if_empty=throw_if_empty,
        cast_str=cast_str,
    )


class FrameValidator(Protocol):
    """Protocol for DataFrame validation strategies."""

    def validate(
        self,
        df: pd.DataFrame,
        *,
        cols: Sequence[str],
        role_map: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """Validate and possibly normalize a DataFrame for plotting.

        Args:
            df: Input DataFrame.
            cols: Columns required/expected by the visualization.
            role_map: Optional mapping from logical role to column name.

        Returns
        -------
            A validated DataFrame, potentially filtered or renamed.

        Raises
        ------
            KeyError: If required columns are missing.
            ValueError: If roles/columns are inconsistent.
        """
        ...

