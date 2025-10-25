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
"""Formatting utilities for reports (values, percents, p-values, df, etc.)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from typing import Any, Literal

import numpy as np
import pandas as pd

DfLike = int | float | tuple[int | float, int | float]


class ReportFormatter:
    """
    Formatting utilities for findings, subtitles, annotations, and tables.

    If an instance has a `ctx` attribute (e.g., set by BasePlot), the following
    optional context defaults will be used unless overridden per call:
      - ctx.report_default_decimals : int   (default: 2)
      - ctx.report_default_unit     : str   (default: None)
    """

    def __init__(self, *, ctx: Any | None = None) -> None:
        self.ctx = ctx

    # ---------- Generic numeric & percent ----------

    def format_numeric_value(
        self,
        value: float | None,
        *,
        decimals: int | None = None,
        unit: str | None = None,
    ) -> str:
        """
        Format a numeric value for reporting.

        - Uses ctx.report_default_decimals / ctx.report_default_unit when available,
          unless explicitly overridden via parameters.
        - Returns "NA" for None/NaN.
        """
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "NA"

        ctx_dec = getattr(self.ctx, "report_default_decimals", 2)
        ctx_unit = getattr(self.ctx, "report_default_unit", None)

        d = ctx_dec if decimals is None else decimals
        u = ctx_unit if unit is None else unit

        s = (
            f"{int(value)}"
            # suppress decimals for exact integers when decimals=0
            if isinstance(value, int) and (decimals is None or decimals == 0)
            else f"{value:.{int(d)}f}"
        )
        return f"{s} {u}" if u else s

    def format_percent(
        self,
        p: float | None,
        *,
        decimals: int = 1,
        scale_0to1: bool = True,
    ) -> str:
        """
        Format a percentage.

        - If `scale_0to1=True`, treats inputs like 0.123 as 12.3%.
        - Returns "NA" for None/NaN.
        """
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return "NA"

        val = p if scale_0to1 else (p / 100.0)
        return f"{val:.{int(decimals)}%}"

    def max_decimals_in_series(self, s: pd.Series) -> int:
        """Infer the maximum number of decimal places in raw numeric values.

        Integers -> 0. Floats are parsed via Decimal(str(...)) to avoid binary fp artifacts.
        """
        def dec_count(v) -> int:
            # ints (including numpy ints)
            if isinstance(v, (int, np.integer)):
                return 0
            # floats or numerics coerced to string
            try:
                d = Decimal(str(v)).normalize()
                exp = d.as_tuple().exponent
                # exponent is negative for decimal places; e.g., 1.230 -> exponent -3
                return max(-exp, 0)
            except (InvalidOperation, ValueError, TypeError):
                return 0

        x = s.dropna()
        if x.empty:
            return 0
        return int(max(dec_count(v) for v in x))

    def mean_decimals_from_series(self, s: pd.Series) -> int:
        """Return decimals for displaying the mean.

        Rule: show the mean to one more decimal than the most precise raw value.
        Empty series defaults to 1 decimal.

        Examples
        --------
        raw values as integers  -> returns 1
        raw values to tenths    -> returns 2
        raw values to hundredth -> returns 3
        """
        raw_max = self.max_decimals_in_series(s.dropna())
        # Apply the rounding rule for the mean: display it to one more decimal place than the
        # most precise raw data value. For example, if inputs are whole numbers → 1 decimal;
        # if inputs are to tenths → 2 decimals. We infer the maximum raw precision, then add 1.
        # (Affects formatting only; the underlying mean value is unchanged.)
        return (raw_max + 1) if raw_max is not None else 1

    def median_decimals_from_series(self, s: pd.Series, median_value: float | None = None) -> int:
        """Return decimals for displaying the median.

        Uses the maximum raw precision in `s`. For even-length series where the
        median is the average of two values (and a `median_value` is provided),
        increases precision by one if the median is not equal to either middle value.
        """
        x = s.dropna()
        if x.empty:
            return getattr(self, "report_default_decimals", 2)
        raw_max = self.max_decimals_in_series(x)
        n = len(x)
        if n % 2 == 0 and median_value is not None:
            xs = np.sort(np.asarray(x, dtype=float))
            m1, m2 = xs[n//2 - 1], xs[n//2]
            if not (np.isclose(median_value, m1) or np.isclose(median_value, m2)):
                return raw_max + 1
        return raw_max

    # ---------- Alpha & p-values ----------

    def format_alpha(
        self,
        alpha: float | None,
        *,
        style: Literal["decimal", "percent"] = "decimal",
        tol: float = 1e-9,
    ) -> str:
        """
        Format a significance level (alpha) using conventional precision rules.

        Rules:
        - Snap exact common levels:
            0.05 -> "0.05" / "5%"
            0.01 -> "0.01" / "1%"
            0.001 -> "0.001" / "0.1%"
        - Otherwise:
            * decimal: up to 3 decimals (trim trailing zeros)
            * percent: up to 2 decimals (trim trailing zeros), plus '%'

        Returns "NA" for None/NaN.
        """
        if alpha is None or (isinstance(alpha, float) and math.isnan(alpha)):
            return "NA"

        if abs(alpha - 0.05) < tol:
            return "5%" if style == "percent" else "0.05"
        if abs(alpha - 0.01) < tol:
            return "1%" if style == "percent" else "0.01"
        if abs(alpha - 0.001) < tol:
            return "0.1%" if style == "percent" else "0.001"

        if style == "percent":
            pct = alpha * 100.0
            s = f"{pct:.2f}".rstrip("0").rstrip(".")
            return f"{s}%"

        # style == "decimal"
        s = f"{alpha:.3f}".rstrip("0").rstrip(".")
        return s

    def format_p_value(
        self,
        p: float | None,
        *,
        decimals: int = 3,
        sci_threshold: float = 1e-4,
    ) -> str:
        """
        Format a p-value using common reporting standards.

        Rules:
        - None/NaN -> 'NA'
        - p == 0   -> 'p < 0.00...1' (10^-decimals)
        - 0 < p < 0.001:
            * if p < sci_threshold -> scientific notation ('p = 1.23e-05')
            * else -> 'p < 0.001'
        - p >= 0.001 -> 'p = {value}' to `decimals` places (trim trailing zeros)
        """
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return "NA"

        if p == 0:
            return "p < " + f"{10**-decimals:.{decimals}f}"

        if 0 < p < 0.001:
            if p < sci_threshold:
                return f"p = {p:.{decimals}e}"
            return "p < 0.001"

        return f"p = {p:.{decimals}f}".rstrip("0").rstrip(".")

    def format_test_statistic(
        self,
        value: float | None,
        *,
        decimals: int = 2,
    ) -> str:
        """
        Format a test statistic (e.g., Chi-square, t, F, z).

        - Defaults to 2 decimals (APA-style).
        - Returns "NA" for None/NaN.
        """
        return self.format_numeric_value(value, decimals=decimals, unit=None)

    # ---------- Degrees of freedom ----------

    def format_df(
        self,
        df: DfLike | None,
        *,
        decimals_for_fractional: int = 1,
        infinity_symbol: str = "∞",
    ) -> str:
        """
        Format degrees of freedom (df) for reporting.

        Rules:
        - Integer df -> no decimals (e.g., 'df = 28').
        - Fractional df (Welch/Satterthwaite) -> 1 decimal by default (configurable).
        - Paired df (F-tests) -> 'df1, df2' inside parentheses (e.g., 'F(2, 45)').
        - None/NaN -> 'df = NA'.
        - Infinite -> 'df = ∞' (configurable symbol).

        Accepts:
        - int or float: single df
        - (df1, df2): tuple for F-tests
        """
        def _is_nan(x) -> bool:
            return isinstance(x, float) and math.isnan(x)

        def _fmt_one(x: int | float) -> str:
            if x is None or _is_nan(x):
                return "NA"
            if math.isinf(x):
                return infinity_symbol
            # exact integer?
            if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer()):
                return f"{int(x)}"
            # fractional df -> limited decimals
            return f"{float(x):.{int(decimals_for_fractional)}f}"

        if df is None:
            return "df = NA"

        # Paired df (e.g., F-tests)
        if isinstance(df, (tuple, list)) and len(df) == 2:
            df1, df2 = df
            return f"df = ({_fmt_one(df1)}, {_fmt_one(df2)})"

        # Single df
        return f"df = {_fmt_one(df)}"
