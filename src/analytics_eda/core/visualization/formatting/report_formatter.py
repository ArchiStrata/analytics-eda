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
        cap = int(getattr(self.ctx, "report_max_decimals", 6))
        ctx_unit = getattr(self.ctx, "report_default_unit", None)

        d = ctx_dec if decimals is None else min(int(decimals), cap)
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
        The result is capped by `report_max_decimals` to prevent over-precision.
        """
        cap = int(getattr(self, "report_max_decimals", 6))

        def dec_count(v) -> int:
            if isinstance(v, int | np.integer):
                return 0
            try:
                d = Decimal(str(v)).normalize()
                exp = d.as_tuple().exponent
                return max(-exp, 0)
            except (InvalidOperation, ValueError, TypeError):
                return 0

        x = s.dropna()
        if x.empty:
            return 0
        raw = int(max(dec_count(v) for v in x))
        return min(raw, cap)

    def format_mean(
        self,
        s: pd.Series,
        mean_value: float | None = None,
    ) -> tuple[float | None, int, str]:
        """
        Return (mean, mean_round_decimals, mean_formatted) for a numeric Series.

        - If `mean_value` is provided, it is used; otherwise the mean is computed
        from finite numeric values in `s`. Empty/invalid -> mean=None.
        - Decimal places follow `mean_decimals_from_series(s)`.
        - Formatted string uses `format_numeric_value(...)` and respects ctx defaults.
        """
        # Coerce to numeric and keep only finite values for precision inference / mean
        nums = pd.to_numeric(s, errors="coerce")
        nums = nums[np.isfinite(nums)]

        mean = mean_value if mean_value is not None else (float(nums.mean()) if not nums.empty else None)
        mean_round_decimals = int(self.mean_decimals_from_series(nums))
        mean_formatted = self.format_numeric_value(mean, decimals=mean_round_decimals)

        return mean, mean_round_decimals, mean_formatted

    def mean_decimals_from_series(self, s: pd.Series) -> int:
        """Show mean to one more decimal than the most precise raw value, capped."""
        cap = int(getattr(self, "report_max_decimals", 6))
        default = int(getattr(self, "report_default_decimals", 2))
        raw_max = self.max_decimals_in_series(s.dropna())
        base = (raw_max + 1) if raw_max is not None else default
        return min(base, cap)

    def median_decimals_from_series(self, s: pd.Series, median_value: float | None = None) -> int:
        """Use max raw precision; for even-n, add 1 if median is an average of distinct middles; cap."""
        cap = int(getattr(self, "report_max_decimals", 6))
        default = int(getattr(self, "report_default_decimals", 2))

        x = s.dropna()
        if x.empty:
            return default

        raw_max = self.max_decimals_in_series(x)
        n = len(x)
        out = raw_max
        if n % 2 == 0 and median_value is not None:
            xs = np.sort(np.asarray(x, dtype=float))
            m1, m2 = xs[n // 2 - 1], xs[n // 2]
            if not (np.isclose(median_value, m1) or np.isclose(median_value, m2)):
                out = raw_max + 1
        return min(out, cap)

    def format_median(
        self,
        s: pd.Series,
        median_value: float | None = None,
    ) -> tuple[float | None, int, str]:
        """
        Return (median, median_round_decimals, median_formatted) for a numeric Series.

        - If `median_value` is provided, use it; otherwise compute from finite numeric values in `s`.
        - Decimal places follow `median_decimals_from_series(s, median_value)`.
        - Formatted string uses `format_numeric_value(...)` and respects ctx defaults.
        """
        nums = pd.to_numeric(s, errors="coerce")
        nums = nums[np.isfinite(nums)]

        median = median_value if median_value is not None else (float(nums.median()) if not nums.empty else None)
        median_round_decimals = int(self.median_decimals_from_series(nums, median))
        median_formatted = self.format_numeric_value(median, decimals=median_round_decimals)

        return median, median_round_decimals, median_formatted

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

    def format_cohens_d(
        self,
        d: float | None,
        *,
        decimals: int | None = None,
        tiny_threshold: float = 0.10,
    ) -> str:
        """
        Format Cohen's d with sensible defaults.

        Rules:
        - None/NaN -> "NA"
        - Default 2 decimals; if |d| < tiny_threshold and no explicit `decimals`,
            use 3 decimals to avoid "0.00".
        - Leading zero for |d| < 1, sign preserved.
        """
        if d is None or (isinstance(d, float) and math.isnan(d)):
            return "NA"

        # Decide precision
        dec = 2 if decimals is None else int(decimals)
        if decimals is None and abs(d) < tiny_threshold:
            dec = 3

        # Use the existing numeric formatter to respect ctx defaults (unit=None)
        return self.format_numeric_value(float(d), decimals=dec, unit=None)

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
        if isinstance(df, tuple | list) and len(df) == 2:
            df1, df2 = df
            return f"df = ({_fmt_one(df1)}, {_fmt_one(df2)})"

        # Single df
        return f"df = {_fmt_one(df)}"
