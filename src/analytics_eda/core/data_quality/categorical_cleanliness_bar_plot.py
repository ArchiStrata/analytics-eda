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
"""Bar chart summarizing categorical label cleanliness issues."""

from collections.abc import Iterable
from dataclasses import dataclass, field
import re
from typing import Any

import pandas as pd

from analytics_eda.core.visualization.context.plot_context import AxisFormat
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import named_only_validator

from ..visualization.base_plot import BasePlot


@dataclass
class CategoricalCleanlinessBarContext(SeriesBarChartContext):
    """Categorical cleanliness summary.

    Counts (with percentages) of rows exhibiting:
      - Leading/trailing whitespace
      - Mixed casing (inconsistent capitalization across the same token)
      - Non-standard characters (fails a configurable regex)
      - Invalid categories (not in an allowed list)

    Notes
    -----
    - Percentages are computed over *non-null* values.
    - A single row can contribute to multiple issue categories.
    - "Mixed casing" is determined per lowercased, stripped token by checking
      whether multiple distinct casings appear in the data. Rows whose token
      casing is not the group's canonical (most frequent) casing are flagged.
    """

    title_template: str = "Categorical Cleanliness for {name}{modifiers}"
    xlabel: str = "Percent of non‑null"
    ylabel: str = "Issue Type"
    is_orientation_vertical: bool = False

    x_format: AxisFormat = field(
        default_factory=lambda: AxisFormat(kind="percent", decimals=1, percent_scale_0to1=True)
    )
    y_format: AxisFormat = field(
        default_factory=lambda: AxisFormat(kind="category")
    )

    show_subtitle: bool = True

    # Cleanliness rules
    # Regex of allowed characters. Default allows letters, digits, whitespace, common punctuation: -_/.,&()'
    allowed_char_pattern: str = r"^[\w\s\-\_/.,&()']*$"

    # If provided, values outside this set are "invalid".
    # Matching is done on stripped tokens; case sensitivity is configurable.
    allowed_categories: Iterable[str] | None = None
    case_sensitive_allowed: bool = False

    # Treat empty-string after strip as invalid (common in data-entry exports)
    treat_empty_as_invalid: bool = True


class CategoricalCleanlinessBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Visual audit of categorical label cleanliness.

    Why
    ----
    Messy category labels (extra whitespace, inconsistent casing, odd symbols,
    or values outside an allowed list) inflate cardinality, break joins, and
    skew group-bys. A quick, visual tally helps decide normalization rules and
    quantify the cleanup effort.

    What
    ----
    A horizontal bar chart that summarizes four issue types over the *non-null*
    values of the series:
      • Leading/Trailing Whitespace
      • Mixed Casing (label variants that differ only by case)
      • Non-Standard Characters (not matching an allowed regex)
      • Invalid Category (not in an optional allowlist)

    Returns (BasePlot.run schema)
    ------------------------------
      {
        "descriptive_stats": {
          "total": int,                 # total observations (incl. NA)
          "total_nonnull": int,         # non-null observations used as the % base
          "bars": {}
        }
      }

    Notes
    -----
    - Percentages use the non-null base (`total_nonnull`).
    """

    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=named_only_validator(dropna=False, cast_str=False)
        )
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    # ---- compute ----
    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute per-issue counts and percent-of-nonnull bars.

        Builds the bars payload (whitespace, mixed casing, non-standard chars,
        invalid category), caches draw arrays, and returns the descriptive dict.
        """
        total = int(s.size)
        nonnull_mask = ~s.isna()
        total_nonnull = int(nonnull_mask.sum())

        if total_nonnull == 0:
            # Nothing to assess
            desc = self.default_descriptive()
            desc.update({"total": total, "total_nonnull": 0, "skip_plot": True, "error": "no non-null values"})
            return desc

        s_nonnull = s[nonnull_mask].astype("string")  # robust NA-aware string dtype
        # Prepare normalized tokens
        stripped = s_nonnull.str.strip()

        # 1) Leading/trailing whitespace
        whitespace_issue_mask = (s_nonnull != stripped)

        # 2) Mixed casing
        # Group by lowercased, stripped token; find the most frequent original casing per group;
        # mark rows whose casing != canonical casing for that group.
        lowered = stripped.str.lower()
        # Build canonical (most frequent) casing per lowercase token
        # Value counts over original (stripped) within each lowered group
        df = pd.DataFrame({"lower": lowered, "orig": stripped})
        # frequency of each original within each lower
        freq = (
            df.groupby(["lower", "orig"], dropna=False)
              .size()
              .rename("n")
              .reset_index()
        )
        # pick the argmax orig per lower as canonical
        idx = freq.groupby("lower")["n"].idxmax()
        canonical = (
            freq.loc[idx, ["lower", "orig"]]
            .set_index("lower")["orig"]
            .to_dict()
        )
        canonical_series = lowered.map(canonical)
        mixed_case_issue_mask = stripped != canonical_series

        # 3) Non-standard characters (fails regex)
        allowed_pat = re.compile(self.ctx.allowed_char_pattern)
        nonstandard_char_mask = ~stripped.fillna("").map(lambda x: bool(allowed_pat.match(x)))

        # 4) Invalid categories (not in allowed list)
        if self.ctx.allowed_categories is not None:
            allowed = set(self.ctx.allowed_categories)
            if not self.ctx.case_sensitive_allowed:
                allowed = {str(a).strip().lower() for a in allowed}

            def _is_allowed(val: str | None) -> bool:
                if val is None:
                    return False
                txt = str(val).strip()
                if self.ctx.treat_empty_as_invalid and txt == "":
                    return False
                return (txt if self.ctx.case_sensitive_allowed else txt.lower()) in allowed

            invalid_category_mask = ~stripped.map(_is_allowed)
        else:
            # If no allowlist given, we don't flag invalids
            invalid_category_mask = pd.Series(False, index=stripped.index, dtype=bool)

        # Tallies
        n_ws = int(whitespace_issue_mask.sum())
        n_mc = int(mixed_case_issue_mask.sum())
        n_ns = int(nonstandard_char_mask.sum())
        n_iv = int(invalid_category_mask.sum())

        counts_dict = {
            "Leading/Trailing Whitespace": n_ws,
            "Mixed Casing": n_mc,
            "Non-Standard Characters": n_ns,
            "Invalid Category": n_iv,
        }

        # Build bar payload (percent-of-nonnull), then cache for draw
        extra_params = {
            "allowed_char_pattern": self.ctx.allowed_char_pattern,
            "allowed_categories_count": (len(self.ctx.allowed_categories)
                                            if self.ctx.allowed_categories is not None else 0),
            "case_sensitive_allowed": bool(self.ctx.case_sensitive_allowed),
            "treat_empty_as_invalid": bool(self.ctx.treat_empty_as_invalid)
        }
        desc = self.build_series_bar_desc(s, counts_dict, denominator_key="pct_of_nonnull", extra_params=extra_params)

        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Return a short narrative of cleanliness issues.

        Summarizes the overall rate and highlights the most frequent issue(s),
        handling ties where applicable.
        """
        total_nonnull = desc["total_nonnull"]

        findings = {
            # how many nonnull values were included in the plot analysis?
            "context": f"N (non‑null) = {total_nonnull:,}",
            "primary_finding": "",
            "secondary_finding": None
        }

        if not desc or desc.get("total", 0) == 0 or desc.get("total_nonnull", 0) == 0:
            findings["primary_finding"] = "The series is empty."
            return findings

        pct_any_issue = desc["pct_subset"]
        total_issues = desc["subset_count"]

        # Nothing to report → all values are clean
        if total_issues == 0:
            findings["primary_finding"] = "No cleanliness issues detected."
            return findings

        # how many issues are there and how common are they?
        findings["primary_finding"] = f"{self.formatter.format_percent(pct_any_issue)} of values show at least one cleanliness issue ({total_issues:,} rows)."

        # which issues had the most, how common, and how many?
        bars = desc.get("bars", {})
        denom_key = desc.get("denominator_key", "pct_of_nonnull")
        top_labels: list[str] = list(desc.get("top_labels", []))

        # Build a readable tie-aware message
        # sort for determinism
        top_labels = sorted(top_labels)
        parts = []
        for lbl in top_labels:
            v = bars.get(lbl, {})
            pct = float(v.get(denom_key, 0.0))
            cnt = int(v.get("count", 0))
            parts.append(f"{lbl} ({self.formatter.format_percent(pct)}, {cnt} rows)")

        if len(parts) == 1:
            findings["secondary_finding"] = f"Most frequent issue: {parts[0]}."
        else:
            findings["secondary_finding"] = "Most frequent issues (tie): " + ", ".join(parts) + "."

        return findings

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Return a concise subtitle summarizing the overall issue rate."""
        if not desc or desc.get("total", 0) == 0 or desc.get("total_nonnull", 0) == 0:
            return ""

        total_nonnull = desc["total_nonnull"]
        total_issues = desc.get("subset_count", 0)
        pct_any_issue = float(desc.get("pct_subset", 0.0))

        # Case 1: all clean
        if total_issues == 0:
            return "No cleanliness issues detected"

        # Case 2: some issues found
        return f"{self.formatter.format_percent(pct_any_issue)} of {total_nonnull:,} non-null values have cleanliness issues"

    def clean_series(self, s: pd.Series) -> tuple[pd.Series, dict]:
        """
        Return a cleaned copy of the series and a metadata dict describing what changed.

        Cleaning steps (controlled by the context):
        1) Strip leading/trailing whitespace.
        2) Normalize casing per token group (most frequent casing, or a chosen policy).
        3) Remove/normalize non-standard characters (regex-driven).
        4) Handle invalid categories (map to NaN or 'Other').

        Returns
        -------
        cleaned : pd.Series
            Same dtype as input (object/string/category), with cleaned labels.
        meta : dict
            {
            "n_stripped": int,
            "n_case_normalized": int,
            "n_nonstandard_replaced": int,
            "n_invalid_mapped": int,
            "policy": { ...copy of relevant ctx options... }
            }
        """
        # Work on non-null string view
        nonnull_mask = ~s.isna()
        s_out = s.copy()
        if nonnull_mask.sum() == 0:
            return s_out, {
                "n_stripped": 0,
                "n_case_normalized": 0,
                "n_nonstandard_replaced": 0,
                "n_invalid_mapped": 0,
                "policy": {
                    "allowed_char_pattern": self.ctx.allowed_char_pattern,
                    "allowed_categories": list(self.ctx.allowed_categories) if self.ctx.allowed_categories is not None else None,
                    "case_sensitive_allowed": self.ctx.case_sensitive_allowed,
                    "treat_empty_as_invalid": self.ctx.treat_empty_as_invalid,
                }
            }

        # 1) strip
        s_str = s_out[nonnull_mask].astype("string")
        stripped = s_str.str.strip()
        n_stripped = int((s_str != stripped).sum())

        # 2) case normalization (default: canonical = most frequent casing per lowered token)
        lowered = stripped.str.lower()
        df = pd.DataFrame({"lower": lowered, "orig": stripped})
        freq = (df.groupby(["lower", "orig"], dropna=False).size()
                .rename("n").reset_index())
        idx = freq.groupby("lower")["n"].idxmax()
        canonical = (freq.loc[idx, ["lower", "orig"]]
                    .set_index("lower")["orig"]
                    .to_dict())
        canonical_series = lowered.map(canonical)
        n_case_norm = int((stripped != canonical_series).sum())

        # 3) non-standard characters → keep only those matching allowed_char_pattern
        pat = re.compile(self.ctx.allowed_char_pattern)
        # Example policy: drop disallowed chars (you may prefer replace with space, etc.)
        def _sanitize(txt: str) -> str:
            return txt if pat.match(txt) else "".join(ch for ch in txt if pat.match(ch) or pat.match(ch+""))
        sanitized = stripped.map(_sanitize)
        n_nonstd = int((sanitized != stripped).sum())

        # 4) invalid categories
        n_invalid = 0
        if self.ctx.allowed_categories is not None:
            allowed = set(self.ctx.allowed_categories)
            if not self.ctx.case_sensitive_allowed:
                allowed = {str(a).strip().lower() for a in allowed}

            def _is_allowed(val: str) -> bool:
                t = val.strip()
                if self.ctx.treat_empty_as_invalid and t == "":
                    return False
                key = t if self.ctx.case_sensitive_allowed else t.lower()
                return key in allowed

            is_allowed_mask = sanitized.fillna("").map(_is_allowed)
            # Example policy: map invalid to NaN (you could make this configurable)
            n_invalid = int((~is_allowed_mask).sum())
            sanitized = sanitized.where(is_allowed_mask, other=pd.NA)

        # Apply back
        s_out.loc[nonnull_mask] = sanitized

        # If input was categorical, preserve category dtype (rebuild categories)
        if pd.api.types.is_categorical_dtype(s.dtype):
            s_out = s_out.astype("category")

        meta = {
            "n_stripped": n_stripped,
            "n_case_normalized": n_case_norm,
            "n_nonstandard_replaced": n_nonstd,
            "n_invalid_mapped": n_invalid,
            "policy": {
                "allowed_char_pattern": self.ctx.allowed_char_pattern,
                "allowed_categories": list(self.ctx.allowed_categories) if self.ctx.allowed_categories is not None else None,
                "case_sensitive_allowed": self.ctx.case_sensitive_allowed,
                "treat_empty_as_invalid": self.ctx.treat_empty_as_invalid,
                # Extend later with explicit case/invalid handling policies if you add them
            }
        }
        return s_out, meta
