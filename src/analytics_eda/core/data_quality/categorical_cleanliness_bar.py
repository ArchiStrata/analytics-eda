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

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List, Iterable
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..utils.base_plot import BasePlot, PlotContext
from ..utils.named_series_mixin import NamedSeriesMixin


@dataclass
class CategoricalCleanlinessBarContext(PlotContext):
    """
    Categorical cleanliness summary: counts (with percentages) of rows exhibiting:
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
    title_template: str = "Categorical Cleanliness for {name}"
    xlabel: str = "Count"
    ylabel: str = "Issue Type"
    figsize: Tuple[int, int] = (10, 6)

    # Plot/logic knobs
    show_percent_labels: bool = True            # annotate bars with % of non-null
    sort_by_count_ascending: bool = False       # sort bars by count
    max_bars: Optional[int] = None              # not usually needed (we have 4 bars), but kept for parity

    # Cleanliness rules
    # Regex of allowed characters. Default allows letters, digits, whitespace, common punctuation: -_/.,&()'
    allowed_char_pattern: str = r"^[\w\s\-\_/.,&()']*$"

    # If provided, values outside this set are "invalid".
    # Matching is done on stripped tokens; case sensitivity is configurable.
    allowed_categories: Optional[Iterable[str]] = None
    case_sensitive_allowed: bool = False

    # Treat empty-string after strip as invalid (common in data-entry exports)
    treat_empty_as_invalid: bool = True


class CategoricalCleanlinessBarPlot(NamedSeriesMixin, BasePlot):
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

    How
    ----
    - Coerces to NA-aware strings and strips whitespace for checks.
    - Mixed casing is detected by grouping stripped labels case-insensitively
      and flagging any label whose casing differs from the canonical (most
      frequent) case for that token.
    - Non-standard characters are flagged using `allowed_char_pattern` (regex).
    - Invalid categories are those not present in `allowed_categories`
      (with optional case-sensitivity and empty-as-invalid handling).
    - Bars display counts; annotations include percentages of the non-null base.
    - If no issues are found, plotting is skipped and `skip_plot` is set.

    Returns (BasePlot.run schema)
    ------------------------------
      {
        "descriptive_stats": {
          "total": int,                 # total observations (incl. NA)
          "total_nonnull": int,         # non-null observations used as the % base
          "issue_labels": List[str],    # issue names in plotted order
          "issue_counts": List[int],    # counts per issue (over non-null)
          "issue_pcts": List[float],    # percentages per issue (of total_nonnull)
          "n_whitespace": int,
          "n_mixed_case": int,
          "n_nonstandard_chars": int,
          "n_invalid_category": int
        },
        "inferential_stats": {},
        "chart_metadata": {"title","xlabel","ylabel","data_source","file_name"}
      }

    Notes
    -----
    - Percentages use the non-null base (`total_nonnull`).
    - Ordering can be configured via `sort_by_count_ascending`; you may also cap
      bars with `max_bars` (mostly moot here since there are four issues).
    """

    # ---- chart metadata ----
    def build_chart_metadata(self, series: pd.Series) -> Dict[str, Any]:
        label = self.ctx.name or (series.name if series.name else "Value")
        title = self.ctx.title_template.format(name=label)
        return {
            "title": title,
            "xlabel": self.ctx.xlabel,
            "ylabel": self.ctx.ylabel,
            "data_source": self.ctx.data_source,
            "file_name": self.ctx.file_name,
        }

    # ---- defaults when empty ----
    def default_descriptive(self) -> Dict[str, Any]:
        return {
            "total": 0,
            "total_nonnull": 0,
            "issue_labels": ["Leading/Trailing Whitespace", "Mixed Casing", "Non-Standard Characters", "Invalid Category"],
            "issue_counts": [0, 0, 0, 0],
            "issue_pcts": [0.0, 0.0, 0.0, 0.0],
            "n_whitespace": 0,
            "n_mixed_case": 0,
            "n_nonstandard_chars": 0,
            "n_invalid_category": 0,
        }

    # ---- compute ----
    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
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

            def _is_allowed(val: Optional[str]) -> bool:
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

        counts = [n_ws, n_mc, n_ns, n_iv]
        if self.ctx.sort_by_count_ascending:
            order = np.argsort(counts)
        else:
            order = np.argsort(counts)[::-1]
        labels_ordered = np.array(
            ["Leading/Trailing Whitespace", "Mixed Casing", "Non-Standard Characters", "Invalid Category"],
            dtype=object
        )[order]
        counts_ordered = np.array(counts, dtype=int)[order]
        pcts_ordered = (counts_ordered / max(1, total_nonnull)).astype(float)

        # Optional cap (mostly redundant with 4 bars)
        if self.ctx.max_bars is not None and self.ctx.max_bars > 0:
            labels_ordered = labels_ordered[: int(self.ctx.max_bars)]
            counts_ordered = counts_ordered[: int(self.ctx.max_bars)]
            pcts_ordered = pcts_ordered[: int(self.ctx.max_bars)]

        desc = {
            "total": total,
            "total_nonnull": total_nonnull,
            "issue_labels": labels_ordered.tolist(),
            "issue_counts": counts_ordered.tolist(),
            "issue_pcts": pcts_ordered.tolist(),
            "n_whitespace": n_ws,
            "n_mixed_case": n_mc,
            "n_nonstandard_chars": n_ns,
            "n_invalid_category": n_iv,
        }

        # Skip plotting if nothing to show
        if counts_ordered.sum() == 0:
            desc["skip_plot"] = True
            desc["error"] = "no cleanliness issues detected"

        return desc

    # ---- inferential (none) ----
    def compute_inferential(self, s: pd.Series, desc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # ---- draw ----
    def draw(self, s: pd.Series, desc: Dict[str, Any], inf: Dict[str, Any], chart_metadata: Dict[str, Any]):
        sns.set_palette("colorblind")
        fig, ax = plt.subplots(figsize=self.ctx.figsize)

        labels: List[str] = desc["issue_labels"]
        counts: List[int] = desc["issue_counts"]
        pcts: List[float] = desc["issue_pcts"]

        # Horizontal bar chart (labels are short, but horizontal stays consistent with other data-quality plots)
        sns.barplot(x=counts, y=labels, ax=ax)

        # Labels & title
        ax.set_title(chart_metadata["title"])
        ax.set_xlabel(chart_metadata["xlabel"] or "Count")
        ax.set_ylabel(chart_metadata["ylabel"] or "Issue Type")

        # Annotate with percentages of non-null if requested
        if self.ctx.show_percent_labels:
            for i, (v, pct) in enumerate(zip(counts, pcts)):
                ax.text(v, i, f" {v} ({pct*100:.1f}%)", va="center", ha="left", fontsize="small")

        # Footer summary
        footer = (
            f"N (non-null) = {desc['total_nonnull']:,}; "
            f"Whitespace = {desc['n_whitespace']:,}; "
            f"Mixed Casing = {desc['n_mixed_case']:,}; "
            f"Non-Standard Chars = {desc['n_nonstandard_chars']:,}; "
            f"Invalid = {desc['n_invalid_category']:,}"
        )
        fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize="small", color="gray")

        return fig, ax


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
                    # Optional policies you might add:
                    # "case_policy": "most_frequent" | "lower" | "title"
                    # "invalid_policy": "nan" | "other"
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
