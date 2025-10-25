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
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import seaborn as sns

from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import numeric_validator
from ..visualization.base_plot import BasePlot, PlotContext
from .binning_rules import choose_bins

@dataclass
class CentralTendencyHistogramContext(PlotContext):
    title_template: str = "Distribution of {name}{modifiers}: Central Tendency"
    xlabel: str = "Value"
    ylabel: str = "Count"
    show_subtitle: bool = True
    enable_legend: bool = True

    # plot-specific knob
    bins: Optional[int] = None  # if None, use choose_bins
    min_peak_strength: float = 0.05            # suppress modes if weaker than this fraction
    max_mode_lines: int = 3                    # hard cap on how many mode lines to draw

class CentralTendencyHistogramPlot(BasePlot):
    """
    A histogram designed to highlight the central tendency of a numeric variable.

    Why this matters:
        Understanding where data values cluster is essential for summarizing distributions,
        comparing groups, and identifying skewness. Highlighting mean, median, and mode(s)
        helps analysts and decision-makers quickly see how typical values align or diverge.

    What this plot does:
        - Accepts a pandas Series of numeric values.
        - Plots a histogram with annotated vertical lines for mean, median, and mode(s).
        - Adapts mode detection for unimodal, bimodal, or multimodal distributions.
        - Optionally saves the figure to disk.
        - Returns descriptive statistics (sample size, mean, median, modes, modality, peak strength)
          and chart metadata for reporting and reproducibility.
    """
    def __init__(self, ctx):
        parts = PlotParts(
            series_validator=numeric_validator()
        )
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """
        Return the semantic version of this plot implementation.
        """
        return "1.0.0"

    def default_descriptive(self) -> Dict[str, Any]:
        bins = self.ctx.bins if self.ctx.bins is not None else 0
        return {
            "params": {
                "bins": int(bins),
                "bins_arg": None,
                "bin_rule": None,
                "mode_method": None,
                "min_peak_strength": float(getattr(self.ctx, "min_peak_strength", 0.05)),
                "max_mode_lines": int(getattr(self.ctx, "max_mode_lines", 3)),
            },
            "n": 0,
            "mean": None,
            "median": None,
            "modes": [],
        }

    def compute_descriptive(self, s: pd.Series) -> Dict[str, Any]:
        x = s.dropna()
        n = int(x.size)

        # Determine bins (supports int-like or array-like edges)
        if self.ctx.bins is not None:
            b = self.ctx.bins

            # int-like (Python int or numpy integer)
            if isinstance(b, (int, np.integer)):
                chosen_bins = max(int(b), 1)
                bin_rule = "explicit:int"

            # array-like edges (list/tuple/ndarray) -> must be monotonic increasing with >= 2 edges
            elif isinstance(b, (list, tuple, np.ndarray)):
                edges = np.asarray(b, dtype=float)

                # Valid edges require at least 2 unique, strictly increasing values
                unique_edges = np.unique(edges)
                if unique_edges.size < 2:
                    # Not enough edges, fall back to automatic rule
                    chosen_bins, bin_rule = choose_bins(x)
                else:
                    # Enforce strict monotonic increase
                    if np.any(np.diff(unique_edges) <= 0):
                        # Non-increasing edges; fall back to automatic rule
                        chosen_bins, bin_rule = choose_bins(x)
                    else:
                        chosen_bins = unique_edges  # histogram will accept these as bin edges
                        bin_rule = "explicit:edges"

            else:
                # Unrecognized type; fall back to automatic selection
                chosen_bins, bin_rule = choose_bins(x)

        else:
            chosen_bins, bin_rule = choose_bins(x)

        bins_arg = (chosen_bins if (isinstance(chosen_bins, (list, tuple, np.ndarray)) and len(chosen_bins) > 0)
                    else (max(int(chosen_bins), 1) if isinstance(chosen_bins, int) else 1))

        mean = float(x.mean()) if n else None
        mean_round_decimals = self.formatter.mean_decimals_from_series(x)
        mean_formatted = self.formatter.format_numeric_value(mean, decimals=mean_round_decimals)

        median = float(x.median()) if n else None
        median_round_decimals = self.formatter.median_decimals_from_series(x, median)
        median_formatted = self.formatter.format_numeric_value(median, decimals=median_round_decimals)

        # Defaults
        mode_method: Optional[str] = None
        candidate_modes: List[float] = []
        modality: str = "none"          # "unimodal" | "bimodal" | "multimodal" | "none"
        peak_strength: float = None  # fraction of observations in the strongest peak

        if n == 0:
            pass  # keep defaults
        elif n < 3:
            # --- Very small n guard ---
            vc = x.value_counts()
            peak_strength = float(vc.max()) / n
            raw_modes = vc.index[vc.eq(vc.max())].tolist()
            if len(raw_modes) == 1:
                candidate_modes = [float(raw_modes[0])]
                mode_method = "series.mode"
                modality = "unimodal"
            else:
                # two distinct values in n=2 -> no clear mode
                mode_method = "insufficient_n:no_clear_mode"
        else:
            # Try exact mode first
            raw_modes = x.mode().tolist()
            vc = x.value_counts()
            peak_strength = float(vc.max()) / n if n else None

            if len(raw_modes) == 1:
                candidate_modes = [float(raw_modes[0])]
                mode_method = "series.mode"
                modality = "unimodal"
            else:
                # --- Histogram-based approximation for continuous/multimodal cases ---
                counts, edges = np.histogram(x.to_numpy(), bins=bins_arg, density=False)
                top = int(np.argmax(counts))
                max_count = counts[top]
                top_bins = np.where(counts == max_count)[0]

                # Strength based on histogram peak (overrides value-count strength)
                peak_strength = float(max_count) / n if n else None

                candidate_modes = [float(0.5 * (edges[i] + edges[i + 1])) for i in top_bins]
                mode_method = "histogram_bin_centers"
                modality = ("unimodal" if len(candidate_modes) == 1
                            else "bimodal" if len(candidate_modes) == 2
                            else "multimodal")

        params = {
            "bins": self.ctx.bins,
            "bins_arg": bins_arg,
            "mode_method": mode_method,
            "min_peak_strength": float(self.ctx.min_peak_strength),
            "max_mode_lines": int(self.ctx.max_mode_lines),
            "bin_rule": bin_rule,
            "mean_round_decimals": int(mean_round_decimals),
        }
        return {
            "params": params,
            "n": n,
            "mean": mean,
            "mean_formatted": mean_formatted,
            "median": median,
            "median_formatted": median_formatted,
            "modes": candidate_modes,
            "modality": modality,
            "peak_strength": peak_strength,
        }
    
    def draft_descriptive_findings(self, desc: Dict[str, Any]) -> Dict[str, Any]:
        n = desc.get("n", 0)

        mean = desc.get("mean")
        mean_formatted = desc.get("mean_formatted")

        median = desc.get("median")
        median_formatted = desc.get("median_formatted")

        modes = desc.get("modes", [])
        modality = desc.get("modality", "none")

        if n == 0:
            return {
                "context": "No non-null values available.",
                "primary_finding": None,
                "secondary_finding": None,
            }

        context = f"n = {n:,} observations"

        # Primary finding: where the data centers
        if modality == "unimodal" and modes:
            primary = f"The distribution is unimodal with a central peak around {self.formatter.format_numeric_value(modes[0])}."
        elif modality == "bimodal" and len(modes) >= 2:
            primary = f"The distribution is bimodal with peaks near {self.formatter.format_numeric_value(modes[0])} and {self.formatter.format_numeric_value(modes[1])}."
        elif modality == "multimodal":
            primary = "The distribution is multimodal, indicating several distinct peaks."
        else:
            primary = "No clear mode is present in the distribution."

        # Secondary finding: relationship between mean and median
        if not np.isnan(mean) and not np.isnan(median):
            if abs(mean - median) < 1e-6:  # effectively equal
                secondary = f"Mean and median are nearly identical at {mean_formatted}, suggesting a symmetric distribution."
            elif mean > median:
                secondary = f"Mean ({mean_formatted}) is greater than median ({median_formatted}), suggesting right-skew."
            else:
                secondary = f"Mean ({mean_formatted}) is less than median ({median_formatted}), suggesting left-skew."
        else:
            secondary = None

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": secondary,
        }
    
    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        n = desc.get("n", 0)

        mean = desc.get("mean")
        mean_formatted = desc.get("mean_formatted")

        median = desc.get("median")
        median_formatted = desc.get("median_formatted")

        modes = desc.get("modes", [])
        modality = desc.get("modality", "none")

        if n == 0:
            return "No non-null observations available."

        # Base: sample size
        subtitle = f"{n:,} observations. "

        # Add modality
        if modality == "unimodal" and modes:
            subtitle += f"Central peak around {self.formatter.format_numeric_value(modes[0])}. "
        elif modality == "bimodal" and len(modes) >= 2:
            subtitle += f"Bimodal with peaks near {self.formatter.format_numeric_value(modes[0])} and {self.formatter.format_numeric_value(modes[1])}. "
        elif modality == "multimodal":
            subtitle += "Multiple peaks indicate a multimodal distribution. "
        else:
            subtitle += "No clear mode detected. "

        # Add mean/median relationship
        if not np.isnan(mean) and not np.isnan(median):
            if abs(mean - median) < 1e-6:
                subtitle += f"Mean and median align at {mean_formatted}, suggesting symmetry."
            elif mean > median:
                subtitle += f"Mean ({mean_formatted}) exceeds median ({median_formatted}), suggesting right-skew."
            else:
                subtitle += f"Mean ({mean_formatted}) is below median ({median_formatted}), suggesting left-skew."

        return subtitle.strip()

    def draw(
        self,
        s: pd.Series,
        desc: Dict[str, Any],
        inf: Dict[str, Any],
        chart_metadata: Dict[str, Any],
        *,
        fig,
        ax,
        palette,
    ):

        bins_arg = desc["params"]["bins_arg"]

        sns.histplot(s, bins=bins_arg, ax=ax, color=self.neutral_grey(variant="light"))

        if desc["n"] > 0:
            mean = desc.get("mean")
            mean_formatted = desc.get("mean_formatted")

            median = desc.get("median")
            median_formatted = desc.get("median_formatted")

            ax.axvline(mean, label=f"Mean = {mean_formatted}", color=palette[1])
            ax.axvline(median, label=f"Median = {median_formatted}", color=palette[2])

            # Modes (presentation rules here)
            candidate_modes = desc.get("modes", []) or []
            peak_strength = float(desc.get("peak_strength") or None)

            # Suppress if peak too weak
            if (len(candidate_modes) > 0) and not (np.isnan(peak_strength)) and (peak_strength >= self.ctx.min_peak_strength):
                # Cap number of lines
                to_draw = candidate_modes[: int(self.ctx.max_mode_lines)]
                for i, center in enumerate(to_draw, start=1):
                    label = "Mode" if len(to_draw) == 1 else f"Mode {i}"
                    ax.axvline(center, label=f"{label} ≈ {self.formatter.format_numeric_value(center)}", color=palette[2 + i])

        return fig, ax
