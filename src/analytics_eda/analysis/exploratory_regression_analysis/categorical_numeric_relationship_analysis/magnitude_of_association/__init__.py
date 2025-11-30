"""Magnitude-of-association plots for categorical↔numeric relationships."""

from .magnitude_central_tendency_anova_kruskal_plot import (
    MagnitudeCentralTendencyAnovaKruskalContext,
    MagnitudeCentralTendencyAnovaKruskalPlot,
)
from .magnitude_distribution_overlap_density_plot import (
    MagnitudeDistributionOverlapDensityContext,
    MagnitudeDistributionOverlapDensityPlot,
)
from .magnitude_effect_size_bar_plot import (
    MagnitudeEffectSizeBarContext,
    MagnitudeEffectSizeBarPlot,
)

__all__ = [
    "MagnitudeCentralTendencyAnovaKruskalContext",
    "MagnitudeCentralTendencyAnovaKruskalPlot",
    "MagnitudeDistributionOverlapDensityContext",
    "MagnitudeDistributionOverlapDensityPlot",
    "MagnitudeEffectSizeBarContext",
    "MagnitudeEffectSizeBarPlot",
]
