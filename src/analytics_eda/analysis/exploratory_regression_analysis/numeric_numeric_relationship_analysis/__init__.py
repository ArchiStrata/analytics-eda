"""Exploratory regression: numeric↔numeric relationship analysis utilities.

This subpackage includes structure (scatter/LOWESS), magnitude (r, R², OLS),
and direction (OLS slope) diagnostics and plotting helpers.
"""

from .direction_association_scatter_ols_trend_plot import (
    DirectionAssociationScatterOLSTrendContext,
    DirectionAssociationScatterOLSTrendPlot,
)
from .magnitude_association_residual_plot import (
    MagnitudeAssociationResidualContext,
    MagnitudeAssociationResidualPlot,
)
from .magnitude_association_scatter_ols_plot import (
    MagnitudeAssociationScatterOLSContext,
    MagnitudeAssociationScatterOLSPlot,
)
from .numeric_numeric_relationship_analysis import numeric_numeric_relationship_analysis
from .relationship_structure_scatter_lowess_plot import (
    RelationshipStructureScatterLowessContext,
    RelationshipStructureScatterLowessPlot,
)
from .relationship_structure_scatter_plot import (
    RelationshipStructureScatterContext,
    RelationshipStructureScatterPlot,
)
