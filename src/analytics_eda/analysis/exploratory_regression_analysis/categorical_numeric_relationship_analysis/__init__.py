"""Categorical↔Numeric relationship analysis.

Plots and report builders for group sizes, variance checks, effect sizes,
distribution overlap, and Tukey HSD post-hoc comparisons.
"""

from .categorical_numeric_relationship_analysis import (
    CategoricalNumericRelationshipAnalysis,
    CategoricalNumericRelationshipAnalysisContext,
)
from .direction_of_association.direction_of_association_analysis import (
    DirectionOfAssociationAnalysis,
    DirectionOfAssociationAnalysisContext,
)
from .direction_of_association.direction_posthoc_tukey_hsd_plot import (
    DirectionPosthocTukeyHsdContext,
    DirectionPosthocTukeyHsdPlot,
)
from .magnitude_of_association.magnitude_central_tendency_anova_kruskal_plot import (
    MagnitudeCentralTendencyAnovaKruskalContext,
    MagnitudeCentralTendencyAnovaKruskalPlot,
)
from .magnitude_of_association.magnitude_distribution_overlap_density_plot import (
    MagnitudeDistributionOverlapDensityContext,
    MagnitudeDistributionOverlapDensityPlot,
)
from .magnitude_of_association.magnitude_effect_size_bar_plot import (
    MagnitudeEffectSizeBarContext,
    MagnitudeEffectSizeBarPlot,
)
from .magnitude_of_association.magnitude_of_association_analysis import (
    MagnitudeOfAssociationAnalysis,
    MagnitudeOfAssociationAnalysisContext,
)
from .relationship_structure.relationship_structure_analysis import (
    RelationshipStructureAnalysis,
    RelationshipStructureAnalysisContext,
)
from .relationship_structure.relationship_structure_group_size_bar_plot import (
    RelationshipStructureGroupSizeBarContext,
    RelationshipStructureGroupSizeBarPlot,
)
from .relationship_structure.relationship_structure_variance_homogeneity_box_plot import (
    RelationshipStructureVarianceHomogeneityBoxPlot,
    RelationshipStructureVarianceHomogeneityContext,
)
