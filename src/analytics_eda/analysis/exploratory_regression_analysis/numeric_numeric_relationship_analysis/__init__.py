# TODO: numeric_numeric_relationship_analysis
from .relationship_structure_scatter_plot import RelationshipStructureScatterContext, RelationshipStructureScatterPlot
from .relationship_structure_scatter_lowess_plot import RelationshipStructureScatterLowessContext, RelationshipStructureScatterLowessPlot
from .magnitude_association_scatter_ols_plot import MagnitudeAssociationScatterOLSContext, MagnitudeAssociationScatterOLSPlot

# TODO: Magnitude of Association: Residual Plot
# Role: Diagnoses model adequacy & assumptions, not association strength itself.
# * Descriptive Stats
# ** Mean of residuals (should ≈ 0)
# ** Residual standard deviation (σ̂, unexplained variance)
# ** Range or spread of residuals
# ** Potential patterns (heteroscedasticity, curvature, clustering)
# * Inferential Stats
# ** Normality test of residuals (e.g., Shapiro–Wilk, optional for EDA)
# ** Homoscedasticity test (e.g., Breusch–Pagan, White’s test, optional)
# ** Outlier/influence diagnostics (Cook’s distance, leverage, optional if going deeper)

# TODO: Direction of Association: Scatter Plot + OLS Regression Line (Trend Line)
# * Descriptive Stats:
# ** Slope coefficient (β₁) from OLS (captures the direction and steepness)
# ** Intercept (β₀, for context)
# ** Sign of slope (positive, negative, or ~0)
# * Inferential Stats:
# ** Significance test for slope (t-test, p-value for H₀: β₁ = 0)
# ** Confidence interval for slope (β₁) (range of plausible directions)
