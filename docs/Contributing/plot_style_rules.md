# Plot Style Rules

## Plot Class

- Class name must follow the format: `{{Pillar}}{{PlotPurpose}}Plot`.  
  _Example: `CentralTendencyHistogramPlot`_
- Python file name must be the **snake_case** version of the class name.
- Each class must focus on **one big idea (purpose)** for clear data storytelling.
- Docstring requirements:
  - First sentence states the **big idea** (no header).
  - Include **Why this matters** (purpose).
  - Include **What this plot does** (high-level explanation for a business user).

## Descriptive Statistics

_(default_descriptive, compute_descriptive, compute_descriptive_frame)_

- Override `default_descriptive` only when descriptive stats must be returned for an empty **Series** or **DataFrame**.
- Use `compute_descriptive` for **Series**; use `compute_descriptive_frame` for **DataFrames**.
- Any descriptive stat parameters in context must be included in `descriptive_stats`.
- **Draft Findings (recommended):** Override `draft_descriptive_findings` to return short, human-readable statements derived strictly from `desc`.
  - Keep findings factual, concise, and **non-interpretive**.
  - Limit to ~1–3 bullet-sized statements per plot to respect cognitive load.
  - Use consistent, machine-readable keys so reports can aggregate across plots.
  - Prefer key–value phrasing (e.g., `"central_tendency": "Median > Mean (right-skewed hint)"`).
  - Include units and rounding consistent with the plot.
  - Do not duplicate raw numbers already present in `descriptive_stats`; instead, summarize them.
  - If `desc` is empty or insufficient, return `{}`.
  - Example keys: `summary`, `distribution`, `cardinality`, `coverage`, `outliers`, `data_quality`.

## Inferential Statistics

_(default_inferential, compute_inferential, compute_inferential_frame)_

- Override `default_inferential` only when inferential stats must be returned for an empty **Series** or **DataFrame**.
- Use `compute_inferential` for **Series**; use `compute_inferential_frame` for **DataFrames**.
- Any inferential stat parameters in context must be included in `inferential_stats`.
- `inferential_stats` must be organized by name.
- Hypothesis tests must include: **Statistic, P-Value, Alpha, Reject**.
- **Draft Findings (recommended):** Override `draft_inferential_findings` to return short, human-readable statements that summarize _only what is supported_ by `inf` (optionally using `desc` for context).
  - Keep findings concise, factual, and focused strictly on statistical evidence.
  - Limit to ~1–3 bullet-sized statements per plot to respect cognitive load.
  - Use consistent, machine-readable keys so reports can aggregate across plots.
  - Report decision and direction with threshold (e.g., `"t_test": "Difference is statistically significant at α=0.05 (p=0.012)"`).
  - Include **effect size** and **confidence intervals** when available.
  - Do not duplicate raw numbers already present in `inferential_stats`; instead, summarize them.
  - When assumptions fail (normality, equal variance, independence), include a finding that flags it.
  - Avoid practical/causal claims—focus on statistical evidence only.
  - If `inf` is empty or tests are invalid, return `{}`.
  - Example keys: `hypothesis_tests`, `model_fit`, `assumptions`, `effect_size`.

## Chart Metadata

_(Plot Context & title_kwargs)_

- `title_template` must be clear, concise, and professional, effectively telling the data story.
- `xlabel` and `ylabel` must be clear, concise, and professional, effectively telling the data story.
- Override `title_kwargs` only when additional title information is required for storytelling.
- **Semantic Versioning (recommended):** Each plot must define a semantic version via `plot_semantic_version`.
  - Default version is **`0.1.0`**, which indicates the plot is in an **experimental or draft state**.
  - Update the version to **`1.0.0`** once the plot is **production-ready** (stable API, validated output schema, consistent visual design).
  - Increment **minor version** (e.g., `1.1.0`) for **backward-compatible enhancements** (new descriptive stats, styling options, additional draft findings).
  - Increment **patch version** (e.g., `1.0.1`) for **bug fixes or minor corrections**.
  - Increment **major version** (e.g., `2.0.0`) when introducing **breaking changes** to plot behavior, schema, or interpretation.

## Visualization Design

### Chart Types

- **Diagnostic Charts** (exploratory; raw data; not always used in final reporting)

  - **Histogram**: continuous numeric data (interval, ratio) to show distributions.
  - **Box Plot**: continuous numeric data to show median, quartiles, and outliers. Can also compare groups (categorical discrete vs. numeric continuous).
  - **Scatter Plot**: continuous numeric data (two or more variables) to show relationships.

- **Presentation Charts** (summary data; means or aggregates)
  - **Bar Chart**: discrete categorical data (nominal, ordinal) or discrete numeric counts.
  - **Line Chart**: continuous numeric data across ordered intervals (time series, measurements).
  - **Scatter Plot (means)**: aggregated numeric values across categories.
  - **Slopegraph**: ordered categorical or continuous data comparing two points in time or conditions.
  - **Small Multiples**: repeated bar, line, or scatter plots across categories for comparison.

**Discrete vs. Continuous Guidance**

- **Categorical (always discrete)**: use bar charts, box plots (for numeric outcome by category), or grouped small multiples.
- **Numeric Discrete (counts)**: use bar charts for frequency, or histograms if counts approximate distributions.
- **Numeric Continuous (measurements)**: use histograms, line charts, scatter plots, or box plots depending on purpose.

### Design Principles

**Philosophy & Integrity**

- **Graphical Excellence**: present data with substance, statistics, and design. Deliver the greatest number of ideas in the shortest time with the least ink.
- **Honesty & Transparency**: always represent data truthfully, using proportional scales and avoiding misleading aspect ratios, selective presentation, or exaggeration. Clearly show how data was processed and displayed.
  - **Appropriate Measurement Scales**: ensure visualizations respect the correct level of measurement (nominal, ordinal, interval, ratio). Misapplication of scales distorts interpretation.
  - **Categorical Data is Discrete**: categories (nominal or ordinal) are always discrete, though not all discrete data is categorical.
  - **Discrete vs. Continuous Handling**: plots must correctly account for whether data is discrete (counts, categories) or continuous (measurements) and use appropriate visual forms.
- **Data-Ink Ratio**: maximize ink devoted to actual data; remove redundant or decorative elements; avoid chartjunk (e.g., 3D effects, excessive gridlines, embellishments); push non-message-critical elements to the background.

**Communication**

- **Single Big Idea**: each visual must communicate one clear, effective data story.
- **Clarity**: make complex data easy to understand without distortion.
- **Comparisons**: enable clear recognition of trends, patterns, and differences.
- **Narrative**: guide the viewer to insights and conclusions.
- **Simplicity vs. Complexity**: balance readability with completeness.

**Execution**

- **Axis Integrity & Labeling**: do not omit labels or units on axes. Labels must be clear, concise, and non-obstructive. Prefer direct labeling on data points over distant legends. Align text left; avoid diagonal labels for readability.
- **Accessibility**: use colorblind-friendly palettes and legible typefaces.
- **Layout & Aesthetics**: use whitespace strategically; less is more. Maintain a clean, professional, and appealing style.

### Human Perception & Cognition

**Gestalt Principles**

- **Proximity**: group elements that are close together.
- **Similarity**: use consistent color, shape, or size to show relation.
- **Enclosure**: group elements within a boundary.
- **Closure**: allow viewers to perceive whole forms even with gaps.
- **Continuity**: guide the eye along smooth, natural paths.
- **Connection**: linked objects are perceived as related.

**Preattentive Attributes** (direct attention and create visual hierarchy)

- **Size**
- **Color** (use sparingly; base color grey; apply consistently; ensure accessibility)
- **Position** (most important elements at top/left)
- **Highlighting** (bold preferred over italics; avoid underlining)
- **Case** (capitalize short key words for scanning)
- **Typeface** (minimize font variation)
- **Inversion** (reverse text/background sparingly)

**Cognitive & Memory Principles**

- **Iconic Memory**: preattentive features are processed instantly.
- **Short-Term Memory**: limit visuals to ~4 chunks of information.
- **Long-Term Memory**: reinforce learning by combining visual + verbal cues.

### Draw Functions

- Use `draw` for **Series**; use `draw_frame` for **DataFrame**.
