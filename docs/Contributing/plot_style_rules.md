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
- Include units and consistent rounding with the plot configuration.

### Draft Descriptive Findings (recommended)

Implement `draft_descriptive_findings(desc) -> Dict[str, Any]` to emit short, human-readable statements derived strictly from `desc`.  
Findings are **plot-scoped** — each plot tells one piece of the story.

#### Standard schema (required keys)

```json
{
  "context": "str", // what slice this plot is based on (sample size, group, timeframe)
  "primary_finding": "str", // one big idea this plot is designed to communicate
  "secondary_finding": "str | None" // optional nuance; omit or set None if not applicable
}
```

#### Authoring rules

- Keep findings factual, concise, non-interpretive (no recommendations or causes)
- Limit to ~1 bullet for `primary_finding` and 0–1 for `secondary_finding` to respect cognitive load.
- Use phrasing consistent with the plot’s theme (e.g., central tendency, dispersion, shape, extremes, comparison).
- Do not repeat raw numbers already in `descriptive_stats`; summarize them.
- If `desc` is empty `{}`, return `{}`.
- If there is `context` to report (e.g., total rows, total non-nulls, slice metadata) but the series is insufficient for meaningful findings:
  - Always populate `context` with a short explanation of what data was available
  - Set `primary_finding` and `secondary_finding` to None.
  - Optionally include a neutral `primary_finding` message such as `"The series is empty."` when appropriate.

#### Secondary finding – when to include

Good candidates:

- Contrast or Caveat – subtle exception to the main story.
- Supporting Detail – a secondary stat that reinforces the primary.
- Complementary Angle – another perspective that deepens interpretation without shifting focus.
- Outliers or Anomalies – notable but non-defining extremes.

Do not include if it:

- Repeats the primary in different words.
- Distracts from the main message.
- Is pure detail without interpretive value (belongs in `descriptive_stats`).

## Inferential Statistics

_(default_inferential, compute_inferential, compute_inferential_frame)_

- Override `default_inferential` only when inferential stats must be returned for an empty **Series** or **DataFrame**.
- Use `compute_inferential` for **Series**; use `compute_inferential_frame` for **DataFrames**.
- Any inferential stat parameters in context must be included in `inferential_stats`.
- `inferential_stats` must be organized by name.
- Hypothesis tests must include: **Statistic, P-Value, Alpha, Reject**.

### Draft Inferential Findings (recommended)

Implement `draft_inferential_findings(inf) -> Dict[str, Any]` to emit short, human-readable statements derived strictly from `inf` (optionally using `desc` for added context).  
Findings are **plot-scoped** — each plot communicates only the inference it was designed to test.

#### Standard schema (required keys)

```json
{
  "context": "str", // what test/sample this inference is based on (sample size, groups, model type)
  "primary_finding": "str", // one key result supported by statistical evidence
  "secondary_finding": "str | None" // optional nuance; omit or set None if not applicable
}
```

#### Authoring rules

- Keep findings factual, concise, evidence-based.
- Limit to ~1 bullet for `primary_finding` and 0–1 for `secondary_finding` to respect cognitive load.
- Report results in plain language, but tie them directly to the evidence (statistic, p-value, confidence interval, effect size).
- Use phrasing consistent with the test or model (e.g., significance test, correlation, regression fit).
- Do not duplicate raw numbers already in `inferential_stats`; summarize them.
- If `inf` is empty, invalid, or inconclusive, return `{}`.
- Always flag when assumptions fail (e.g., non-normality, unequal variance).
- Avoid practical or causal claims — stay strictly within what the statistical evidence supports.

#### Secondary finding – when to include

Good candidates:

- Effect Size – magnitude of difference/association beyond significance.
- Confidence Interval – adds nuance about estimate precision.
- Assumptions Check – e.g., “Equal variance assumption not met.”
- Model Fit Detail – e.g., “R² = 0.42 explains moderate variance.”

Do not include if it:

- Repeats the primary in different words.
- Distracts from the main inference.
- Provides raw stats without interpretive value (those belong in `inferential_stats`).

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

## Unit Testing

All plots must have a data driven pytest called test\_{{plot snake case}}\_data_driven.

### What changes between plots

- The parametrized cases (input factories + kwargs + expected),
- The test function name (e.g., test_balance_rare_categories_plot_data_driven),
- The Plot and Context classes under test.

#### Standard Test Shape

```python
@pytest.mark.parametrize(
    "make_series_or_df, kwargs, expect",
    [
        # cases...
    ],
    ids=[ ... ],
)
def test_<plot_snake_case>_data_driven(make_series_or_df, kwargs, expect, tmp_path, assert_plot_metadata):
    data = make_series_or_df()

    # Route artifacts to tmp_path if saving
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = <PlotContext>(**kwargs)
    plot = <PlotClass>(ctx)

    payload = plot.run(data)

    # Single assertion point that validates metadata, stats, findings, and artifacts
    assert_plot_metadata(payload, expect, tmp_path)
```

### Expected Payload Contract (from plot.run)

payload must be a dict with these keys (omit blocks not implemented by the plot):

- chart_metadata
  Required fields: title, xlabel, ylabel, data_source, version
  Conditional: file_name (present only when save_path in context)

- descriptive_stats
  Dict of descriptive outputs computed by the plot

- draft_descriptive_findings
  Dict with keys: context (str), primary_finding (str), secondary_finding (str|None)
  (omit or return {} for empty/insufficient data)

- inferential_stats (if implemented)
  Organized by test/model name; each hypothesis test includes statistic, p_value, alpha, reject; effect sizes / CIs when available

- draft_inferential_findings (if implemented)
  Same keys as descriptive findings; evidence-focused

### Required Scenarios

- Empty Input (Series/DataFrame):
  Asserts default/empty descriptive_stats, {} findings, no file saved.

- Happy-Path / Plot-Specific Cases:
  Save artifact when requested; assert metadata, stats, and findings according to expect.

Edge Conditions (plot-specific):

- Boundary thresholds (e.g., α, rarity cutoffs)
- Minimal categories / small N
- Missing values / zero-count categories
- Capping/aggregation (“Other”), sorting flags, percent vs. count formatting

### Stability & Precision

- Determinism: Set seeds (np.random.seed) and use fixed inputs.
- Numeric Tolerances: Prefer callables or pytest.approx for derived metrics.
- Text Resilience: Use \*\_contains substring checks for findings to avoid brittle tests.
- Artifacts Isolation: Always redirect save_path to tmp_path.
