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

## Inferential Statistics

_(default_inferential, compute_inferential, compute_inferential_frame)_

- Override `default_inferential` only when inferential stats must be returned for an empty **Series** or **DataFrame**.
- Use `compute_inferential` for **Series**; use `compute_inferential_frame` for **DataFrames**.
- Any inferential stat parameters in context must be included in `inferential_stats`.
- `inferential_stats` must be organized by name.
- Hypothesis tests must include: **Statistic, P-Value, Alpha, Reject**.

## Chart Metadata

_(Plot Context & title_kwargs)_

- `title_template` must be clear, concise, and professional, effectively telling the data story.
- `xlabel` and `ylabel` must be clear, concise, and professional, effectively telling the data story.
- Override `title_kwargs` only when additional title information is required for storytelling.

## Visual Draw

_(draw, draw_frame)_

- Use `draw` for **Series**; use `draw_frame` for **DataFrames**.
- Ensure the visual communicates a **single big idea** that tells an effective data story.
- Do not omit labels or units on axes.
- Use `colorblind` friendly palette.
