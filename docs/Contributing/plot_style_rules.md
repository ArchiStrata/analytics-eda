# Plot Style Rules

Plot Class

- Plot class name should include the {{Pillar}}{{Plot Purpose}}Plot. The plot purpose may include the type of plot used (i.e. histogram, violin).
- Python file name should be snake case of plot class name.
- Plot class docstring must include the big idea, why the plot is important (purpose), and what the plot does at a high-level for a business user.
- Plot class should focus on a single big idea (purpose) for effective data story telling.

Descriptive Statistics (default_descriptive, compute_descriptive, compute_descriptive_frame)

- override default_descriptive to include default descriptive stats only if need to return descriptive stats when the series or dataframe is empty.
- use compute_descriptive for plots that work with Series or compute_descriptive_frame for plots that work with Dataframe.
- if there are any descriptive stat params in the context they are included in descriptive_stats.

Interential Statistics (default_inferential, compute_inferential, compute_inferential_frame)

- override default_inferential to include default inferential stats only if need to return inferential stats when the series or dataframe is empty.
- use compute_inferential for plots that work with Series or compute_inferential_frame for plots that work with Dataframe.
- if there are any inferential stat params in the context they are included in inferential_stats.
- inferential_stats are organized by their name.
- interential hypothesis tests should include statistic, p-value, alpha, and reject.

Chart Metadata (Plot Context & title_kwargs)

- title_template should be clear concise professional and tell an effective data story
- xlabel and ylabel should be clear concise professional and tell an effective data story
- title_kwargs should only be overridden if additional information is required in the title to tell an effective data story.

Visual Draw (draw, draw_frame)

- Use draw when the plot uses Series or draw_frame when the plot uses Dataframe.
