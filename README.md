# analytics-eda

Exploratory Data Analysis (EDA) is critical but time-consuming and often inconsistent across projects.

`analytics-eda` is a lightweight Python library that streamlines EDA with fast, automated analysis to uncover data issues, validate assumptions, and support modeling decisions.

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Features](#features)
- [API Reference](#api-reference)
- [Contributing Guidelines](#contributing-guidelines)
- [License](#license)

## Installation

**Requires**: Python 3.11 or later

```bash
pip install analytics-eda
```

## Quickstart

Here's how to generate a univariate numeric EDA report from a pandas series:

```python
import json
import numpy as np
import pandas as pd

from analytics_eda.univariate.numeric.univariate_numeric_analysis import univariate_numeric_analysis

# Series to analyze
rng = np.random.default_rng(0)
data = rng.normal(loc=0.0, scale=1.0, size=15)
normal_15_series = pd.Series(data, name="normal_series")

# Analyze numeric series
report_file_path = univariate_numeric_analysis(
    normal_15_series
)

# Access json report
result = json.loads(report_file_path.read_text())
print(json.dumps(result, indent=2))

```

## Features

### Univariate Exploratory Data Analysis

Univariate EDA analyzes a single variable at a time to understand its distribution, detect anomalies, and assess data quality.  
This step is essential for uncovering data issues before modeling and making informed preprocessing decisions.

Supports:

- **Numeric data**:

  - Identifies missing data
  - Computes descriptive statistics (mean, median, std, skew, etc.)
  - Evaluates normality with tests and visualizations
  - Detects outliers using IQR, Z-score, and robust Z-score
  - Runs inferential tests (e.g. goodness-of-fit, variance checks)

- **Categorical data**:
  - Summarizes frequency distributions
  - Flags rare or dominant categories
  - Assesses cardinality and missing values

---

### Bivariate Exploratory Data Analysis

Bivariate EDA compares two variables to assess relationships and patterns across groups.  
It helps detect important group-level differences and guides feature selection or encoding strategies.

Supports:

- **Numeric vs. Categorical**:
  - Performs statistical tests (e.g. homogeneity of variance, distribution overlap)
  - Runs global hypothesis tests to compare group distributions
  - Estimates effect size between segments
  - Analyzes numeric patterns within categorical groups

## API Reference

See our [API Reference.md](docs/API%20Reference/API%20Reference.md) for details.

## Contributing Guidelines

We welcome contributions! Please see [CONTRIBUTING.md](docs/Contributing/CONTRIBUTING.md) for details.

## License

This project is open source under the Apache License 2.0. See [LICENSE](LICENSE) for details.
