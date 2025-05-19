# Developer Machine Setup

## Prerequisites

1. [Install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

## Local Environment Setup (First Time Only)

```bash
# Clone your fork
git clone https://github.com/your-username/analytics-eda.git
cd analytics-eda

# Create conda environment
conda env create -f environment.yml

# Activate conda environment
# NOTE: Always activate conda environment before writing/running code or tests.
conda activate analytics_eda

# Run tests
pytest
```
