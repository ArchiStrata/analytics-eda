# Copyright 2025 ArchiStrata, LLC and Andrew Dabrowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import uuid
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

logger = logging.getLogger(__name__)


def missing_data_analysis(
        series: pd.Series,
        report_dir: Path,
        report_log_id = str(uuid.uuid4())
    ) -> dict:
    """
    Perform missing data analysis on a pandas Series.

    Args:
        series (pd.Series): Series to analyze.
        report_dir (Path): Directory for saving report files.
        report_log_id (str): report log id.

    Returns:
        dict: {
            'total': int,
            'missing': int,
            'pct_missing': float,
            'missing_count': str (file path to plot)
        }
    """
    logger.info(
        "Starting missing_data_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )

    # Build counts (and fill NaNs)
    status = series.isna().map({False: "Present", True: "Missing"})
    counts = (
        status
        .value_counts()
        .reindex(["Present", "Missing"])
        .fillna(0)
        .astype(int)
    )

    # Summary stats
    total       = int(counts.sum())
    missing     = int(counts["Missing"])
    pct_missing = missing / total if total else 0.0

    summary = {
        'total':       total,
        'missing':     missing,
        'pct_missing': pct_missing
    }

    # Build pct series for plotting
    if total:
        pct = counts / total * 100
    else:
        pct = pd.Series([0.0, 0.0], index=counts.index)

    df = pd.DataFrame({
        "status": counts.index,
        "pct":    pct.values,
        "count":  counts.values
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x="status",
        y="pct",
        hue="status",
        data=df,
        palette=["#4A5568", "#2D9CDB"],
        legend=False,
        ax=ax
    )

    for i, row in df.iterrows():
        ax.text(
            i,
            row["pct"] + 1,
            f"{row['count']:,}\n({row['pct']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(
        f"Missing Data for “{series.name}”: "
        f"{missing:,} of {total:,} values "
        f"({pct_missing * 100:.1f}%)",
        pad=12
    )
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Total", labelpad=8)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    sns.despine(left=True)
    plt.tight_layout()

    filename = (
        f"{series.name.replace(' ', '_')}_missing_data_barplot.png"
        if series.name else
        "series_missing_data_barplot.png"
    )
    missing_data_count_path = report_dir / filename
    fig.savefig(missing_data_count_path, dpi=300)
    plt.close(fig)

    summary['missing_data_barplot'] = str(missing_data_count_path)

    logger.info(
        "Completed missing_data_analysis",
        extra={
            'series_name': series.name,
            'report_log_id': report_log_id
        }
    )
    return summary
