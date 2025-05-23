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
from pathlib import Path
import logging
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)

def visualize_time_series_structure(df: pd.DataFrame,
                                  numeric_col: str,
                                  time_col: str,
                                  report_dir: Path,
                                  rolling_window: int = 12,
                                  report_log_id = str(uuid.uuid4())) -> dict:
    """
    Visualize Time Series Structure
    ---------------------------------------
    Generates and saves key structure plots for a univariate time series, and
    returns a mapping of plot names to file paths.

    Parameters:
    - df (pd.DataFrame): DataFrame with datetime index or a datetime column.
    - numeric_col (str): Name of the numeric series column.
    - time_col (str): Name of the datetime column (if not already index).
    - report_dir (Path): Directory under which plots are saved.
    - rolling_window (int): Window size for rolling statistics.

    Returns:
    - visuals (dict): {visual_name: file_path, ...}
    """
    logger.info(
        "Starting visualize_time_series_structure",
        extra={
            'time_col': time_col,
            'numeric_col': numeric_col,
            'report_log_id': report_log_id
        }
    )

    visuals = {}
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare time series
    ts = df.copy()
    if time_col in ts.columns:
        if not pd.api.types.is_datetime64_any_dtype(ts[time_col]):
            raise TypeError(f"'{time_col}' must be datetime64 dtype before structure analysis.")
        ts = ts.set_index(time_col)
    ts = ts.sort_index()
    series = ts[numeric_col].dropna()

    # 2. Plot raw time series
    raw_path = report_dir / 'raw_line_plot.png'
    max_time, max_val = series.idxmax(), series.max()
    min_time, min_val = series.idxmin(), series.min()

    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, linewidth=2, label=numeric_col)
    plt.scatter([max_time, min_time], [max_val, min_val],
                color='firebrick', zorder=5)
    plt.annotate(f'Peak: {max_val:.1f}',
                 xy=(max_time, max_val),
                 xytext=(max_time, max_val * 1.05),
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=9)
    plt.annotate(f'Tough: {min_val:.1f}',
                 xy=(min_time, min_val),
                 xytext=(min_time, min_val * 0.95),
                 arrowprops=dict(arrowstyle='->'),
                 fontsize=9)
    
    plt.title(f'{numeric_col} Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(numeric_col, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.savefig(raw_path, dpi=300)
    plt.close()
    visuals['raw_line_plot'] = str(raw_path)

    # 3. Plot rolling statistics
    roll_path = report_dir / 'rolling_statistics.png'

    rolling_mean   = series.rolling(rolling_window, min_periods=1).mean()
    rolling_med    = series.rolling(rolling_window, min_periods=1).median()
    rolling_std    = series.rolling(rolling_window, min_periods=1).std()
    rolling_q25    = series.rolling(rolling_window, min_periods=1).quantile(0.25)
    rolling_q75    = series.rolling(rolling_window, min_periods=1).quantile(0.75)

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_mean,    label=f'Rolling Mean ({rolling_window})',    linewidth=2)
    plt.plot(rolling_med,     label=f'Rolling Median ({rolling_window})',  linestyle='--')

    # Std‐bands (mean ± 2σ)
    upper_band = rolling_mean + 2*rolling_std
    lower_band = rolling_mean - 2*rolling_std
    plt.fill_between(series.index, lower_band, upper_band,
                     color='grey', alpha=0.2, label='±2 Std Dev')

    # IQR shading
    plt.fill_between(series.index, rolling_q25, rolling_q75,
                     color='blue', alpha=0.1, label='25–75th Percentile')

    # Styling
    plt.title(f'{numeric_col} with Rolling Statistics', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(numeric_col, fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    plt.savefig(roll_path, dpi=300)
    plt.close()
    visuals['rolling_statistics'] = str(roll_path)

    # 4. ACF
    acf_path = report_dir / 'acf.png'
    fig_acf = plot_acf(series, lags=min(len(series)//2, 40))
    fig_acf.suptitle('Autocorrelation (ACF)')
    fig_acf.tight_layout()
    fig_acf.savefig(acf_path, dpi=300)
    plt.close(fig_acf)
    visuals['acf'] = str(acf_path)

    # 5. PACF
    pacf_path = report_dir / 'pacf.png'
    fig_pacf = plot_pacf(series, lags=min(len(series)//2, 40))
    fig_pacf.suptitle('Partial Autocorrelation (PACF)')
    fig_pacf.tight_layout()
    fig_pacf.savefig(pacf_path, dpi=300)
    plt.close(fig_pacf)
    visuals['pacf'] = str(pacf_path)

    # 6. STL decomposition
    stl_path = report_dir / 'stl_decomposition.png'

    # Infer seasonal period (e.g., 12 for monthly, 7 for daily if weekly seasonality)
    freq = ts.index.inferred_freq
    if freq is None:
        freq = pd.infer_freq(ts.index)

    period = None
    if freq is not None:
        if 'ME' in str(freq):
            period = 12
        elif 'D' in str(freq):
            period = 7
    if period:
        stl = STL(series, period=period)
        result = stl.fit()
        fig = result.plot()
        fig.suptitle('STL Decomposition')
        fig.tight_layout()
        fig.savefig(stl_path, dpi=300)
        plt.close(fig)
        visuals['stl_decomposition'] = str(stl_path)
    else:
        logger.warning(
            "visualize_time_series_structure: Skipped STL decomposition (could not infer period)",
            extra={
                'time_col': time_col,
                'numeric_col': numeric_col,
                'report_log_id': report_log_id
            }
        )

    logger.info(
        "Completed visualize_time_series_structure",
        extra={
            'time_col': time_col,
            'numeric_col': numeric_col,
            'report_log_id': report_log_id
        }
    )

    return visuals
