# Configuration Guide

This document explains all configuration parameters, their effects, and recommendations for different scenarios.

## Overview

Configuration is managed through the `Config` dataclass in `tracking_analysis/core/config.py`. All parameters have sensible defaults, but tuning can improve analysis quality for specific datasets.

## Data Configuration

### `data_path`
- **Type**: `str`
- **Default**: `'./data'`
- **Description**: Path to directory containing CSV data files
- **Notes**: All `.csv` files matching the naming convention will be loaded

### `output_path`
- **Type**: `str`
- **Default**: `'./output'`
- **Description**: Path for saving results, plots, and states

### `frame_rate`
- **Type**: `int`
- **Default**: `50`
- **Description**: Recording frame rate in Hz (frames per second)
- **Impact**: Affects velocity calculations and lag time conversion
- **✅ Recommendation**: Match your actual recording rate

### `screen_width` / `screen_height`
- **Type**: `int`
- **Default**: `1920` / `980`
- **Description**: Screen dimensions in pixels
- **Impact**: Used for coordinate validation and normalization

---

## Velocity Calculation

### `velocity_method`
- **Type**: `Literal['difference', 'savgol']`
- **Default**: `'savgol'`

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| `difference` | Fast, simple, no parameters | Noisy output | Quick analysis, high-SNR data |
| `savgol` | Smooth, preserves dynamics | Slower, requires tuning | Publication-quality analysis |

**✅ Recommendation**: Use `savgol` for final analysis, `difference` for quick exploration.

### `savgol_window`
- **Type**: `int` (must be odd)
- **Default**: `5`
- **Range**: `3` to `21`
- **Description**: Window size for Savitzky-Golay filter

| Value | Effect |
|-------|--------|
| 3-5 | Minimal smoothing, preserves rapid changes |
| 7-11 | Moderate smoothing, good balance |
| 13-21 | Heavy smoothing, may blur fast movements |

**✅ Recommendation**: Start with `5`, increase if velocities are too noisy.

### `savgol_polyorder`
- **Type**: `int`
- **Default**: `2`
- **Range**: `1` to `min(5, savgol_window - 1)`
- **Description**: Polynomial order for Savitzky-Golay filter

| Value | Effect |
|-------|--------|
| 1 | Linear fit - maximum smoothing |
| 2 | Quadratic - good for most cases |
| 3-5 | Higher order - preserves more detail |

**✅ Recommendation**: Use `2` (quadratic) for typical tracking data.

---

## Outlier Handling

### `outlier_method`
- **Type**: `Literal['none', 'iqr', 'zscore', 'mad']`
- **Default**: `'iqr'`

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| `none` | No removal | Preserves all data | Outliers affect statistics |
| `iqr` | Q1 - k*IQR, Q3 + k*IQR | Robust to non-normal data | May remove legitimate extremes |
| `zscore` | mean ± k*std | Simple, well-understood | Assumes normality |
| `mad` | median ± k*MAD | Very robust | May be over-conservative |

**✅ Recommendation**: Use `iqr` for most cases, `mad` if data has extreme outliers.

### `outlier_threshold`
- **Type**: `float`
- **Default**: `1.5`
- **Range**: `1.0` to `5.0`

| Value | Effect | Use Case |
|-------|--------|----------|
| 1.0 | Aggressive - removes ~7% of normal data | Very noisy data |
| 1.5 | Standard IQR rule | Most cases |
| 2.0-3.0 | Conservative | Clean data, preserve extremes |
| 4.0-5.0 | Very conservative - rarely removes | Trust all data points |

**✅ Recommendation**: Start with `1.5`, adjust based on data quality.

---

## Cross-Correlation Analysis

### `normalize_xcorr`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Normalize cross-correlation to [-1, 1] range

| Value | Effect |
|-------|--------|
| `True` | Correlation values in [-1, 1], comparable across trials |
| `False` | Raw covariance values, depends on signal amplitude |

**✅ Recommendation**: Always use `True` unless you need raw covariances.

### `max_lag_frames`
- **Type**: `int`
- **Default**: `50`
- **Description**: Maximum lag to search in frames

At 50 Hz frame rate:
- 50 frames = ±1 second lag
- 25 frames = ±0.5 second lag
- 100 frames = ±2 seconds lag

**✅ Recommendation**: Use `50` (1 second) for typical reaction times.

---

## Statistical Analysis

### `alpha`
- **Type**: `float`
- **Default**: `0.05`
- **Description**: Significance level for statistical tests

| Value | Effect |
|-------|--------|
| 0.001 | Very conservative, reduces false positives |
| 0.01 | Conservative |
| 0.05 | Standard significance level |
| 0.10 | Liberal, exploratory analysis |

### `correction_method`
- **Type**: `Literal['bonferroni', 'holm', 'fdr_bh', 'none']`
- **Default**: `'holm'`

| Method | Description | Use Case |
|--------|-------------|----------|
| `bonferroni` | Family-wise error rate, most conservative | Few comparisons |
| `holm` | Step-down, more powerful than Bonferroni | Most cases |
| `fdr_bh` | False discovery rate | Many comparisons |
| `none` | No correction | Single comparison |

**✅ Recommendation**: Use `holm` for a good balance of power and error control.

---

## Missing Data Handling

### `missing_data_method`
- **Type**: `Literal['drop', 'interpolate', 'forward_fill']`
- **Default**: `'interpolate'`

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| `drop` | Remove rows with missing values | Simple, no assumptions | Loses data, disrupts time series |
| `interpolate` | Linear interpolation | Preserves data length | Assumes linear motion |
| `forward_fill` | Fill with last known value | Good for brief gaps | Can create plateaus |

**✅ Recommendation**: Use `interpolate` for brief gaps, `drop` if many missing values.

### `max_gap_frames`
- **Type**: `int`
- **Default**: `5`
- **Description**: Maximum gap size to interpolate (larger gaps use drop)

---

## Configuration Examples

### High-Quality Publication Analysis

```python
config = Config(
    velocity_method='savgol',
    savgol_window=7,
    savgol_polyorder=2,
    outlier_method='iqr',
    outlier_threshold=1.5,
    normalize_xcorr=True,
    max_lag_frames=50,
    alpha=0.05,
    correction_method='holm'
)
```

### Quick Exploratory Analysis

```python
config = Config(
    velocity_method='difference',
    outlier_method='none',
    normalize_xcorr=True,
    max_lag_frames=25,
    alpha=0.10,
    correction_method='none'
)
```

### Conservative Analysis (Preserve All Data)

```python
config = Config(
    velocity_method='savgol',
    savgol_window=5,
    outlier_method='mad',
    outlier_threshold=3.0,
    missing_data_method='drop',
    alpha=0.01,
    correction_method='bonferroni'
)
```

### Noisy Data Analysis

```python
config = Config(
    velocity_method='savgol',
    savgol_window=11,  # More smoothing
    savgol_polyorder=2,
    outlier_method='mad',  # Robust to outliers
    outlier_threshold=1.5,
    normalize_xcorr=True
)
```

---

## Parameter Info Dictionary

The `PARAMETER_INFO` dictionary in `config.py` provides runtime-accessible descriptions:

```python
from tracking_analysis.core.config import PARAMETER_INFO

# Get info about a parameter
info = PARAMETER_INFO['velocity_method']
print(f"Description: {info['description']}")
print(f"Pros: {info['pros']}")
print(f"Cons: {info['cons']}")
```

This is used by the Streamlit UI to display help text for each parameter.
