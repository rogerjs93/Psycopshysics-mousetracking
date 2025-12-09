# API Reference

This document provides detailed API documentation for all public classes and functions.

---

## Core Module (`tracking_analysis.core`)

### Config

Configuration dataclass for analysis parameters.

```python
from tracking_analysis.core import Config
```

#### Constructor

```python
Config(
    data_path: str = './data',
    output_path: str = './output',
    frame_rate: int = 50,
    screen_width: int = 1920,
    screen_height: int = 980,
    velocity_method: Literal['difference', 'savgol'] = 'savgol',
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
    outlier_method: Literal['none', 'iqr', 'zscore', 'mad'] = 'iqr',
    outlier_threshold: float = 1.5,
    normalize_xcorr: bool = True,
    max_lag_frames: int = 50,
    missing_data_method: Literal['drop', 'interpolate', 'forward_fill'] = 'interpolate',
    max_gap_frames: int = 5,
    alpha: float = 0.05,
    correction_method: Literal['bonferroni', 'holm', 'fdr_bh', 'none'] = 'holm'
)
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict()` | `dict` | Convert config to dictionary |
| `from_dict(d)` | `Config` | Create config from dictionary (classmethod) |
| `validate()` | `bool` | Validate configuration parameters |

---

### DataLoader

Load and parse tracking data from CSV files.

```python
from tracking_analysis.core import DataLoader
```

#### Constructor

```python
DataLoader(data_path: str, use_cache: bool = True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | `str` | required | Path to data directory |
| `use_cache` | `bool` | `True` | Cache loaded data for faster reloading |

#### Methods

##### `load_all()`

Load all data files matching the naming convention.

```python
def load_all(
    participants: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    sd_sizes: Optional[List[int]] = None
) -> pd.DataFrame
```

**Returns**: DataFrame with columns: `Frame`, `Target_X`, `Target_Y`, `Mouse_X`, `Mouse_Y`, `participant_id`, `condition`, `sd_size`, `trial_id`

##### `load_file()`

Load a single CSV file.

```python
def load_file(filepath: str) -> pd.DataFrame
```

##### `get_metadata()`

Get metadata for all available files.

```python
def get_metadata() -> pd.DataFrame
```

**Returns**: DataFrame with columns: `filepath`, `participant_id`, `condition`, `sd_size`, `version`

##### `clear_cache()`

Clear the data cache.

```python
def clear_cache() -> None
```

---

### Preprocessor

Preprocess tracking data (velocity, outliers, missing data).

```python
from tracking_analysis.core import Preprocessor
```

#### Constructor

```python
Preprocessor(config: Config)
```

#### Methods

##### `preprocess()`

Apply all preprocessing steps to data.

```python
def preprocess(
    data: pd.DataFrame,
    compute_velocity: bool = True,
    handle_outliers: bool = True,
    handle_missing: bool = True
) -> pd.DataFrame
```

**Returns**: DataFrame with added columns: `velocity_x`, `velocity_y`, `velocity_magnitude`, `target_velocity_x`, `target_velocity_y`, `target_velocity_magnitude`

##### `compute_velocity()`

Calculate velocity from position data.

```python
def compute_velocity(
    data: pd.DataFrame,
    position_cols: List[str] = ['Mouse_X', 'Mouse_Y']
) -> pd.DataFrame
```

##### `remove_outliers()`

Remove outliers from data.

```python
def remove_outliers(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame
```

##### `handle_missing_data()`

Handle missing values in data.

```python
def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame
```

---

### MetricsCalculator

Calculate tracking performance metrics.

```python
from tracking_analysis.core import MetricsCalculator
```

#### Constructor

```python
MetricsCalculator(config: Config)
```

#### Methods

##### `compute_trial_metrics()`

Compute all metrics for each trial.

```python
def compute_trial_metrics(data: pd.DataFrame) -> pd.DataFrame
```

**Returns**: DataFrame with columns: `trial_id`, `participant_id`, `condition`, `sd_size`, `rmse`, `rmse_x`, `rmse_y`, `mean_error`, `max_error`, `mean_velocity`, `std_velocity`

##### `compute_rmse()`

Compute Root Mean Square Error.

```python
def compute_rmse(
    data: pd.DataFrame,
    target_cols: Tuple[str, str] = ('Target_X', 'Target_Y'),
    response_cols: Tuple[str, str] = ('Mouse_X', 'Mouse_Y')
) -> float
```

##### `compute_tracking_error()`

Compute frame-by-frame tracking error.

```python
def compute_tracking_error(data: pd.DataFrame) -> pd.Series
```

---

### CrossCorrelationAnalyzer

Analyze velocity cross-correlations for lag detection.

```python
from tracking_analysis.core import CrossCorrelationAnalyzer
```

#### Constructor

```python
CrossCorrelationAnalyzer(config: Config)
```

#### Methods

##### `compute_cross_correlation()`

Compute cross-correlation for a single trial.

```python
def compute_cross_correlation(
    data: pd.DataFrame,
    target_col: str = 'target_velocity_magnitude',
    response_col: str = 'velocity_magnitude'
) -> dict
```

**Returns**: Dictionary with keys:
- `lags`: Array of lag values (in frames)
- `correlation`: Array of correlation values
- `peak_lag`: Lag at peak correlation (frames)
- `peak_lag_ms`: Lag at peak correlation (milliseconds)
- `peak_correlation`: Maximum correlation value
- `tracking_type`: 'predictive', 'reactive', or 'synchronous'

##### `compute_batch_xcorr()`

Compute cross-correlations for all trials.

```python
def compute_batch_xcorr(data: pd.DataFrame) -> pd.DataFrame
```

**Returns**: DataFrame with columns: `trial_id`, `participant_id`, `condition`, `sd_size`, `peak_lag`, `peak_lag_ms`, `peak_correlation`, `tracking_type`

---

### StatisticalAnalyzer

Perform statistical tests and compute effect sizes.

```python
from tracking_analysis.core import StatisticalAnalyzer
```

#### Constructor

```python
StatisticalAnalyzer(config: Config)
```

#### Methods

##### `two_way_anova()`

Perform two-way ANOVA.

```python
def two_way_anova(
    data: pd.DataFrame,
    metric: str,
    factor1: str,
    factor2: str
) -> dict
```

##### `one_way_anova()`

Perform one-way ANOVA.

```python
def one_way_anova(
    data: pd.DataFrame,
    metric: str,
    factor: str
) -> dict
```

##### `posthoc_tests()`

Perform post-hoc pairwise comparisons.

```python
def posthoc_tests(
    data: pd.DataFrame,
    metric: str,
    groupby: str,
    correction: Optional[str] = None
) -> List[dict]
```

##### `cohens_d()`

Calculate Cohen's d effect size.

```python
def cohens_d(
    group1: pd.Series,
    group2: pd.Series
) -> float
```

##### `compare_conditions()`

Compare two conditions with t-test and effect size.

```python
def compare_conditions(
    data: pd.DataFrame,
    metric: str,
    condition_col: str = 'condition'
) -> dict
```

##### `full_analysis()`

Run complete statistical analysis.

```python
def full_analysis(
    data: pd.DataFrame,
    metric: str
) -> dict
```

---

### StateManager

Save and load analysis states for reproducibility.

```python
from tracking_analysis.core import StateManager
```

#### Constructor

```python
StateManager(state_dir: str = './states')
```

#### Methods

##### `save_state()`

Save analysis state to disk.

```python
def save_state(
    config: Optional[Config] = None,
    trial_metrics: Optional[pd.DataFrame] = None,
    xcorr_results: Optional[pd.DataFrame] = None,
    statistical_results: Optional[dict] = None,
    custom_data: Optional[dict] = None,
    state_id: Optional[str] = None,
    description: str = ''
) -> str
```

**Returns**: State ID string

##### `load_state()`

Load a saved state from disk.

```python
def load_state(state_id: str) -> dict
```

**Returns**: Dictionary with keys: `config`, `trial_metrics`, `xcorr_results`, `statistical_results`, `custom_data`, `metadata`

##### `list_states()`

List all saved states.

```python
def list_states() -> List[dict]
```

**Returns**: List of dictionaries with state metadata

##### `delete_state()`

Delete a saved state.

```python
def delete_state(state_id: str) -> bool
```

---

## Visualization Module (`tracking_analysis.visualization`)

### StaticPlotter

Create static matplotlib visualizations.

```python
from tracking_analysis.visualization import StaticPlotter
```

#### Constructor

```python
StaticPlotter(config: Config)
```

#### Methods

##### `plot_trajectory()`

Plot tracking trajectory.

```python
def plot_trajectory(
    data: pd.DataFrame,
    show_target: bool = True,
    show_mouse: bool = True,
    show_error: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure
```

##### `boxplot_comparison()`

Create comparison boxplot.

```python
def boxplot_comparison(
    data: pd.DataFrame,
    metric: str,
    groupby: str,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure
```

##### `error_heatmap()`

Create tracking error heatmap.

```python
def error_heatmap(
    data: pd.DataFrame,
    bins: int = 50,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure
```

##### `correlation_heatmap()`

Create correlation matrix heatmap.

```python
def correlation_heatmap(
    data: pd.DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure
```

---

### AnimationBuilder

Create animated visualizations.

```python
from tracking_analysis.visualization import AnimationBuilder
```

#### Constructor

```python
AnimationBuilder(config: Config)
```

#### Methods

##### `create_tracking_animation()`

Create MP4 animation of tracking trial.

```python
def create_tracking_animation(
    data: pd.DataFrame,
    output_path: str,
    fps: int = 30,
    show_trail: bool = True,
    trail_length: int = 50,
    figsize: Tuple[int, int] = (12, 8)
) -> None
```

##### `create_interactive_animation()`

Create interactive HTML animation.

```python
def create_interactive_animation(
    data: pd.DataFrame,
    output_path: str,
    title: Optional[str] = None
) -> None
```

---

### ReportBuilder

Generate analysis reports.

```python
from tracking_analysis.visualization import ReportBuilder
```

#### Constructor

```python
ReportBuilder(config: Config, output_dir: str = './reports')
```

#### Methods

##### `generate_markdown_report()`

Generate Markdown report.

```python
def generate_markdown_report(
    trial_metrics: pd.DataFrame,
    xcorr_results: Optional[pd.DataFrame] = None,
    stat_results: Optional[dict] = None,
    output_path: str = 'report.md',
    title: str = 'Analysis Report',
    include_plots: bool = True
) -> str
```

**Returns**: Path to generated report

##### `generate_html_report()`

Generate HTML report.

```python
def generate_html_report(
    trial_metrics: pd.DataFrame,
    xcorr_results: Optional[pd.DataFrame] = None,
    stat_results: Optional[dict] = None,
    output_path: str = 'report.html',
    title: str = 'Analysis Report',
    include_interactive: bool = True
) -> str
```

---

## CLI Module (`tracking_analysis.cli`)

### Command-Line Interface

```bash
python -m tracking_analysis.cli <command> [options]
```

#### Commands

| Command | Description |
|---------|-------------|
| `info` | Show dataset information |
| `analyze` | Run analysis pipeline |
| `states` | Manage saved states |
| `ui` | Launch Streamlit interface |

For detailed options, run:

```bash
python -m tracking_analysis.cli <command> --help
```

---

## Constants and Types

### PARAMETER_INFO

Dictionary containing parameter descriptions, pros, and cons.

```python
from tracking_analysis.core.config import PARAMETER_INFO

info = PARAMETER_INFO['velocity_method']
# {'description': '...', 'pros': '...', 'cons': '...', 'options': [...]}
```

### Type Aliases

```python
from typing import Literal

VelocityMethod = Literal['difference', 'savgol']
OutlierMethod = Literal['none', 'iqr', 'zscore', 'mad']
MissingDataMethod = Literal['drop', 'interpolate', 'forward_fill']
CorrectionMethod = Literal['bonferroni', 'holm', 'fdr_bh', 'none']
TrackingType = Literal['predictive', 'reactive', 'synchronous']
```
