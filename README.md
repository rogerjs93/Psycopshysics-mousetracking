# 🎯 Tracking Analysis Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive application for analyzing psychophysics tracking data from blob tracking experiments with auditory feedback conditions.

<p align="center">
  <img src="https://img.shields.io/badge/Analysis-RMSE%20%7C%20Cross--Correlation%20%7C%20Statistics-blueviolet" alt="Analysis Types">
</p>

---

## ✨ Features

### 📊 Data Loading
- **Drag & Drop Upload** - Upload CSV files directly in the browser
- **Folder Path Input** - Load data from a local directory

### 🔬 Analysis Capabilities
- **Multiple velocity methods**: Simple difference or Savitzky-Golay smoothing
- **Robust outlier handling**: IQR, Z-score, or MAD methods  
- **Cross-correlation analysis**: Detect predictive vs reactive tracking patterns
- **Statistical analysis**: Mann-Whitney U, Kruskal-Wallis, Spearman correlations
- **Rich visualizations**: Interactive plots, heatmaps, and animated trial replays

### 📈 Visualizations
- RMSE box plots and violin plots by condition
- Cross-correlation lag distributions
- Animated tracking trajectory playback
- Psychometric functions (performance vs blob size)
- Learning curves across trials
- Speed-accuracy tradeoff analysis
- Pursuit gain distributions

### 💾 Export & Save
- Save/Load analysis states for reproducibility
- Export results as JSON or Markdown reports
- Download high-quality figures

## 📊 Research Questions

This tool helps answer questions like:

1. **Does auditory feedback improve tracking accuracy?** (dynamic vs static conditions)
2. **Can observers discriminate blobs with different SD sizes?** (21, 31, 34 arcmin)
3. **Is tracking predictive or reactive?** (cross-correlation lag analysis)
4. **Are there individual differences in tracking ability?**

---

## 🚀 Quick Start

### Option 1: Standalone Application (Recommended)

**No Python installation required!**

1. Download `TrackingAnalysis.zip` from the [Releases](../../releases) page
2. Extract to any folder
3. Run `TrackingAnalysis.exe`
4. Your browser will open automatically at `http://localhost:8501`

### Option 2: Run from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/tracking-analysis.git
cd tracking-analysis

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run tracking_analysis/ui/app.py
```

### Using the Command Line

```bash
# Show dataset info
python -m tracking_analysis.cli info ./data

# Run full analysis
python -m tracking_analysis.cli analyze --data-path ./data --output ./results

# Run with custom parameters
python -m tracking_analysis.cli analyze \
    --data-path ./data \
    --output ./results \
    --velocity-method savgol \
    --outlier-method iqr \
    --save-state
```

### Using as a Python Library

```python
from tracking_analysis.core import (
    Config, DataLoader, Preprocessor,
    MetricsCalculator, CrossCorrelationAnalyzer,
    StatisticalAnalyzer, StateManager
)

# Configure
config = Config(
    data_path='./data',
    velocity_method='savgol',
    outlier_method='iqr',
    normalize_xcorr=True
)

# Load data
loader = DataLoader(config.data_path)
data = loader.load_all()

# Preprocess
preprocessor = Preprocessor(config)
processed_data = preprocessor.preprocess(data)

# Calculate metrics
metrics_calc = MetricsCalculator(config)
trial_metrics = metrics_calc.compute_trial_metrics(processed_data)

# Cross-correlation analysis
xcorr_analyzer = CrossCorrelationAnalyzer(config)
xcorr_results = xcorr_analyzer.compute_batch_xcorr(processed_data)

# Statistics
stats_analyzer = StatisticalAnalyzer(config)
results = stats_analyzer.generate_results_summary(trial_metrics, metric='rmse')

print(results)
```

---

## 📖 Usage Guide

### 1. Load Data
- Use **drag & drop** to upload CSV files, OR
- Enter the **folder path** containing your data files

### 2. Run Analysis
- Click **▶️ Run Analysis** to process all trials
- Default configuration is applied automatically
- (Optional) Visit **⚙️ Configure** to customize parameters

### 3. View Results
- **📊 Results Overview** - Summary statistics and metrics
- **📈 Detailed Analysis** - In-depth trial-by-trial analysis
- **📉 Visualizations** - Interactive plots and charts

### 4. Research Questions
- Select analysis scope (all data or filtered subset)
- Configure thresholds for performance classification
- View statistical test results with interpretations

### 5. Export
- Download results as JSON or Markdown
- Save analysis state for later

---

## 📁 Data Format

### Expected CSV Structure

| Column | Description |
|--------|-------------|
| `Frame` | Frame number (0-indexed) |
| `Target_X` | Target X position (pixels) |
| `Target_Y` | Target Y position (pixels) |
| `Mouse_X` | Mouse/response X position (pixels) |
| `Mouse_Y` | Mouse/response Y position (pixels) |

### File Naming Convention

```
Participant_XXXX_Tracking_blob_experiment_XXarcmin_vX_dynamic.csv
Participant_XXXX_Tracking_blob_experiment_XXarcmin_vX_static.csv
```

Where:
- `XXXX`: Participant ID (4 digits)
- `XX`: SD size in arcmin (21, 31, or 34)
- `dynamic/static`: Condition (with/without auditory feedback)

## ⚙️ Configuration Parameters

### Velocity Calculation

| Parameter | Options | Default | Notes |
|-----------|---------|---------|-------|
| `velocity_method` | `difference`, `savgol` | `savgol` | Savgol is smoother but slower |
| `savgol_window` | 3-21 (odd) | 5 | Larger = more smoothing |
| `savgol_polyorder` | 1-5 | 2 | Must be < window |

### Outlier Removal

| Parameter | Options | Default | Notes |
|-----------|---------|---------|-------|
| `outlier_method` | `none`, `iqr`, `zscore`, `mad` | `iqr` | IQR robust to non-normal data |
| `outlier_threshold` | 1.0-5.0 | 1.5 | Lower = more aggressive |

### Cross-Correlation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `normalize_xcorr` | `True` | Bounds values to [-1, 1] |
| `max_lag_frames` | 50 | At 50fps: 1 second max lag |
| `frame_rate` | 50 | For time conversion |

## 📈 Understanding Results

### RMSE (Root Mean Square Error)

Measures overall tracking accuracy. Lower values indicate better tracking performance.

### Cross-Correlation Lag

| Lag Value | Interpretation |
|-----------|----------------|
| Positive | Reactive tracking (mouse follows target) |
| Negative | Predictive tracking (mouse leads target) |
| Zero | Synchronous tracking |

### Effect Sizes

| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

## 🧪 Running Tests

```bash
# Run all tests
pytest tracking_analysis/tests/

# Run with coverage
pytest tracking_analysis/tests/ --cov=tracking_analysis

# Run specific test file
pytest tracking_analysis/tests/test_data_loader.py -v
```

## 📂 Project Structure

```
tracking_analysis/
├── core/                    # Core analysis modules
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # CSV loading and metadata
│   ├── preprocessing.py    # Velocity, outliers, missing data
│   ├── metrics.py          # RMSE and tracking metrics
│   ├── cross_correlation.py # Velocity cross-correlation
│   ├── statistics.py       # Statistical tests
│   └── state_manager.py    # State persistence
├── visualization/           # Visualization modules
│   ├── static_plots.py     # Matplotlib plots
│   ├── animation.py        # Animated visualizations
│   └── report_builder.py   # Report generation
├── cli/                     # Command-line interface
│   └── main.py             # CLI commands
├── ui/                      # Streamlit web interface
│   └── app.py              # Main Streamlit app
└── tests/                   # Test suite
    ├── conftest.py         # Shared fixtures
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_metrics.py
    ├── test_cross_correlation.py
    ├── test_statistics.py
    └── test_state_manager.py
```

## 🔄 State Management

Save analysis states for reproducibility and fast reloading:

```python
from tracking_analysis.core import StateManager

# Save state
manager = StateManager('./results/states')
state_id = manager.save_state(
    config=config,
    trial_metrics=trial_metrics,
    xcorr_results=xcorr_results,
    statistical_results=stats_results
)

# List states
states = manager.list_states()

# Load state
loaded = manager.load_state(state_id)
```

## 📋 CLI Reference

```
tracking_analysis - Psychophysics Tracking Data Analysis Tool

Commands:
  info       Show dataset information
  analyze    Run analysis pipeline
  states     Manage saved states
  ui         Launch Streamlit web interface

analyze options:
  --data-path, -d      Path to data directory (required)
  --output, -o         Output directory (required)
  --participants       Filter by participant IDs
  --conditions         Filter by conditions (dynamic, static)
  --sd-sizes           Filter by SD sizes (21, 31, 34)
  --velocity-method    Velocity calculation (difference, savgol)
  --outlier-method     Outlier removal (none, iqr, zscore, mad)
  --outlier-threshold  Outlier threshold value
  --normalize-xcorr    Normalize cross-correlation
  --max-lag            Maximum lag frames
  --save-state         Save analysis state
  --verbose, -v        Verbose output
```

## 📊 Generating Reports

```python
from tracking_analysis.visualization import ReportBuilder

builder = ReportBuilder(output_dir='./reports')

# Generate Markdown report
builder.generate_markdown_report(
    trial_metrics=trial_metrics,
    xcorr_results=xcorr_results,
    stat_results=stat_results,
    output_path='analysis_report.md'
)

# Generate HTML report
builder.generate_html_report(
    trial_metrics=trial_metrics,
    output_path='analysis_report.html'
)
```

## 📝 License

MIT License - see LICENSE file for details.

## 👥 Authors

Psychophysics and Reaction Time Methods - Group Project

---

<p align="center">
  Built with ❤️ using <a href="https://streamlit.io/">Streamlit</a>
</p>
