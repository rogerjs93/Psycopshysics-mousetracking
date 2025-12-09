# Quick Start Guide

This guide will get you up and running with the Tracking Analysis tool in under 5 minutes.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Your tracking data in CSV format

## Installation

### Step 1: Navigate to Project Directory

```bash
cd "c:/Users/roger/Desktop/Roger/studies/N-Neuroscience/Psychophysics and Reaction Time Methods/group project2"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Your First Analysis

### Option A: Use the Web Interface (Recommended)

```bash
streamlit run tracking_analysis/ui/app.py
```

This opens a browser window with an interactive dashboard where you can:

1. **Load Data**: Set your data path and filter participants/conditions
2. **Configure Analysis**: Adjust velocity calculation, outlier removal, etc.
3. **Run Analysis**: Click "Run Full Analysis Pipeline"
4. **View Results**: See metrics, plots, and statistical summaries
5. **Save State**: Export your analysis for later

### Option B: Use the Command Line

```bash
# Get dataset overview
python -m tracking_analysis.cli info ./data

# Run analysis
python -m tracking_analysis.cli analyze --data-path ./data --output ./results --save-state
```

### Option C: Use Python Code

```python
from tracking_analysis.core import Config, DataLoader, Preprocessor, MetricsCalculator

# Load data
loader = DataLoader('./data')
data = loader.load_all()
print(f"Loaded {len(data)} trials")

# Configure and preprocess
config = Config(data_path='./data')
preprocessor = Preprocessor(config)
processed = preprocessor.preprocess(data)

# Calculate metrics
calc = MetricsCalculator(config)
metrics = calc.compute_trial_metrics(processed)
print(metrics.groupby(['sd_size', 'condition'])['rmse'].mean())
```

## Understanding the Data

Your CSV files should contain:

| Frame | Target_X | Target_Y | Mouse_X | Mouse_Y |
|-------|----------|----------|---------|---------|
| 0     | 960.12   | 490.34   | 960.00  | 490.00  |
| 1     | 962.45   | 488.12   | 961.50  | 489.50  |
| ...   | ...      | ...      | ...     | ...     |

File names follow this pattern:
```
Participant_1341_Tracking_blob_experiment_21arcmin_v1_dynamic.csv
          ↓                                   ↓    ↓     ↓
   Participant ID                        SD size  ver condition
```

## Key Concepts

### SD Size (21, 31, 34 arcmin)
The standard deviation of the Gaussian blob - affects visibility and tracking difficulty.

### Condition (dynamic/static)
- **Dynamic**: With auditory feedback during tracking
- **Static**: Without auditory feedback

### RMSE (Root Mean Square Error)
Primary metric for tracking accuracy. Lower = better tracking.

### Cross-Correlation Lag
- **Positive lag**: Reactive tracking (mouse follows target)
- **Negative lag**: Predictive tracking (mouse anticipates target)

## Common Tasks

### Filter by Participants

```python
# In code
loader = DataLoader('./data')
data = loader.load_all(participants=['1341', '3272'])

# CLI
python -m tracking_analysis.cli analyze --data-path ./data --participants 1341 3272 --output ./results
```

### Compare Conditions

```python
from tracking_analysis.core import StatisticalAnalyzer

stats = StatisticalAnalyzer(config)
comparison = stats.compare_conditions(metrics, 'rmse')
print(comparison)
```

### Generate Visualizations

```python
from tracking_analysis.visualization import StaticPlotter

plotter = StaticPlotter(config)

# Boxplot comparing conditions
fig = plotter.boxplot_comparison(metrics, metric='rmse', groupby='condition')
fig.savefig('condition_comparison.png')

# Trajectory plot
fig = plotter.plot_trajectory(trial_data, show_target=True, show_mouse=True)
fig.savefig('trajectory.png')
```

### Save and Reload Analysis State

```python
from tracking_analysis.core import StateManager

# Save
manager = StateManager('./states')
state_id = manager.save_state(config=config, trial_metrics=metrics)
print(f"Saved as: {state_id}")

# Later, reload
loaded = manager.load_state(state_id)
metrics = loaded['trial_metrics']
```

## Troubleshooting

### "No data files found"
- Check that your data path is correct
- Ensure CSV files follow the naming convention
- Files must have `.csv` extension

### "ImportError: No module named..."
```bash
pip install -r requirements.txt
```

### "Memory error with large datasets"
- Process participants in batches
- Use `loader.load_file()` for individual files instead of `load_all()`

### "Streamlit won't start"
```bash
# Check if streamlit is installed
pip show streamlit

# If not installed
pip install streamlit
```

## Next Steps

- Read the full [README](../README.md) for detailed documentation
- Check [CONFIGURATION.md](CONFIGURATION.md) for parameter tuning
- See [EXAMPLES.md](EXAMPLES.md) for advanced usage patterns
- Run `pytest tracking_analysis/tests/` to verify installation
