# Usage Examples

This document provides comprehensive examples for common analysis tasks.

## Table of Contents

1. [Basic Analysis Workflow](#basic-analysis-workflow)
2. [Cross-Correlation Analysis](#cross-correlation-analysis)
3. [Statistical Comparisons](#statistical-comparisons)
4. [Visualization Examples](#visualization-examples)
5. [Animation Generation](#animation-generation)
6. [Report Generation](#report-generation)
7. [Advanced Filtering](#advanced-filtering)
8. [Batch Processing](#batch-processing)

---

## Basic Analysis Workflow

### Complete Pipeline

```python
from tracking_analysis.core import (
    Config, DataLoader, Preprocessor,
    MetricsCalculator, CrossCorrelationAnalyzer,
    StatisticalAnalyzer, StateManager
)

# 1. Configuration
config = Config(
    data_path='./data',
    output_path='./results',
    velocity_method='savgol',
    outlier_method='iqr',
    outlier_threshold=1.5,
    normalize_xcorr=True
)

# 2. Load Data
loader = DataLoader(config.data_path)
data = loader.load_all()
print(f"Loaded {len(data)} trials")
print(f"Participants: {data['participant_id'].unique()}")

# 3. Preprocess
preprocessor = Preprocessor(config)
processed_data = preprocessor.preprocess(data)

# 4. Calculate Metrics
metrics_calc = MetricsCalculator(config)
trial_metrics = metrics_calc.compute_trial_metrics(processed_data)

# 5. Cross-Correlation Analysis
xcorr_analyzer = CrossCorrelationAnalyzer(config)
xcorr_results = xcorr_analyzer.compute_batch_xcorr(processed_data)

# 6. Statistical Analysis
stats_analyzer = StatisticalAnalyzer(config)
stats_results = stats_analyzer.full_analysis(trial_metrics, 'rmse')

# 7. Save State
state_manager = StateManager('./results/states')
state_id = state_manager.save_state(
    config=config,
    trial_metrics=trial_metrics,
    xcorr_results=xcorr_results,
    statistical_results=stats_results
)
print(f"Saved state: {state_id}")
```

---

## Cross-Correlation Analysis

### Basic Cross-Correlation

```python
from tracking_analysis.core import CrossCorrelationAnalyzer, Config
import pandas as pd

config = Config(
    normalize_xcorr=True,
    max_lag_frames=50,
    frame_rate=50
)

analyzer = CrossCorrelationAnalyzer(config)

# Single trial analysis
trial_data = processed_data[processed_data['trial_id'] == 'trial_001']
xcorr = analyzer.compute_cross_correlation(trial_data)

print(f"Peak lag: {xcorr['peak_lag']} frames")
print(f"Peak lag (ms): {xcorr['peak_lag_ms']} ms")
print(f"Peak correlation: {xcorr['peak_correlation']:.3f}")
print(f"Tracking type: {xcorr['tracking_type']}")  # predictive, reactive, synchronous
```

### Batch Cross-Correlation with Interpretation

```python
# Compute for all trials
xcorr_results = analyzer.compute_batch_xcorr(processed_data)

# Summarize by condition
summary = xcorr_results.groupby('condition').agg({
    'peak_lag_ms': ['mean', 'std'],
    'peak_correlation': ['mean', 'std']
})
print(summary)

# Count tracking types
type_counts = xcorr_results.groupby(['condition', 'tracking_type']).size()
print(type_counts)
```

### Visualization of Cross-Correlation

```python
from tracking_analysis.visualization import StaticPlotter
import matplotlib.pyplot as plt

plotter = StaticPlotter(config)

# Plot cross-correlation function for a single trial
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Cross-correlation function
ax1 = axes[0]
lags = xcorr['lags']
corr = xcorr['correlation']
ax1.plot(lags / config.frame_rate * 1000, corr)  # Convert to ms
ax1.axvline(x=xcorr['peak_lag_ms'], color='r', linestyle='--', label=f"Peak: {xcorr['peak_lag_ms']:.1f} ms")
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax1.set_xlabel('Lag (ms)')
ax1.set_ylabel('Correlation')
ax1.set_title('Cross-Correlation Function')
ax1.legend()

# Right: Distribution of lags across all trials
ax2 = axes[1]
ax2.hist(xcorr_results['peak_lag_ms'], bins=30, edgecolor='black')
ax2.axvline(x=0, color='r', linestyle='--', label='Zero lag')
ax2.set_xlabel('Peak Lag (ms)')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Peak Lags')
ax2.legend()

plt.tight_layout()
plt.savefig('cross_correlation_analysis.png', dpi=150)
```

---

## Statistical Comparisons

### ANOVA Analysis

```python
from tracking_analysis.core import StatisticalAnalyzer

stats = StatisticalAnalyzer(config)

# Two-way ANOVA: SD size × Condition
anova_results = stats.two_way_anova(
    trial_metrics, 
    metric='rmse',
    factor1='sd_size',
    factor2='condition'
)

print("=== ANOVA Results ===")
print(f"SD Size effect: F={anova_results['f_factor1']:.2f}, p={anova_results['p_factor1']:.4f}")
print(f"Condition effect: F={anova_results['f_factor2']:.2f}, p={anova_results['p_factor2']:.4f}")
print(f"Interaction: F={anova_results['f_interaction']:.2f}, p={anova_results['p_interaction']:.4f}")
```

### Effect Sizes

```python
# Cohen's d for condition comparison
effect_size = stats.cohens_d(
    trial_metrics[trial_metrics['condition'] == 'dynamic']['rmse'],
    trial_metrics[trial_metrics['condition'] == 'static']['rmse']
)
print(f"Cohen's d (dynamic vs static): {effect_size:.3f}")

# Interpretation
if abs(effect_size) < 0.2:
    print("Negligible effect")
elif abs(effect_size) < 0.5:
    print("Small effect")
elif abs(effect_size) < 0.8:
    print("Medium effect")
else:
    print("Large effect")
```

### Post-Hoc Tests

```python
# Pairwise comparisons for SD sizes
posthoc = stats.posthoc_tests(
    trial_metrics, 
    metric='rmse', 
    groupby='sd_size',
    correction='holm'
)

print("=== Post-Hoc Comparisons ===")
for comparison in posthoc:
    print(f"{comparison['group1']} vs {comparison['group2']}: "
          f"t={comparison['t_stat']:.2f}, p_adj={comparison['p_adjusted']:.4f}")
```

---

## Visualization Examples

### Trajectory Plots

```python
from tracking_analysis.visualization import StaticPlotter

plotter = StaticPlotter(config)

# Single trial trajectory
trial_data = processed_data[processed_data['trial_id'] == 'trial_001']

fig = plotter.plot_trajectory(
    trial_data,
    show_target=True,
    show_mouse=True,
    show_error=True,  # Color by tracking error
    title="Trial 001 - Tracking Trajectory"
)
fig.savefig('trajectory_single.png', dpi=150)
```

### Comparison Boxplots

```python
# Compare RMSE across conditions
fig = plotter.boxplot_comparison(
    trial_metrics,
    metric='rmse',
    groupby='condition',
    title='Tracking Accuracy by Condition'
)
fig.savefig('boxplot_condition.png', dpi=150)

# Compare across SD sizes with condition as hue
fig = plotter.boxplot_comparison(
    trial_metrics,
    metric='rmse',
    groupby='sd_size',
    hue='condition',
    title='Tracking Accuracy by SD Size and Condition'
)
fig.savefig('boxplot_sd_condition.png', dpi=150)
```

### Heatmaps

```python
# Error heatmap
fig = plotter.error_heatmap(
    trial_data,
    bins=50,
    title='Tracking Error Distribution'
)
fig.savefig('error_heatmap.png', dpi=150)

# Correlation matrix
metrics_for_corr = trial_metrics[['rmse', 'mean_velocity', 'peak_lag_ms', 'peak_correlation']]
fig = plotter.correlation_heatmap(metrics_for_corr)
fig.savefig('correlation_matrix.png', dpi=150)
```

---

## Animation Generation

### MP4 Video

```python
from tracking_analysis.visualization import AnimationBuilder

animator = AnimationBuilder(config)

# Create animated replay of a trial
trial_data = processed_data[processed_data['trial_id'] == 'trial_001']

animator.create_tracking_animation(
    trial_data,
    output_path='trial_001_animation.mp4',
    fps=30,
    show_trail=True,
    trail_length=50
)
print("Animation saved to trial_001_animation.mp4")
```

### Interactive HTML

```python
# Create interactive Plotly animation
animator.create_interactive_animation(
    trial_data,
    output_path='trial_001_interactive.html',
    title='Interactive Trial Replay'
)
print("Interactive animation saved to trial_001_interactive.html")
```

### Batch Animation Generation

```python
import os

output_dir = './animations'
os.makedirs(output_dir, exist_ok=True)

# Get unique trials
trial_ids = processed_data['trial_id'].unique()[:10]  # First 10 trials

for trial_id in trial_ids:
    trial_data = processed_data[processed_data['trial_id'] == trial_id]
    
    animator.create_tracking_animation(
        trial_data,
        output_path=f'{output_dir}/{trial_id}.mp4',
        fps=25
    )
    print(f"Created animation for {trial_id}")
```

---

## Report Generation

### Markdown Report

```python
from tracking_analysis.visualization import ReportBuilder

builder = ReportBuilder(config, output_dir='./reports')

builder.generate_markdown_report(
    trial_metrics=trial_metrics,
    xcorr_results=xcorr_results,
    stat_results=stats_results,
    output_path='analysis_report.md',
    title='Blob Tracking Experiment Analysis',
    include_plots=True
)
print("Report saved to ./reports/analysis_report.md")
```

### HTML Report

```python
builder.generate_html_report(
    trial_metrics=trial_metrics,
    xcorr_results=xcorr_results,
    stat_results=stats_results,
    output_path='analysis_report.html',
    title='Blob Tracking Experiment Analysis',
    include_interactive=True  # Include Plotly plots
)
print("HTML report saved to ./reports/analysis_report.html")
```

---

## Advanced Filtering

### Filter by Multiple Criteria

```python
# Load all data
loader = DataLoader('./data')
all_data = loader.load_all()

# Filter by participant
data_1341 = all_data[all_data['participant_id'] == '1341']

# Filter by condition
dynamic_data = all_data[all_data['condition'] == 'dynamic']

# Filter by SD size
sd21_data = all_data[all_data['sd_size'] == 21]

# Combined filter
filtered = all_data[
    (all_data['participant_id'].isin(['1341', '3272'])) &
    (all_data['condition'] == 'dynamic') &
    (all_data['sd_size'].isin([21, 31]))
]
print(f"Filtered to {len(filtered)} rows")
```

### Custom Trial Selection

```python
# Get trial summary
trial_summary = trial_metrics.groupby('trial_id').agg({
    'rmse': 'mean',
    'condition': 'first',
    'sd_size': 'first',
    'participant_id': 'first'
})

# Find best performing trials
best_trials = trial_summary.nsmallest(10, 'rmse')['trial_id'].tolist()

# Find worst performing trials
worst_trials = trial_summary.nlargest(10, 'rmse')['trial_id'].tolist()

print("Best trials:", best_trials)
print("Worst trials:", worst_trials)
```

---

## Batch Processing

### Process Multiple Participants

```python
import pandas as pd
from pathlib import Path

results = []

# Get unique participants
loader = DataLoader('./data')
metadata = loader.get_metadata()
participants = metadata['participant_id'].unique()

for participant in participants:
    print(f"Processing {participant}...")
    
    # Load participant data
    participant_data = loader.load_all(participants=[participant])
    
    # Preprocess
    processed = preprocessor.preprocess(participant_data)
    
    # Calculate metrics
    metrics = metrics_calc.compute_trial_metrics(processed)
    
    # Aggregate per participant
    summary = {
        'participant_id': participant,
        'mean_rmse': metrics['rmse'].mean(),
        'std_rmse': metrics['rmse'].std(),
        'n_trials': len(metrics)
    }
    results.append(summary)

# Combine results
results_df = pd.DataFrame(results)
results_df.to_csv('./results/participant_summary.csv', index=False)
print(results_df)
```

### Parallel Processing (for large datasets)

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_participant(participant_id, config):
    """Process a single participant."""
    loader = DataLoader(config.data_path)
    preprocessor = Preprocessor(config)
    metrics_calc = MetricsCalculator(config)
    
    data = loader.load_all(participants=[participant_id])
    processed = preprocessor.preprocess(data)
    metrics = metrics_calc.compute_trial_metrics(processed)
    
    return {
        'participant_id': participant_id,
        'mean_rmse': metrics['rmse'].mean(),
        'std_rmse': metrics['rmse'].std()
    }

# Get participants
loader = DataLoader('./data')
metadata = loader.get_metadata()
participants = list(metadata['participant_id'].unique())

# Process in parallel
n_workers = min(4, multiprocessing.cpu_count())
results = []

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {
        executor.submit(process_participant, p, config): p 
        for p in participants
    }
    
    for future in as_completed(futures):
        participant = futures[future]
        try:
            result = future.result()
            results.append(result)
            print(f"Completed {participant}")
        except Exception as e:
            print(f"Error processing {participant}: {e}")

results_df = pd.DataFrame(results)
print(results_df)
```

---

## CLI Examples

```bash
# Show dataset information
python -m tracking_analysis.cli info ./data

# Basic analysis
python -m tracking_analysis.cli analyze --data-path ./data --output ./results

# Analysis with specific participants
python -m tracking_analysis.cli analyze \
    --data-path ./data \
    --output ./results \
    --participants 1341 3272 3311

# Analysis with custom parameters
python -m tracking_analysis.cli analyze \
    --data-path ./data \
    --output ./results \
    --velocity-method savgol \
    --outlier-method mad \
    --outlier-threshold 2.0 \
    --save-state \
    --verbose

# List saved states
python -m tracking_analysis.cli states list ./results/states

# Load and view a state
python -m tracking_analysis.cli states load ./results/states STATE_ID

# Launch web interface
python -m tracking_analysis.cli ui
```
