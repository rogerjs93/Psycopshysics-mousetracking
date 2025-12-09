"""
Metrics Module
==============

Calculates tracking performance metrics including Euclidean error,
RMSE, tracking accuracy, and various derived measures.

Classes:
    MetricsCalculator: Main class for computing tracking metrics

Functions:
    euclidean_error: Calculate frame-by-frame tracking error
    rmse_per_trial: Calculate RMSE for each trial
    tracking_accuracy_by_condition: Aggregate metrics by experimental condition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .config import Config, FRAME_RATE, FRAME_DURATION_MS


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TrialMetrics:
    """
    Metrics for a single trial.
    
    Attributes:
        participant_id: Participant identifier
        size: Blob size (e.g., "21arcmin")
        size_pixels: Blob size in pixels
        condition: Experimental condition
        condition_label: Human-readable condition label
        filename: Source filename
        n_frames: Number of frames analyzed
        
        rmse: Root Mean Square Error
        mean_error: Mean Euclidean error
        std_error: Standard deviation of error
        median_error: Median error
        max_error: Maximum error
        min_error: Minimum error
        
        rmse_x: RMSE in X dimension
        rmse_y: RMSE in Y dimension
        
        tracking_gain: Ratio of mouse movement to target movement
        initial_error: Error at trial start
        final_error: Error at trial end
    """
    participant_id: str
    size: str
    size_pixels: int
    condition: str
    condition_label: str
    filename: str
    n_frames: int
    
    # Error metrics
    rmse: float
    mean_error: float
    std_error: float
    median_error: float
    max_error: float
    min_error: float
    
    # Dimensional RMSE
    rmse_x: float
    rmse_y: float
    
    # Additional metrics
    tracking_gain: float
    initial_error: float
    final_error: float


# =============================================================================
# CORE METRIC FUNCTIONS
# =============================================================================

def euclidean_error(
    target_x: np.ndarray,
    target_y: np.ndarray,
    mouse_x: np.ndarray,
    mouse_y: np.ndarray
) -> np.ndarray:
    """
    Calculate Euclidean distance between target and mouse positions.
    
    Error = sqrt((target_x - mouse_x)² + (target_y - mouse_y)²)
    
    Args:
        target_x: Target X positions
        target_y: Target Y positions
        mouse_x: Mouse X positions
        mouse_y: Mouse Y positions
        
    Returns:
        Array of Euclidean errors (same length as inputs)
        
    Example:
        >>> error = euclidean_error(
        ...     df['Target_X'].values, df['Target_Y'].values,
        ...     df['Mouse_X'].values, df['Mouse_Y'].values
        ... )
        >>> print(f"Mean error: {error.mean():.2f} pixels")
    """
    dx = target_x - mouse_x
    dy = target_y - mouse_y
    return np.sqrt(dx**2 + dy**2)


def calculate_rmse(errors: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    RMSE = sqrt(mean(errors²))
    
    Args:
        errors: Array of error values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.nanmean(errors**2))


def calculate_tracking_gain(
    target_x: np.ndarray,
    target_y: np.ndarray,
    mouse_x: np.ndarray,
    mouse_y: np.ndarray
) -> float:
    """
    Calculate tracking gain (ratio of response amplitude to target amplitude).
    
    Gain = mouse_movement_magnitude / target_movement_magnitude
    
    A gain of 1.0 indicates perfect scaling, <1 indicates undertracking,
    >1 indicates overtracking.
    
    Args:
        target_x, target_y: Target positions
        mouse_x, mouse_y: Mouse positions
        
    Returns:
        Tracking gain ratio
    """
    # Calculate total movement for target and mouse
    target_dx = np.diff(target_x)
    target_dy = np.diff(target_y)
    target_movement = np.sum(np.sqrt(target_dx**2 + target_dy**2))
    
    mouse_dx = np.diff(mouse_x)
    mouse_dy = np.diff(mouse_y)
    mouse_movement = np.sum(np.sqrt(mouse_dx**2 + mouse_dy**2))
    
    if target_movement == 0:
        return np.nan
    
    return mouse_movement / target_movement


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add error calculation columns to DataFrame.
    
    Adds columns:
    - Error: Euclidean distance between target and mouse
    - Error_X: X-dimension error (Target_X - Mouse_X)
    - Error_Y: Y-dimension error (Target_Y - Mouse_Y)
    
    Args:
        df: DataFrame with Target_X, Target_Y, Mouse_X, Mouse_Y columns
        
    Returns:
        DataFrame with added error columns
    """
    df = df.copy()
    
    df['Error_X'] = df['Target_X'] - df['Mouse_X']
    df['Error_Y'] = df['Target_Y'] - df['Mouse_Y']
    df['Error'] = np.sqrt(df['Error_X']**2 + df['Error_Y']**2)
    
    return df


# =============================================================================
# TRIAL-LEVEL METRICS
# =============================================================================

def compute_trial_metrics(df: pd.DataFrame) -> TrialMetrics:
    """
    Compute all metrics for a single trial.
    
    Args:
        df: DataFrame containing data for one trial
            Must have columns: Target_X, Target_Y, Mouse_X, Mouse_Y
            Should have metadata columns: participant_id, size, condition, etc.
            
    Returns:
        TrialMetrics dataclass with all computed metrics
    """
    # Extract position arrays
    target_x = df['Target_X'].values
    target_y = df['Target_Y'].values
    mouse_x = df['Mouse_X'].values
    mouse_y = df['Mouse_Y'].values
    
    # Calculate errors
    errors = euclidean_error(target_x, target_y, mouse_x, mouse_y)
    error_x = target_x - mouse_x
    error_y = target_y - mouse_y
    
    # Extract metadata (use defaults if not present)
    participant_id = df['participant_id'].iloc[0] if 'participant_id' in df.columns else 'unknown'
    size = df['size'].iloc[0] if 'size' in df.columns else 'unknown'
    size_pixels = df['size_pixels'].iloc[0] if 'size_pixels' in df.columns else 0
    condition = df['condition'].iloc[0] if 'condition' in df.columns else 'unknown'
    condition_label = df['condition_label'].iloc[0] if 'condition_label' in df.columns else condition
    filename = df['filename'].iloc[0] if 'filename' in df.columns else 'unknown'
    
    # Compute metrics
    return TrialMetrics(
        participant_id=participant_id,
        size=size,
        size_pixels=size_pixels,
        condition=condition,
        condition_label=condition_label,
        filename=filename,
        n_frames=len(df),
        
        rmse=calculate_rmse(errors),
        mean_error=np.nanmean(errors),
        std_error=np.nanstd(errors),
        median_error=np.nanmedian(errors),
        max_error=np.nanmax(errors),
        min_error=np.nanmin(errors),
        
        rmse_x=calculate_rmse(error_x),
        rmse_y=calculate_rmse(error_y),
        
        tracking_gain=calculate_tracking_gain(target_x, target_y, mouse_x, mouse_y),
        initial_error=errors[0] if len(errors) > 0 else np.nan,
        final_error=errors[-1] if len(errors) > 0 else np.nan
    )


def rmse_per_trial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RMSE for each trial in the dataset.
    
    Args:
        df: DataFrame with tracking data, must have 'filename' column
            to identify separate trials
            
    Returns:
        DataFrame with one row per trial containing RMSE and metadata
        
    Example:
        >>> rmse_df = rmse_per_trial(all_data)
        >>> rmse_df.groupby(['size', 'condition'])['rmse'].mean()
    """
    if 'filename' not in df.columns:
        raise ValueError("DataFrame must have 'filename' column to identify trials")
    
    results = []
    
    for filename in df['filename'].unique():
        trial_df = df[df['filename'] == filename]
        metrics = compute_trial_metrics(trial_df)
        
        results.append({
            'participant_id': metrics.participant_id,
            'size': metrics.size,
            'size_pixels': metrics.size_pixels,
            'condition': metrics.condition,
            'condition_label': metrics.condition_label,
            'filename': metrics.filename,
            'n_frames': metrics.n_frames,
            'rmse': metrics.rmse,
            'mean_error': metrics.mean_error,
            'std_error': metrics.std_error,
            'median_error': metrics.median_error,
            'max_error': metrics.max_error,
            'min_error': metrics.min_error,
            'rmse_x': metrics.rmse_x,
            'rmse_y': metrics.rmse_y,
            'tracking_gain': metrics.tracking_gain,
            'initial_error': metrics.initial_error,
            'final_error': metrics.final_error
        })
    
    return pd.DataFrame(results)


# =============================================================================
# AGGREGATED METRICS
# =============================================================================

def tracking_accuracy_by_condition(
    trial_metrics: pd.DataFrame,
    groupby: List[str] = ['size', 'condition']
) -> pd.DataFrame:
    """
    Aggregate trial metrics by experimental condition.
    
    Args:
        trial_metrics: DataFrame from rmse_per_trial()
        groupby: Columns to group by
        
    Returns:
        DataFrame with aggregated statistics per condition
        
    Example:
        >>> trial_metrics = rmse_per_trial(all_data)
        >>> summary = tracking_accuracy_by_condition(trial_metrics)
        >>> print(summary[['size', 'condition', 'rmse_mean', 'rmse_std']])
    """
    agg_funcs = {
        'rmse': ['mean', 'std', 'median', 'min', 'max', 'count'],
        'mean_error': ['mean', 'std'],
        'tracking_gain': ['mean', 'std'],
        'n_frames': ['mean']
    }
    
    # Filter to only columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in trial_metrics.columns}
    
    summary = trial_metrics.groupby(groupby).agg(agg_funcs)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Add participant count
    participant_counts = trial_metrics.groupby(groupby)['participant_id'].nunique()
    summary = summary.merge(
        participant_counts.reset_index().rename(columns={'participant_id': 'n_participants'}),
        on=groupby
    )
    
    return summary


def compute_discrimination_metrics(
    trial_metrics: pd.DataFrame,
    size: int,
    threshold: float = 50.0
) -> Dict[str, Any]:
    """
    Compute metrics to answer discrimination question for a specific blob size.
    
    "Can observers discriminate (track) blobs with X-pixel SD?"
    
    Args:
        trial_metrics: DataFrame from rmse_per_trial()
        size: Blob size in pixels (21, 31, or 34)
        threshold: RMSE threshold for "successful" tracking
        
    Returns:
        Dictionary with discrimination metrics and interpretation
    """
    size_str = f"{size}arcmin"
    size_data = trial_metrics[trial_metrics['size'] == size_str]
    
    if len(size_data) == 0:
        return {
            'size': size,
            'n_trials': 0,
            'error': f"No data for size {size_str}"
        }
    
    # Overall metrics
    mean_rmse = size_data['rmse'].mean()
    std_rmse = size_data['rmse'].std()
    median_rmse = size_data['rmse'].median()
    
    # Proportion of trials below threshold
    n_successful = (size_data['rmse'] < threshold).sum()
    n_total = len(size_data)
    success_rate = n_successful / n_total
    
    # By condition
    condition_stats = {}
    for condition in size_data['condition'].unique():
        cond_data = size_data[size_data['condition'] == condition]
        condition_stats[condition] = {
            'mean_rmse': cond_data['rmse'].mean(),
            'std_rmse': cond_data['rmse'].std(),
            'median_rmse': cond_data['rmse'].median(),
            'n_trials': len(cond_data),
            'success_rate': (cond_data['rmse'] < threshold).mean()
        }
    
    # Interpretation
    if mean_rmse < threshold:
        interpretation = f"YES - Observers CAN discriminate {size}-pixel SD blobs"
        interpretation += f" (Mean RMSE = {mean_rmse:.1f} < {threshold} threshold)"
    else:
        interpretation = f"NO - Observers CANNOT reliably discriminate {size}-pixel SD blobs"
        interpretation += f" (Mean RMSE = {mean_rmse:.1f} > {threshold} threshold)"
    
    return {
        'size': size,
        'size_str': size_str,
        'n_trials': n_total,
        'n_participants': size_data['participant_id'].nunique(),
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'median_rmse': median_rmse,
        'threshold': threshold,
        'n_successful': n_successful,
        'success_rate': success_rate,
        'condition_stats': condition_stats,
        'can_discriminate': mean_rmse < threshold,
        'interpretation': interpretation
    }


# =============================================================================
# TIME-SERIES METRICS
# =============================================================================

def compute_error_timeseries(
    df: pd.DataFrame,
    groupby: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute error statistics at each time point.
    
    Useful for visualizing how tracking error evolves over time.
    
    Args:
        df: DataFrame with tracking data (must have Error column or position columns)
        groupby: Optional grouping columns (e.g., ['size', 'condition'])
        
    Returns:
        DataFrame with error statistics at each frame
    """
    # Add error column if not present
    if 'Error' not in df.columns:
        df = add_error_columns(df)
    
    if groupby is None:
        # Overall time series
        timeseries = df.groupby('Frame').agg({
            'Error': ['mean', 'std', 'median', 'count']
        })
        timeseries.columns = ['error_mean', 'error_std', 'error_median', 'n_samples']
        timeseries = timeseries.reset_index()
        
        # Add time columns
        timeseries['time_ms'] = timeseries['Frame'] * FRAME_DURATION_MS
        timeseries['time_sec'] = timeseries['Frame'] / FRAME_RATE
        
        # Add confidence interval (95%)
        timeseries['error_ci95'] = 1.96 * timeseries['error_std'] / np.sqrt(timeseries['n_samples'])
        
    else:
        # Time series per group
        timeseries = df.groupby(groupby + ['Frame']).agg({
            'Error': ['mean', 'std', 'median', 'count']
        })
        timeseries.columns = ['error_mean', 'error_std', 'error_median', 'n_samples']
        timeseries = timeseries.reset_index()
        
        timeseries['time_ms'] = timeseries['Frame'] * FRAME_DURATION_MS
        timeseries['time_sec'] = timeseries['Frame'] / FRAME_RATE
        timeseries['error_ci95'] = 1.96 * timeseries['error_std'] / np.sqrt(timeseries['n_samples'])
    
    return timeseries


# =============================================================================
# METRICS CALCULATOR CLASS
# =============================================================================

class MetricsCalculator:
    """
    Main class for computing tracking metrics.
    
    Provides a unified interface for all metric calculations.
    
    Attributes:
        config: Configuration object
        
    Example:
        >>> calculator = MetricsCalculator(config)
        >>> 
        >>> # Compute trial-level metrics
        >>> trial_metrics = calculator.compute_all_trials(all_data)
        >>> 
        >>> # Get condition summary
        >>> summary = calculator.get_condition_summary(trial_metrics)
        >>> 
        >>> # Answer research questions
        >>> q1 = calculator.can_discriminate(trial_metrics, size=21)
        >>> print(q1['interpretation'])
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize MetricsCalculator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
    
    def add_error_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add error calculation columns to DataFrame."""
        return add_error_columns(df)
    
    def compute_all_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute metrics for all trials in the dataset.
        
        Args:
            df: DataFrame with all tracking data
            
        Returns:
            DataFrame with one row per trial containing all metrics
        """
        return rmse_per_trial(df)
    
    def get_condition_summary(
        self,
        trial_metrics: pd.DataFrame,
        groupby: List[str] = ['size', 'condition']
    ) -> pd.DataFrame:
        """
        Get summary statistics by condition.
        
        Args:
            trial_metrics: DataFrame from compute_all_trials()
            groupby: Columns to group by
            
        Returns:
            Summary DataFrame
        """
        return tracking_accuracy_by_condition(trial_metrics, groupby)
    
    def can_discriminate(
        self,
        trial_metrics: pd.DataFrame,
        size: int,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Answer discrimination question for a specific blob size.
        
        Args:
            trial_metrics: DataFrame from compute_all_trials()
            size: Blob size in pixels
            threshold: RMSE threshold (uses 50 if not specified)
            
        Returns:
            Dictionary with discrimination metrics and interpretation
        """
        if threshold is None:
            threshold = 50.0  # Default threshold
        
        return compute_discrimination_metrics(trial_metrics, size, threshold)
    
    def get_error_timeseries(
        self,
        df: pd.DataFrame,
        groupby: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get error statistics over time.
        
        Args:
            df: Tracking data DataFrame
            groupby: Optional grouping columns
            
        Returns:
            Time series DataFrame
        """
        return compute_error_timeseries(df, groupby)
    
    def compute_all_discrimination_questions(
        self,
        trial_metrics: pd.DataFrame,
        threshold: float = 50.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        Answer all three discrimination questions.
        
        Args:
            trial_metrics: DataFrame from compute_all_trials()
            threshold: RMSE threshold for success
            
        Returns:
            Dictionary with results for each blob size
        """
        results = {}
        
        for size in [21, 31, 34]:
            results[f"size_{size}"] = self.can_discriminate(
                trial_metrics, size, threshold
            )
        
        return results
    
    def get_participant_summary(
        self,
        trial_metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get summary statistics per participant.
        
        Args:
            trial_metrics: DataFrame from compute_all_trials()
            
        Returns:
            DataFrame with one row per participant
        """
        summary = trial_metrics.groupby('participant_id').agg({
            'rmse': ['mean', 'std', 'min', 'max'],
            'tracking_gain': ['mean', 'std'],
            'filename': 'count'
        })
        
        summary.columns = [
            'rmse_mean', 'rmse_std', 'rmse_min', 'rmse_max',
            'gain_mean', 'gain_std', 'n_trials'
        ]
        
        return summary.reset_index()
    
    def compare_conditions(
        self,
        trial_metrics: pd.DataFrame,
        metric: str = 'rmse'
    ) -> Dict[str, Any]:
        """
        Compare tracking performance between auditory feedback conditions.
        
        Args:
            trial_metrics: DataFrame from compute_all_trials()
            metric: Metric to compare (default: 'rmse')
            
        Returns:
            Dictionary with comparison statistics
        """
        dynamic = trial_metrics[trial_metrics['condition'] == 'dynamic'][metric]
        static = trial_metrics[trial_metrics['condition'] == 'static'][metric]
        
        dynamic_mean = dynamic.mean()
        static_mean = static.mean()
        
        # Effect direction
        if dynamic_mean < static_mean:
            effect = "Auditory feedback IMPROVES tracking"
            improvement = ((static_mean - dynamic_mean) / static_mean) * 100
        else:
            effect = "Auditory feedback does NOT improve tracking"
            improvement = ((dynamic_mean - static_mean) / static_mean) * -100
        
        return {
            'metric': metric,
            'dynamic_mean': dynamic_mean,
            'dynamic_std': dynamic.std(),
            'static_mean': static_mean,
            'static_std': static.std(),
            'difference': static_mean - dynamic_mean,
            'percent_change': improvement,
            'effect_direction': effect,
            'dynamic_n': len(dynamic),
            'static_n': len(static)
        }
