"""
Cross-Correlation Module
========================

Implements cross-correlation analysis between target and mouse velocity
signals to determine tracking lag, predictive vs. reactive behavior,
and correlation strength.

Classes:
    CrossCorrelationAnalyzer: Main class for cross-correlation analysis

Functions:
    velocity_crosscorr: Compute cross-correlation between velocity signals
    find_optimal_lag: Find lag with maximum correlation
    aggregate_correlations: Aggregate cross-correlation results by condition
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .config import Config, FRAME_RATE, FRAME_DURATION_MS
from .preprocessing import calculate_velocity


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CrossCorrelationResult:
    """
    Result of cross-correlation analysis for a single trial.
    
    Attributes:
        participant_id: Participant identifier
        size: Blob size
        condition: Experimental condition
        filename: Source filename
        
        optimal_lag_frames: Lag with maximum correlation (in frames)
        optimal_lag_ms: Lag in milliseconds
        max_correlation: Maximum correlation value
        
        correlation_at_zero: Correlation at zero lag
        is_predictive: Whether tracking anticipates target (negative lag)
        is_reactive: Whether tracking follows target (positive lag)
        
        lags: Array of all lag values computed
        correlations_x: Cross-correlation for X velocity
        correlations_y: Cross-correlation for Y velocity
        correlations_combined: Combined X+Y correlation
    """
    participant_id: str
    size: str
    condition: str
    filename: str
    
    optimal_lag_frames: int
    optimal_lag_ms: float
    max_correlation: float
    
    correlation_at_zero: float
    is_predictive: bool
    is_reactive: bool
    
    lags: np.ndarray
    correlations_x: np.ndarray
    correlations_y: np.ndarray
    correlations_combined: np.ndarray


# =============================================================================
# CORE CROSS-CORRELATION FUNCTIONS
# =============================================================================

def normalized_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation between two signals.
    
    Normalizes to [-1, 1] range for interpretability.
    
    Args:
        signal1: First signal (e.g., target velocity)
        signal2: Second signal (e.g., mouse velocity)
        max_lag: Maximum lag to compute (both directions)
        
    Returns:
        Tuple of (lags, correlation_values)
        
    Notes:
        - Positive lag means signal2 follows signal1
        - Negative lag means signal2 leads signal1
        - Correlation range [-1, 1] after normalization
    """
    # Remove mean (center signals)
    s1 = signal1 - np.nanmean(signal1)
    s2 = signal2 - np.nanmean(signal2)
    
    # Handle NaN values
    s1 = np.nan_to_num(s1, nan=0.0)
    s2 = np.nan_to_num(s2, nan=0.0)
    
    # Compute full cross-correlation
    correlation = signal.correlate(s2, s1, mode='full')
    
    # Generate lag array
    n = len(signal1)
    lags = np.arange(-(n-1), n)
    
    # Normalize by the product of standard deviations and length
    norm_factor = n * np.std(s1) * np.std(s2)
    if norm_factor > 0:
        correlation = correlation / norm_factor
    
    # Trim to max_lag if specified
    if max_lag is not None:
        center = len(lags) // 2
        start = center - max_lag
        end = center + max_lag + 1
        
        if start >= 0 and end <= len(lags):
            lags = lags[start:end]
            correlation = correlation[start:end]
    
    return lags, correlation


def raw_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute raw (unnormalized) cross-correlation.
    
    Preserves magnitude information but results are not directly comparable
    across different signal pairs.
    
    Args:
        signal1: First signal
        signal2: Second signal
        max_lag: Maximum lag to compute
        
    Returns:
        Tuple of (lags, correlation_values)
    """
    s1 = signal1 - np.nanmean(signal1)
    s2 = signal2 - np.nanmean(signal2)
    
    s1 = np.nan_to_num(s1, nan=0.0)
    s2 = np.nan_to_num(s2, nan=0.0)
    
    correlation = signal.correlate(s2, s1, mode='full')
    
    n = len(signal1)
    lags = np.arange(-(n-1), n)
    
    if max_lag is not None:
        center = len(lags) // 2
        start = center - max_lag
        end = center + max_lag + 1
        
        if start >= 0 and end <= len(lags):
            lags = lags[start:end]
            correlation = correlation[start:end]
    
    return lags, correlation


def velocity_crosscorr(
    target_velocity: np.ndarray,
    mouse_velocity: np.ndarray,
    normalize: bool = True,
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between target and mouse velocity.
    
    Args:
        target_velocity: Target velocity signal
        mouse_velocity: Mouse velocity signal
        normalize: Whether to normalize to [-1, 1]
        max_lag: Maximum lag in frames
        
    Returns:
        Tuple of (lags, correlations)
        
    Example:
        >>> lags, corr = velocity_crosscorr(target_vx, mouse_vx, normalize=True)
        >>> optimal_lag = lags[np.argmax(corr)]
        >>> max_corr = np.max(corr)
    """
    if normalize:
        return normalized_cross_correlation(target_velocity, mouse_velocity, max_lag)
    else:
        return raw_cross_correlation(target_velocity, mouse_velocity, max_lag)


def find_optimal_lag(
    lags: np.ndarray,
    correlations: np.ndarray
) -> Tuple[int, float]:
    """
    Find the lag with maximum correlation.
    
    Args:
        lags: Array of lag values
        correlations: Array of correlation values
        
    Returns:
        Tuple of (optimal_lag, max_correlation)
        
    Notes:
        - Positive optimal_lag: Mouse follows target (reactive tracking)
        - Negative optimal_lag: Mouse anticipates target (predictive tracking)
        - Zero optimal_lag: Synchronous tracking
    """
    max_idx = np.argmax(correlations)
    optimal_lag = lags[max_idx]
    max_correlation = correlations[max_idx]
    
    return int(optimal_lag), float(max_correlation)


def correlation_at_lag(
    lags: np.ndarray,
    correlations: np.ndarray,
    target_lag: int = 0
) -> float:
    """
    Get correlation value at a specific lag.
    
    Args:
        lags: Array of lag values
        correlations: Array of correlation values
        target_lag: Lag to query
        
    Returns:
        Correlation at specified lag, or NaN if lag not in range
    """
    idx = np.where(lags == target_lag)[0]
    if len(idx) > 0:
        return float(correlations[idx[0]])
    return np.nan


# =============================================================================
# TRIAL-LEVEL CROSS-CORRELATION
# =============================================================================

def compute_trial_crosscorr(
    df: pd.DataFrame,
    normalize: bool = True,
    max_lag_frames: int = 50,
    velocity_method: str = 'difference',
    smooth_window: int = 5
) -> CrossCorrelationResult:
    """
    Compute cross-correlation analysis for a single trial.
    
    Calculates cross-correlation between target and mouse velocity
    in both X and Y dimensions, then combines results.
    
    Args:
        df: DataFrame with trial data (must have position columns)
        normalize: Whether to normalize correlations
        max_lag_frames: Maximum lag to compute
        velocity_method: 'difference' or 'savgol'
        smooth_window: Window for Savitzky-Golay smoothing
        
    Returns:
        CrossCorrelationResult with all analysis outputs
    """
    # Calculate velocity if not present
    if 'Target_Vx' not in df.columns:
        df = calculate_velocity(df, method=velocity_method, smooth_window=smooth_window)
    
    # Extract velocity arrays
    target_vx = df['Target_Vx'].values
    target_vy = df['Target_Vy'].values
    mouse_vx = df['Mouse_Vx'].values
    mouse_vy = df['Mouse_Vy'].values
    
    # Compute cross-correlation for X and Y
    lags_x, corr_x = velocity_crosscorr(target_vx, mouse_vx, normalize, max_lag_frames)
    lags_y, corr_y = velocity_crosscorr(target_vy, mouse_vy, normalize, max_lag_frames)
    
    # Combined correlation (average of X and Y)
    # Both should have same lags
    corr_combined = (corr_x + corr_y) / 2
    
    # Find optimal lag from combined
    optimal_lag, max_corr = find_optimal_lag(lags_x, corr_combined)
    
    # Get correlation at zero lag
    corr_at_zero = correlation_at_lag(lags_x, corr_combined, 0)
    
    # Extract metadata
    participant_id = df['participant_id'].iloc[0] if 'participant_id' in df.columns else 'unknown'
    size = df['size'].iloc[0] if 'size' in df.columns else 'unknown'
    condition = df['condition'].iloc[0] if 'condition' in df.columns else 'unknown'
    filename = df['filename'].iloc[0] if 'filename' in df.columns else 'unknown'
    
    return CrossCorrelationResult(
        participant_id=participant_id,
        size=size,
        condition=condition,
        filename=filename,
        
        optimal_lag_frames=optimal_lag,
        optimal_lag_ms=optimal_lag * FRAME_DURATION_MS,
        max_correlation=max_corr,
        
        correlation_at_zero=corr_at_zero,
        is_predictive=optimal_lag < 0,
        is_reactive=optimal_lag > 0,
        
        lags=lags_x,
        correlations_x=corr_x,
        correlations_y=corr_y,
        correlations_combined=corr_combined
    )


def crosscorr_per_trial(
    df: pd.DataFrame,
    normalize: bool = True,
    max_lag_frames: int = 50,
    velocity_method: str = 'difference',
    smooth_window: int = 5
) -> pd.DataFrame:
    """
    Compute cross-correlation for all trials in dataset.
    
    Args:
        df: DataFrame with all tracking data
        normalize: Whether to normalize correlations
        max_lag_frames: Maximum lag to compute
        velocity_method: Velocity calculation method
        smooth_window: Smoothing window size
        
    Returns:
        DataFrame with cross-correlation results per trial
    """
    if 'filename' not in df.columns:
        raise ValueError("DataFrame must have 'filename' column to identify trials")
    
    results = []
    
    for filename in df['filename'].unique():
        trial_df = df[df['filename'] == filename].copy()
        
        try:
            xcorr_result = compute_trial_crosscorr(
                trial_df,
                normalize=normalize,
                max_lag_frames=max_lag_frames,
                velocity_method=velocity_method,
                smooth_window=smooth_window
            )
            
            results.append({
                'participant_id': xcorr_result.participant_id,
                'size': xcorr_result.size,
                'condition': xcorr_result.condition,
                'filename': xcorr_result.filename,
                'optimal_lag_frames': xcorr_result.optimal_lag_frames,
                'optimal_lag_ms': xcorr_result.optimal_lag_ms,
                'max_correlation': xcorr_result.max_correlation,
                'correlation_at_zero': xcorr_result.correlation_at_zero,
                'is_predictive': xcorr_result.is_predictive,
                'is_reactive': xcorr_result.is_reactive
            })
        except Exception as e:
            print(f"Warning: Could not compute cross-correlation for {filename}: {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# AGGREGATED ANALYSIS
# =============================================================================

def aggregate_correlations(
    xcorr_df: pd.DataFrame,
    groupby: List[str] = ['size', 'condition']
) -> pd.DataFrame:
    """
    Aggregate cross-correlation results by experimental condition.
    
    Args:
        xcorr_df: DataFrame from crosscorr_per_trial()
        groupby: Columns to group by
        
    Returns:
        Summary DataFrame with statistics per condition
    """
    agg_funcs = {
        'optimal_lag_frames': ['mean', 'std', 'median'],
        'optimal_lag_ms': ['mean', 'std', 'median'],
        'max_correlation': ['mean', 'std', 'median', 'min', 'max'],
        'correlation_at_zero': ['mean', 'std'],
        'is_predictive': ['sum', 'mean'],  # Count and proportion
        'is_reactive': ['sum', 'mean']
    }
    
    summary = xcorr_df.groupby(groupby).agg(agg_funcs)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Add counts
    counts = xcorr_df.groupby(groupby).size().reset_index(name='n_trials')
    summary = summary.merge(counts, on=groupby)
    
    # Add participant count
    participant_counts = xcorr_df.groupby(groupby)['participant_id'].nunique()
    summary = summary.merge(
        participant_counts.reset_index().rename(columns={'participant_id': 'n_participants'}),
        on=groupby
    )
    
    return summary


def compute_average_crosscorr_curve(
    df: pd.DataFrame,
    groupby: Optional[List[str]] = None,
    normalize: bool = True,
    max_lag_frames: int = 50,
    velocity_method: str = 'difference'
) -> Dict[str, Any]:
    """
    Compute average cross-correlation curve across trials.
    
    Useful for visualizing the typical cross-correlation pattern.
    
    Args:
        df: DataFrame with tracking data
        groupby: Optional grouping columns
        normalize: Whether to normalize
        max_lag_frames: Maximum lag
        velocity_method: Velocity calculation method
        
    Returns:
        Dictionary with average curves and statistics
    """
    if 'filename' not in df.columns:
        raise ValueError("DataFrame must have 'filename' column")
    
    all_curves = []
    lags_ref = None
    
    for filename in df['filename'].unique():
        trial_df = df[df['filename'] == filename].copy()
        
        try:
            xcorr_result = compute_trial_crosscorr(
                trial_df,
                normalize=normalize,
                max_lag_frames=max_lag_frames,
                velocity_method=velocity_method
            )
            
            all_curves.append(xcorr_result.correlations_combined)
            if lags_ref is None:
                lags_ref = xcorr_result.lags
        except:
            continue
    
    if not all_curves:
        return {'error': 'No valid curves computed'}
    
    # Stack and compute statistics
    curves_array = np.vstack(all_curves)
    
    mean_curve = np.mean(curves_array, axis=0)
    std_curve = np.std(curves_array, axis=0)
    se_curve = std_curve / np.sqrt(len(all_curves))
    ci95_curve = 1.96 * se_curve
    
    # Find optimal lag of average curve
    optimal_lag, max_corr = find_optimal_lag(lags_ref, mean_curve)
    
    return {
        'lags': lags_ref,
        'mean': mean_curve,
        'std': std_curve,
        'se': se_curve,
        'ci95': ci95_curve,
        'n_trials': len(all_curves),
        'optimal_lag_frames': optimal_lag,
        'optimal_lag_ms': optimal_lag * FRAME_DURATION_MS,
        'max_correlation': max_corr
    }


# =============================================================================
# INTERPRETATION HELPERS
# =============================================================================

def interpret_lag(lag_frames: int, lag_ms: float) -> str:
    """
    Generate human-readable interpretation of optimal lag.
    
    Args:
        lag_frames: Lag in frames
        lag_ms: Lag in milliseconds
        
    Returns:
        Interpretation string
    """
    if lag_frames < 0:
        return f"PREDICTIVE: Mouse anticipates target by {abs(lag_ms):.0f}ms ({abs(lag_frames)} frames)"
    elif lag_frames > 0:
        return f"REACTIVE: Mouse follows target by {lag_ms:.0f}ms ({lag_frames} frames)"
    else:
        return "SYNCHRONOUS: Mouse and target move simultaneously"


def interpret_correlation(correlation: float) -> str:
    """
    Generate interpretation of correlation strength.
    
    Args:
        correlation: Correlation value [-1, 1]
        
    Returns:
        Interpretation string
    """
    abs_corr = abs(correlation)
    
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    elif abs_corr >= 0.2:
        strength = "weak"
    else:
        strength = "very weak/negligible"
    
    if correlation >= 0:
        direction = "positive"
    else:
        direction = "negative"
    
    return f"{strength.capitalize()} {direction} correlation (r = {correlation:.3f})"


# =============================================================================
# CROSS-CORRELATION ANALYZER CLASS
# =============================================================================

class CrossCorrelationAnalyzer:
    """
    Main class for cross-correlation analysis.
    
    Provides a unified interface for:
    - Computing trial-level cross-correlations
    - Aggregating results by condition
    - Computing average correlation curves
    - Interpreting results
    
    Attributes:
        config: Configuration object
        
    Example:
        >>> analyzer = CrossCorrelationAnalyzer(config)
        >>> 
        >>> # Compute for all trials
        >>> xcorr_df = analyzer.analyze_all_trials(all_data)
        >>> 
        >>> # Get condition summary
        >>> summary = analyzer.get_condition_summary(xcorr_df)
        >>> 
        >>> # Get interpretation
        >>> for _, row in summary.iterrows():
        ...     print(analyzer.interpret_result(row))
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize CrossCorrelationAnalyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
    
    def analyze_single_trial(self, df: pd.DataFrame) -> CrossCorrelationResult:
        """
        Analyze a single trial.
        
        Args:
            df: DataFrame with single trial data
            
        Returns:
            CrossCorrelationResult
        """
        return compute_trial_crosscorr(
            df,
            normalize=self.config.normalize_xcorr,
            max_lag_frames=self.config.max_lag_frames,
            velocity_method=self.config.velocity_method,
            smooth_window=self.config.smooth_window
        )
    
    def analyze_all_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all trials in dataset.
        
        Args:
            df: DataFrame with all tracking data
            
        Returns:
            DataFrame with cross-correlation results per trial
        """
        return crosscorr_per_trial(
            df,
            normalize=self.config.normalize_xcorr,
            max_lag_frames=self.config.max_lag_frames,
            velocity_method=self.config.velocity_method,
            smooth_window=self.config.smooth_window
        )
    
    def get_condition_summary(
        self,
        xcorr_df: pd.DataFrame,
        groupby: List[str] = ['size', 'condition']
    ) -> pd.DataFrame:
        """
        Get summary statistics by condition.
        
        Args:
            xcorr_df: DataFrame from analyze_all_trials()
            groupby: Columns to group by
            
        Returns:
            Summary DataFrame
        """
        return aggregate_correlations(xcorr_df, groupby)
    
    def get_average_curve(
        self,
        df: pd.DataFrame,
        groupby: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get average cross-correlation curve.
        
        Args:
            df: Tracking data DataFrame
            groupby: Optional grouping
            
        Returns:
            Dictionary with curve data
        """
        return compute_average_crosscorr_curve(
            df,
            groupby=groupby,
            normalize=self.config.normalize_xcorr,
            max_lag_frames=self.config.max_lag_frames,
            velocity_method=self.config.velocity_method
        )
    
    def interpret_result(self, result: Union[CrossCorrelationResult, pd.Series, Dict]) -> Dict[str, str]:
        """
        Generate interpretations for a cross-correlation result.
        
        Args:
            result: CrossCorrelationResult, DataFrame row, or dict
            
        Returns:
            Dictionary with interpretation strings
        """
        # Extract values based on input type
        if isinstance(result, CrossCorrelationResult):
            lag_frames = result.optimal_lag_frames
            lag_ms = result.optimal_lag_ms
            max_corr = result.max_correlation
        elif isinstance(result, pd.Series):
            lag_frames = result.get('optimal_lag_frames', result.get('optimal_lag_frames_mean', 0))
            lag_ms = result.get('optimal_lag_ms', result.get('optimal_lag_ms_mean', 0))
            max_corr = result.get('max_correlation', result.get('max_correlation_mean', 0))
        else:
            lag_frames = result.get('optimal_lag_frames', 0)
            lag_ms = result.get('optimal_lag_ms', 0)
            max_corr = result.get('max_correlation', 0)
        
        return {
            'lag_interpretation': interpret_lag(lag_frames, lag_ms),
            'correlation_interpretation': interpret_correlation(max_corr),
            'summary': f"{interpret_lag(lag_frames, lag_ms)} with {interpret_correlation(max_corr).lower()}"
        }
    
    def compare_conditions(
        self,
        xcorr_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare cross-correlation results between conditions.
        
        Args:
            xcorr_df: DataFrame from analyze_all_trials()
            
        Returns:
            Dictionary with comparison results
        """
        dynamic = xcorr_df[xcorr_df['condition'] == 'dynamic']
        static = xcorr_df[xcorr_df['condition'] == 'static']
        
        dynamic_lag_mean = dynamic['optimal_lag_ms'].mean()
        static_lag_mean = static['optimal_lag_ms'].mean()
        
        dynamic_corr_mean = dynamic['max_correlation'].mean()
        static_corr_mean = static['max_correlation'].mean()
        
        # Determine which condition has better tracking
        lag_diff = abs(dynamic_lag_mean) - abs(static_lag_mean)
        corr_diff = dynamic_corr_mean - static_corr_mean
        
        if corr_diff > 0.05:
            corr_comparison = "Auditory feedback produces STRONGER velocity correlation"
        elif corr_diff < -0.05:
            corr_comparison = "No feedback produces STRONGER velocity correlation"
        else:
            corr_comparison = "Similar correlation strength between conditions"
        
        if lag_diff < -10:  # 10ms difference
            lag_comparison = "Auditory feedback produces SHORTER tracking lag"
        elif lag_diff > 10:
            lag_comparison = "No feedback produces SHORTER tracking lag"
        else:
            lag_comparison = "Similar tracking lag between conditions"
        
        return {
            'dynamic': {
                'mean_lag_ms': dynamic_lag_mean,
                'std_lag_ms': dynamic['optimal_lag_ms'].std(),
                'mean_correlation': dynamic_corr_mean,
                'std_correlation': dynamic['max_correlation'].std(),
                'n_trials': len(dynamic),
                'predictive_rate': dynamic['is_predictive'].mean()
            },
            'static': {
                'mean_lag_ms': static_lag_mean,
                'std_lag_ms': static['optimal_lag_ms'].std(),
                'mean_correlation': static_corr_mean,
                'std_correlation': static['max_correlation'].std(),
                'n_trials': len(static),
                'predictive_rate': static['is_predictive'].mean()
            },
            'comparison': {
                'lag_difference_ms': lag_diff,
                'correlation_difference': corr_diff,
                'lag_interpretation': lag_comparison,
                'correlation_interpretation': corr_comparison
            }
        }
