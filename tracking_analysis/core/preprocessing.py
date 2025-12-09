"""
Preprocessing Module
====================

Handles data preprocessing including velocity calculation, outlier removal,
and missing data handling. All methods are configurable with documented
pros/cons for each option.

Classes:
    Preprocessor: Main preprocessing class with configurable methods

Functions:
    calculate_velocity: Convert position to velocity signals
    remove_outliers: Detect and remove outlier data points
    handle_missing: Handle missing data with various strategies
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple, Dict, Any, List
from scipy import signal
from scipy import stats

from .config import Config, FRAME_RATE, FRAME_DURATION_MS


# =============================================================================
# VELOCITY CALCULATION
# =============================================================================

def calculate_velocity_difference(
    positions: np.ndarray,
    dt: float = FRAME_DURATION_MS / 1000  # Convert to seconds
) -> np.ndarray:
    """
    Calculate velocity using simple first-order difference.
    
    velocity[i] = (position[i] - position[i-1]) / dt
    
    Args:
        positions: 1D array of position values
        dt: Time step in seconds (default: 20ms = 0.02s)
        
    Returns:
        1D array of velocity values (same length, first value is 0)
        
    Pros:
        - Simple and fast computation
        - No assumptions about data smoothness
        - Preserves all temporal information
        
    Cons:
        - Sensitive to noise in position data
        - May produce noisy velocity signals
    """
    velocity = np.zeros_like(positions, dtype=float)
    velocity[1:] = np.diff(positions) / dt
    return velocity


def calculate_velocity_savgol(
    positions: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
    dt: float = FRAME_DURATION_MS / 1000
) -> np.ndarray:
    """
    Calculate velocity using Savitzky-Golay filter differentiation.
    
    Applies smoothing while computing the derivative, reducing noise
    while preserving signal shape.
    
    Args:
        positions: 1D array of position values
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial for fitting
        dt: Time step in seconds
        
    Returns:
        1D array of velocity values
        
    Pros:
        - Reduces noise in velocity signal
        - Preserves signal shape better than moving average
        - Good for noisy tracking data
        
    Cons:
        - May smooth out genuine rapid movements
        - Requires window size parameter tuning
        - Edge effects at trial boundaries
    """
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure window_length > polyorder
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    
    # Handle edge case of very short arrays
    if len(positions) < window_length:
        return calculate_velocity_difference(positions, dt)
    
    # Calculate derivative using Savitzky-Golay filter
    # deriv=1 computes the first derivative
    velocity = signal.savgol_filter(
        positions,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=dt
    )
    
    return velocity


def calculate_velocity(
    df: pd.DataFrame,
    method: Literal['difference', 'savgol'] = 'difference',
    smooth_window: int = 5,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate velocity for tracking data.
    
    Computes velocity for both target and mouse positions in X and Y dimensions.
    
    Args:
        df: DataFrame with position columns (Target_X, Target_Y, Mouse_X, Mouse_Y)
        method: 'difference' for simple difference, 'savgol' for Savitzky-Golay
        smooth_window: Window size for Savitzky-Golay filter
        columns: Specific columns to process (default: all position columns)
        
    Returns:
        DataFrame with added velocity columns (Target_Vx, Target_Vy, Mouse_Vx, Mouse_Vy)
        
    Example:
        >>> df = calculate_velocity(df, method='savgol', smooth_window=5)
        >>> df[['Target_Vx', 'Target_Vy', 'Mouse_Vx', 'Mouse_Vy']].head()
    """
    df = df.copy()
    
    # Default columns to process
    if columns is None:
        columns = ['Target_X', 'Target_Y', 'Mouse_X', 'Mouse_Y']
    
    # Map position columns to velocity column names
    velocity_names = {
        'Target_X': 'Target_Vx',
        'Target_Y': 'Target_Vy',
        'Mouse_X': 'Mouse_Vx',
        'Mouse_Y': 'Mouse_Vy'
    }
    
    for col in columns:
        if col not in df.columns:
            continue
        
        vel_col = velocity_names.get(col, f'{col}_velocity')
        positions = df[col].values
        
        if method == 'savgol':
            velocity = calculate_velocity_savgol(positions, smooth_window)
        else:  # 'difference'
            velocity = calculate_velocity_difference(positions)
        
        df[vel_col] = velocity
    
    # Also compute speed (magnitude of velocity vector) if we have both X and Y
    if 'Target_Vx' in df.columns and 'Target_Vy' in df.columns:
        df['Target_Speed'] = np.sqrt(df['Target_Vx']**2 + df['Target_Vy']**2)
    
    if 'Mouse_Vx' in df.columns and 'Mouse_Vy' in df.columns:
        df['Mouse_Speed'] = np.sqrt(df['Mouse_Vx']**2 + df['Mouse_Vy']**2)
    
    return df


# =============================================================================
# OUTLIER DETECTION AND REMOVAL
# =============================================================================

def detect_outliers_iqr(
    data: np.ndarray,
    threshold: float = 2.5
) -> np.ndarray:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Points are considered outliers if they fall below Q1 - k*IQR
    or above Q3 + k*IQR, where k is the threshold.
    
    Args:
        data: 1D array of values
        threshold: Multiplier for IQR (default: 2.5)
        
    Returns:
        Boolean array where True indicates outlier
        
    Pros:
        - Robust to non-normal distributions
        - Well-suited for reaction time data
        - Does not assume data shape
        
    Cons:
        - May be too aggressive for skewed distributions
        - Threshold requires tuning
    """
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(
    data: np.ndarray,
    threshold: float = 2.5
) -> np.ndarray:
    """
    Detect outliers using Z-score method.
    
    Points with |z-score| > threshold are considered outliers.
    
    Args:
        data: 1D array of values
        threshold: Z-score threshold (default: 2.5)
        
    Returns:
        Boolean array where True indicates outlier
        
    Pros:
        - Simple interpretation
        - Works well for normal distributions
        
    Cons:
        - Assumes normal distribution
        - Sensitive to existing outliers (mean/std affected)
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)
    
    if std == 0:
        return np.zeros(len(data), dtype=bool)
    
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold


def detect_outliers_mad(
    data: np.ndarray,
    threshold: float = 2.5
) -> np.ndarray:
    """
    Detect outliers using Median Absolute Deviation (MAD) method.
    
    More robust than Z-score as it uses median instead of mean.
    MAD = median(|X - median(X)|)
    Modified Z-score = 0.6745 * (X - median) / MAD
    
    Args:
        data: 1D array of values
        threshold: Modified Z-score threshold (default: 2.5)
        
    Returns:
        Boolean array where True indicates outlier
        
    Pros:
        - Highly robust to outliers
        - Does not assume normality
        - Good for heavily skewed data
        
    Cons:
        - May be overly aggressive
        - Less familiar to some researchers
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    
    if mad == 0:
        return np.zeros(len(data), dtype=bool)
    
    # 0.6745 is the scaling factor for normal distribution
    modified_z = 0.6745 * (data - median) / mad
    return np.abs(modified_z) > threshold


def remove_outliers(
    df: pd.DataFrame,
    method: Literal['none', 'iqr', 'zscore', 'mad'] = 'iqr',
    threshold: float = 2.5,
    columns: Optional[List[str]] = None,
    per_trial: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove outliers from tracking data.
    
    Can detect outliers in position or velocity data and remove affected frames.
    
    Args:
        df: DataFrame with tracking data
        method: Detection method ('none', 'iqr', 'zscore', 'mad')
        threshold: Threshold for outlier detection
        columns: Columns to check for outliers (default: Mouse positions)
        per_trial: Whether to detect outliers within each trial separately
        
    Returns:
        Tuple of (cleaned DataFrame, statistics dict with removal info)
        
    Example:
        >>> df_clean, stats = remove_outliers(df, method='iqr', threshold=2.5)
        >>> print(f"Removed {stats['n_removed']} outliers ({stats['percent_removed']:.1f}%)")
    """
    if method == 'none':
        return df.copy(), {'n_removed': 0, 'percent_removed': 0.0, 'method': 'none'}
    
    df = df.copy()
    
    # Default columns to check
    if columns is None:
        columns = ['Mouse_X', 'Mouse_Y']
    
    # Select detection function
    detect_funcs = {
        'iqr': detect_outliers_iqr,
        'zscore': detect_outliers_zscore,
        'mad': detect_outliers_mad
    }
    detect_func = detect_funcs[method]
    
    # Track outliers
    outlier_mask = np.zeros(len(df), dtype=bool)
    
    if per_trial and 'filename' in df.columns:
        # Detect outliers within each trial
        for filename in df['filename'].unique():
            trial_mask = df['filename'] == filename
            trial_indices = df.index[trial_mask]
            
            for col in columns:
                if col in df.columns:
                    data = df.loc[trial_mask, col].values
                    col_outliers = detect_func(data, threshold)
                    outlier_mask[trial_indices[col_outliers]] = True
    else:
        # Detect outliers globally
        for col in columns:
            if col in df.columns:
                data = df[col].values
                col_outliers = detect_func(data, threshold)
                outlier_mask |= col_outliers
    
    # Remove outliers
    n_total = len(df)
    n_removed = outlier_mask.sum()
    df_clean = df[~outlier_mask].copy()
    
    stats = {
        'method': method,
        'threshold': threshold,
        'columns_checked': columns,
        'n_total': n_total,
        'n_removed': n_removed,
        'n_remaining': len(df_clean),
        'percent_removed': (n_removed / n_total) * 100 if n_total > 0 else 0.0
    }
    
    return df_clean, stats


# =============================================================================
# MISSING DATA HANDLING
# =============================================================================

def handle_missing(
    df: pd.DataFrame,
    method: Literal['interpolate', 'drop', 'ffill', 'mean'] = 'interpolate',
    columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle missing data in tracking DataFrame.
    
    Args:
        df: DataFrame with tracking data
        method: Strategy for handling missing data
            - 'interpolate': Linear interpolation between valid points
            - 'drop': Remove rows with missing values
            - 'ffill': Forward fill (carry last valid observation)
            - 'mean': Replace with column mean
        columns: Columns to check/fix (default: position columns)
        
    Returns:
        Tuple of (cleaned DataFrame, statistics dict)
        
    Pros/Cons by Method:
        interpolate: Preserves continuity but may smooth over real gaps
        drop: Conservative but reduces sample size
        ffill: Simple but creates artificial plateaus
        mean: Preserves size but inappropriate for time series
    """
    df = df.copy()
    
    # Default columns
    if columns is None:
        columns = ['Target_X', 'Target_Y', 'Mouse_X', 'Mouse_Y']
    
    # Count initial missing values
    missing_before = df[columns].isna().sum().sum()
    
    if method == 'drop':
        df = df.dropna(subset=columns)
    elif method == 'interpolate':
        for col in columns:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
    elif method == 'ffill':
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    elif method == 'mean':
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    
    # Count remaining missing values
    missing_after = df[columns].isna().sum().sum() if len(df) > 0 else 0
    
    stats = {
        'method': method,
        'missing_before': int(missing_before),
        'missing_after': int(missing_after),
        'rows_before': len(df) + (missing_before if method == 'drop' else 0),
        'rows_after': len(df)
    }
    
    return df, stats


# =============================================================================
# PREPROCESSOR CLASS
# =============================================================================

class Preprocessor:
    """
    Main preprocessing class combining all preprocessing steps.
    
    Provides a unified interface for:
    - Velocity calculation
    - Outlier removal
    - Missing data handling
    - Time window filtering
    
    Attributes:
        config: Configuration object with preprocessing parameters
        
    Example:
        >>> preprocessor = Preprocessor(config)
        >>> df_processed, report = preprocessor.process(df)
        >>> print(report['summary'])
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Preprocessor.
        
        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or Config()
    
    def process(
        self,
        df: pd.DataFrame,
        calculate_vel: bool = True,
        remove_outliers: bool = True,
        handle_missing: bool = True,
        apply_time_window: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply full preprocessing pipeline to tracking data.
        
        Args:
            df: Input DataFrame with tracking data
            calculate_vel: Whether to calculate velocity
            remove_outliers: Whether to remove outliers
            handle_missing: Whether to handle missing data
            apply_time_window: Whether to filter by time window
            
        Returns:
            Tuple of (processed DataFrame, preprocessing report)
        """
        report = {
            'steps_applied': [],
            'original_rows': len(df),
            'final_rows': len(df)
        }
        
        # Step 1: Handle missing data first (before other processing)
        if handle_missing:
            df, missing_stats = self._handle_missing(df)
            report['missing_data'] = missing_stats
            report['steps_applied'].append('missing_data')
        
        # Step 2: Apply time window filter
        if apply_time_window:
            df, window_stats = self._apply_time_window(df)
            report['time_window'] = window_stats
            report['steps_applied'].append('time_window')
        
        # Step 3: Remove outliers
        if remove_outliers:
            df, outlier_stats = self._remove_outliers(df)
            report['outliers'] = outlier_stats
            report['steps_applied'].append('outliers')
        
        # Step 4: Calculate velocity
        if calculate_vel:
            df = self._calculate_velocity(df)
            report['steps_applied'].append('velocity')
        
        report['final_rows'] = len(df)
        report['rows_removed'] = report['original_rows'] - report['final_rows']
        report['percent_retained'] = (report['final_rows'] / report['original_rows'] * 100) if report['original_rows'] > 0 else 0
        
        # Generate summary text
        report['summary'] = self._generate_summary(report)
        
        return df, report
    
    def _calculate_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity using configured method."""
        return calculate_velocity(
            df,
            method=self.config.velocity_method,
            smooth_window=self.config.smooth_window
        )
    
    def _remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove outliers using configured method."""
        return remove_outliers(
            df,
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold
        )
    
    def _handle_missing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle missing data using configured method."""
        return handle_missing(
            df,
            method=self.config.missing_data_method
        )
    
    def _apply_time_window(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Filter data to configured time window."""
        start_frame, end_frame = self.config.get_time_window_frames()
        
        original_len = len(df)
        
        if 'Frame' in df.columns:
            mask = (df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)
            df = df[mask].copy()
        
        stats = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_ms': self.config.time_start_ms,
            'end_ms': self.config.time_end_ms,
            'rows_before': original_len,
            'rows_after': len(df)
        }
        
        return df, stats
    
    def _generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable preprocessing summary."""
        lines = ["Preprocessing Summary:", "=" * 40]
        
        lines.append(f"Original rows: {report['original_rows']}")
        lines.append(f"Final rows: {report['final_rows']} ({report['percent_retained']:.1f}% retained)")
        
        if 'missing_data' in report:
            m = report['missing_data']
            lines.append(f"\nMissing Data ({m['method']}):")
            lines.append(f"  - Before: {m['missing_before']} missing values")
            lines.append(f"  - After: {m['missing_after']} missing values")
        
        if 'time_window' in report:
            t = report['time_window']
            lines.append(f"\nTime Window:")
            lines.append(f"  - Range: {t['start_ms']}ms to {t['end_ms'] or 'end'}ms")
            lines.append(f"  - Frames {t['start_frame']} to {t['end_frame']}")
        
        if 'outliers' in report:
            o = report['outliers']
            lines.append(f"\nOutlier Removal ({o['method']}, threshold={o['threshold']}):")
            lines.append(f"  - Removed: {o['n_removed']} ({o['percent_removed']:.2f}%)")
        
        return "\n".join(lines)
    
    def process_by_trial(
        self,
        df: pd.DataFrame,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process data trial by trial.
        
        Useful when each trial should be processed independently.
        
        Args:
            df: DataFrame with 'filename' column identifying trials
            **kwargs: Arguments passed to process()
            
        Returns:
            Tuple of (processed DataFrame, aggregated report)
        """
        if 'filename' not in df.columns:
            return self.process(df, **kwargs)
        
        processed_dfs = []
        trial_reports = []
        
        for filename in df['filename'].unique():
            trial_df = df[df['filename'] == filename].copy()
            processed_trial, trial_report = self.process(trial_df, **kwargs)
            processed_dfs.append(processed_trial)
            trial_reports.append({
                'filename': filename,
                **trial_report
            })
        
        # Combine results
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Aggregate report
        total_original = sum(r['original_rows'] for r in trial_reports)
        total_final = sum(r['final_rows'] for r in trial_reports)
        
        aggregated_report = {
            'n_trials': len(trial_reports),
            'original_rows': total_original,
            'final_rows': total_final,
            'percent_retained': (total_final / total_original * 100) if total_original > 0 else 0,
            'trial_reports': trial_reports
        }
        
        return combined_df, aggregated_report
