"""
Configuration Module
====================

Defines default parameters, schemas, and validation functions for the
tracking analysis pipeline. All configurable options are documented with
their pros/cons and recommendations.

Classes:
    Config: Main configuration class with validation
    
Constants:
    DEFAULTS: Default parameter values
    PARAMETER_INFO: Detailed info about each parameter (pros, cons, recommendations)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any, Union
from pathlib import Path
import yaml


# =============================================================================
# CONSTANTS - Experiment-specific fixed values
# =============================================================================

# Screen dimensions (pixels)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 980

# Temporal parameters
FRAME_RATE = 50  # fps (frames per second)
FRAME_DURATION_MS = 20  # milliseconds per frame (1000 / 50)
TOTAL_FRAMES = 999  # Total frames per trial (~20 seconds)
TRIAL_DURATION_SEC = TOTAL_FRAMES / FRAME_RATE  # ~20 seconds

# Blob size conditions (Standard Deviation in pixels)
SD_VALUES = [21, 31, 34]

# Condition labels
CONDITION_LABELS = {
    'dynamic': 'Auditory Feedback',
    'static': 'No Feedback'
}


# =============================================================================
# PARAMETER INFORMATION - Pros, cons, and recommendations for UI legends
# =============================================================================

PARAMETER_INFO: Dict[str, Dict[str, Any]] = {
    # ----- Preprocessing Parameters -----
    'velocity_method': {
        'options': ['difference', 'savgol'],
        'default': 'difference',
        'recommended': 'difference',
        'description': 'Method for calculating velocity from position data',
        'options_info': {
            'difference': {
                'name': 'Simple Difference',
                'description': 'Calculates velocity as (position[t] - position[t-1]) / dt',
                'pros': [
                    'Simple and fast computation',
                    'No assumptions about data smoothness',
                    'Preserves all temporal information'
                ],
                'cons': [
                    'Sensitive to noise in position data',
                    'May produce noisy velocity signals'
                ],
                'when_to_use': 'When position data is relatively clean or when preserving exact temporal information is critical'
            },
            'savgol': {
                'name': 'Savitzky-Golay Filter',
                'description': 'Applies smoothing filter before differentiation',
                'pros': [
                    'Reduces noise in velocity signal',
                    'Preserves signal shape better than moving average',
                    'Good for noisy tracking data'
                ],
                'cons': [
                    'May smooth out genuine rapid movements',
                    'Requires window size parameter tuning',
                    'Edge effects at trial boundaries'
                ],
                'when_to_use': 'When position data is noisy or when smoother velocity profiles are needed for cross-correlation'
            }
        }
    },
    
    'smooth_window': {
        'type': 'integer',
        'default': 5,
        'min': 3,
        'max': 21,
        'step': 2,  # Must be odd for Savitzky-Golay
        'recommended': 5,
        'description': 'Window size for Savitzky-Golay smoothing (must be odd)',
        'pros': [
            'Larger window = smoother signal',
            'Smaller window = preserves more detail'
        ],
        'cons': [
            'Too large = loses temporal precision',
            'Too small = insufficient smoothing'
        ],
        'recommendation_logic': 'For 50fps data, 5 frames (100ms) provides good balance'
    },
    
    'outlier_method': {
        'options': ['none', 'iqr', 'zscore', 'mad'],
        'default': 'iqr',
        'recommended': 'iqr',
        'description': 'Method for detecting and removing outlier data points',
        'options_info': {
            'none': {
                'name': 'No Outlier Removal',
                'description': 'Keep all data points as-is',
                'pros': [
                    'Preserves all original data',
                    'No risk of removing valid extreme values'
                ],
                'cons': [
                    'Outliers can skew metrics significantly',
                    'May affect statistical test assumptions'
                ],
                'when_to_use': 'When data quality is high or outliers are meaningful'
            },
            'iqr': {
                'name': 'Interquartile Range (IQR)',
                'description': 'Removes points beyond Q1 - k*IQR or Q3 + k*IQR',
                'pros': [
                    'Robust to non-normal distributions',
                    'Well-suited for reaction time data',
                    'Does not assume data shape'
                ],
                'cons': [
                    'May be too aggressive for skewed distributions',
                    'Threshold (k) requires tuning'
                ],
                'when_to_use': 'Recommended for most psychophysics data'
            },
            'zscore': {
                'name': 'Z-Score',
                'description': 'Removes points with |z| > threshold',
                'pros': [
                    'Simple interpretation',
                    'Works well for normal distributions'
                ],
                'cons': [
                    'Assumes normal distribution',
                    'Sensitive to existing outliers (mean/std affected)'
                ],
                'when_to_use': 'When data is approximately normally distributed'
            },
            'mad': {
                'name': 'Median Absolute Deviation (MAD)',
                'description': 'Robust measure using median instead of mean',
                'pros': [
                    'Highly robust to outliers',
                    'Does not assume normality',
                    'Good for heavily skewed data'
                ],
                'cons': [
                    'May be overly aggressive',
                    'Less familiar to some researchers'
                ],
                'when_to_use': 'When data has many outliers or is heavily skewed'
            }
        }
    },
    
    'outlier_threshold': {
        'type': 'float',
        'default': 2.5,
        'min': 1.5,
        'max': 4.0,
        'step': 0.1,
        'recommended': 2.5,
        'description': 'Threshold for outlier detection (interpretation depends on method)',
        'thresholds_guide': {
            1.5: {'aggressiveness': 'Aggressive', 'typical_removal': '~7%'},
            2.0: {'aggressiveness': 'Moderate-Aggressive', 'typical_removal': '~3%'},
            2.5: {'aggressiveness': 'Moderate', 'typical_removal': '~1%'},
            3.0: {'aggressiveness': 'Conservative', 'typical_removal': '~0.3%'},
            3.5: {'aggressiveness': 'Very Conservative', 'typical_removal': '~0.1%'}
        },
        'recommendation_logic': '2.5 is standard for psychophysics; adjust based on data quality'
    },
    
    'missing_data_method': {
        'options': ['interpolate', 'drop', 'ffill', 'mean'],
        'default': 'interpolate',
        'recommended': 'interpolate',
        'description': 'Strategy for handling missing data points',
        'options_info': {
            'interpolate': {
                'name': 'Linear Interpolation',
                'description': 'Estimates missing values from neighboring points',
                'pros': [
                    'Preserves temporal continuity',
                    'Maintains sample size',
                    'Works well for small gaps'
                ],
                'cons': [
                    'May smooth over real tracking losses',
                    'Assumes linear movement between points'
                ],
                'when_to_use': 'Recommended for most tracking data with occasional missing points'
            },
            'drop': {
                'name': 'Drop Missing',
                'description': 'Remove trials or frames with missing data',
                'pros': [
                    'No artificial data introduced',
                    'Conservative approach'
                ],
                'cons': [
                    'Reduces sample size',
                    'May introduce selection bias'
                ],
                'when_to_use': 'When missing data is systematic or extensive'
            },
            'ffill': {
                'name': 'Forward Fill',
                'description': 'Carry last valid observation forward',
                'pros': [
                    'Simple implementation',
                    'Maintains sample size'
                ],
                'cons': [
                    'Creates artificial plateaus',
                    'Inappropriate for continuous motion'
                ],
                'when_to_use': 'Rarely recommended for tracking data'
            },
            'mean': {
                'name': 'Mean Imputation',
                'description': 'Replace missing with column mean',
                'pros': [
                    'Preserves sample size',
                    'Simple computation'
                ],
                'cons': [
                    'Reduces variance artificially',
                    'Inappropriate for time series',
                    'Ignores temporal structure'
                ],
                'when_to_use': 'Not recommended for tracking data'
            }
        }
    },
    
    # ----- Analysis Time Window -----
    'time_start_ms': {
        'type': 'integer',
        'default': 200,
        'min': 0,
        'max': 1000,
        'step': 50,
        'recommended': 200,
        'description': 'Start of analysis window in milliseconds',
        'recommendation_logic': '200ms excludes initial orienting response/saccade'
    },
    
    'time_end_ms': {
        'type': 'integer',
        'default': None,  # None means use full trial
        'min': 500,
        'max': 20000,
        'step': 100,
        'recommended': None,
        'description': 'End of analysis window in milliseconds (None = full trial)',
        'recommendation_logic': 'Use full trial unless there are known end-of-trial artifacts'
    },
    
    # ----- Cross-Correlation Parameters -----
    'normalize_xcorr': {
        'type': 'boolean',
        'default': True,
        'recommended': True,
        'description': 'Whether to normalize cross-correlation to [-1, 1]',
        'options_info': {
            True: {
                'name': 'Normalized',
                'pros': ['Comparable across conditions', 'Intuitive interpretation'],
                'cons': ['Loses absolute magnitude information']
            },
            False: {
                'name': 'Raw Covariance',
                'pros': ['Preserves effect size information'],
                'cons': ['Not comparable across different signal magnitudes']
            }
        }
    },
    
    'max_lag_frames': {
        'type': 'integer',
        'default': 50,
        'min': 10,
        'max': 200,
        'step': 10,
        'recommended': 50,
        'description': 'Maximum lag to compute in cross-correlation (in frames)',
        'recommendation_logic': '50 frames = 1 second at 50fps; typical human reaction time range'
    },
    
    # ----- Statistical Parameters -----
    'alpha': {
        'type': 'float',
        'default': 0.05,
        'options': [0.001, 0.01, 0.05, 0.10],
        'recommended': 0.05,
        'description': 'Significance level for statistical tests',
        'recommendation_logic': '0.05 is conventional; use 0.01 for stricter control'
    },
    
    'effect_size_measure': {
        'options': ['cohens_d', 'partial_eta_squared', 'generalized_eta_squared', 'omega_squared'],
        'default': 'partial_eta_squared',
        'recommended': 'partial_eta_squared',
        'description': 'Measure of practical significance',
        'options_info': {
            'cohens_d': {
                'name': "Cohen's d",
                'description': 'Standardized mean difference',
                'interpretation': {'small': 0.2, 'medium': 0.5, 'large': 0.8},
                'when_to_use': 'For pairwise comparisons'
            },
            'partial_eta_squared': {
                'name': 'Partial η²',
                'description': 'Proportion of variance explained (controlling for other factors)',
                'interpretation': {'small': 0.01, 'medium': 0.06, 'large': 0.14},
                'when_to_use': 'For ANOVA with multiple factors'
            },
            'generalized_eta_squared': {
                'name': 'Generalized η²',
                'description': 'Comparable across different designs',
                'interpretation': {'small': 0.01, 'medium': 0.06, 'large': 0.14},
                'when_to_use': 'For meta-analysis or cross-study comparison'
            },
            'omega_squared': {
                'name': 'ω²',
                'description': 'Less biased estimate of variance explained',
                'interpretation': {'small': 0.01, 'medium': 0.06, 'large': 0.14},
                'when_to_use': 'For more accurate population estimates'
            }
        }
    },
    
    'posthoc_method': {
        'options': ['bonferroni', 'tukey', 'holm', 'fdr', 'none'],
        'default': 'bonferroni',
        'recommended': 'bonferroni',
        'description': 'Multiple comparison correction method',
        'options_info': {
            'bonferroni': {
                'name': 'Bonferroni',
                'description': 'Divides alpha by number of comparisons',
                'pros': ['Simple', 'Conservative', 'Controls familywise error'],
                'cons': ['May be too conservative with many comparisons']
            },
            'tukey': {
                'name': 'Tukey HSD',
                'description': 'Honest Significant Difference for pairwise comparisons',
                'pros': ['Designed for ANOVA post-hoc', 'Less conservative than Bonferroni'],
                'cons': ['Assumes equal group sizes']
            },
            'holm': {
                'name': 'Holm-Bonferroni',
                'description': 'Step-down procedure, less conservative',
                'pros': ['More powerful than Bonferroni', 'Controls familywise error'],
                'cons': ['Slightly more complex interpretation']
            },
            'fdr': {
                'name': 'FDR (Benjamini-Hochberg)',
                'description': 'Controls false discovery rate instead of familywise error',
                'pros': ['More powerful for many comparisons', 'Good for exploratory analysis'],
                'cons': ['Does not control familywise error rate']
            },
            'none': {
                'name': 'No Correction',
                'description': 'Use uncorrected p-values',
                'pros': ['Maximum power'],
                'cons': ['Inflated Type I error with multiple comparisons']
            }
        }
    },
    
    # ----- Visualization Parameters -----
    'error_representation': {
        'options': ['se', 'sd', 'ci95'],
        'default': 'ci95',
        'recommended': 'ci95',
        'description': 'How to represent variability in plots',
        'options_info': {
            'se': {
                'name': 'Standard Error (SE)',
                'description': 'Standard deviation of the sampling distribution',
                'pros': ['Shows precision of mean estimate'],
                'cons': ['Can be misleading about data spread']
            },
            'sd': {
                'name': 'Standard Deviation (SD)',
                'description': 'Spread of individual observations',
                'pros': ['Shows actual data variability'],
                'cons': ['Does not directly relate to statistical inference']
            },
            'ci95': {
                'name': '95% Confidence Interval',
                'description': 'Range likely containing the true mean',
                'pros': ['Directly interpretable for inference', 'Non-overlapping CIs suggest significance'],
                'cons': ['Wider than SE, may look less precise']
            }
        }
    },
    
    'color_palette': {
        'options': ['colorblind_safe', 'viridis', 'Set2', 'plasma', 'husl'],
        'default': 'colorblind_safe',
        'recommended': 'colorblind_safe',
        'description': 'Color scheme for plots',
        'options_info': {
            'colorblind_safe': {
                'name': 'Colorblind Safe',
                'description': 'Optimized for color vision deficiency accessibility',
                'colors': ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377']
            },
            'viridis': {
                'name': 'Viridis',
                'description': 'Perceptually uniform, good for continuous data'
            },
            'Set2': {
                'name': 'Set2',
                'description': 'Qualitative palette, good for categorical data'
            },
            'plasma': {
                'name': 'Plasma',
                'description': 'Perceptually uniform, high contrast'
            },
            'husl': {
                'name': 'HUSL',
                'description': 'Evenly spaced hues, good for many categories'
            }
        }
    },
    
    'export_formats': {
        'options': ['png', 'svg', 'pdf', 'tiff'],
        'default': ['png', 'svg'],
        'recommended': ['png', 'svg'],
        'description': 'File formats for saving figures',
        'options_info': {
            'png': {'name': 'PNG', 'description': 'Raster, good for web/presentations', 'dpi': 300},
            'svg': {'name': 'SVG', 'description': 'Vector, scalable, editable'},
            'pdf': {'name': 'PDF', 'description': 'Vector, good for publications'},
            'tiff': {'name': 'TIFF', 'description': 'High-quality raster, required by some journals', 'dpi': 300}
        }
    },
    
    'animation_format': {
        'options': ['mp4', 'html', 'both'],
        'default': 'both',
        'recommended': 'both',
        'description': 'Output format for animations',
        'options_info': {
            'mp4': {
                'name': 'MP4 Video',
                'description': 'Standard video format, requires ffmpeg',
                'pros': ['Universal playback', 'Good for presentations'],
                'cons': ['Requires ffmpeg installation', 'Fixed resolution']
            },
            'html': {
                'name': 'Interactive HTML',
                'description': 'Browser-based with playback controls',
                'pros': ['Interactive', 'No external dependencies', 'Zoomable'],
                'cons': ['Larger file size', 'Requires browser']
            },
            'both': {
                'name': 'Both Formats',
                'description': 'Generate both MP4 and HTML',
                'pros': ['Maximum flexibility'],
                'cons': ['Longer generation time', 'More disk space']
            }
        }
    },
    
    'animation_fps': {
        'type': 'integer',
        'default': 30,
        'min': 10,
        'max': 60,
        'step': 5,
        'recommended': 30,
        'description': 'Frames per second for animation output',
        'recommendation_logic': '30fps is smooth enough for most purposes; 60fps for detailed motion analysis'
    },
    
    'animation_speed': {
        'type': 'float',
        'default': 1.0,
        'min': 0.25,
        'max': 4.0,
        'step': 0.25,
        'recommended': 1.0,
        'description': 'Playback speed multiplier (1.0 = real-time)',
        'recommendation_logic': '1.0 for accurate timing; 2.0 for quick review'
    }
}


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULTS: Dict[str, Any] = {
    # Data selection
    'data_path': './data',
    'participants': None,  # None means all
    'sizes': SD_VALUES,
    'conditions': ['dynamic', 'static'],
    
    # Preprocessing
    'velocity_method': 'difference',
    'smooth_window': 5,
    'outlier_method': 'iqr',
    'outlier_threshold': 2.5,
    'missing_data_method': 'interpolate',
    
    # Analysis window
    'time_start_ms': 200,
    'time_end_ms': None,
    
    # Cross-correlation
    'normalize_xcorr': True,
    'max_lag_frames': 50,
    
    # Statistics
    'alpha': 0.05,
    'effect_size_measure': 'partial_eta_squared',
    'posthoc_method': 'bonferroni',
    
    # Visualization
    'error_representation': 'ci95',
    'color_palette': 'colorblind_safe',
    'export_formats': ['png', 'svg'],
    'figure_width': 10,
    'figure_height': 6,
    'figure_dpi': 300,
    
    # Animation
    'animation_format': 'both',
    'animation_fps': 30,
    'animation_speed': 1.0,
    
    # Output
    'output_dir': './results',
    'create_timestamped_folder': True,
    'save_intermediate_results': True,
}


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """
    Configuration class for tracking analysis pipeline.
    
    Validates parameters and provides access to parameter metadata for UI.
    Can be saved/loaded from YAML files for reproducibility.
    
    Attributes:
        data_path: Path to data directory containing CSV files
        participants: List of participant IDs to include (None = all)
        sizes: List of blob sizes to analyze [21, 31, 34]
        conditions: List of conditions ['dynamic', 'static']
        velocity_method: 'difference' or 'savgol'
        smooth_window: Window size for Savitzky-Golay filter (odd integer)
        outlier_method: 'none', 'iqr', 'zscore', or 'mad'
        outlier_threshold: Threshold for outlier detection
        missing_data_method: 'interpolate', 'drop', 'ffill', or 'mean'
        time_start_ms: Analysis window start (ms)
        time_end_ms: Analysis window end (ms, None = full trial)
        normalize_xcorr: Whether to normalize cross-correlation
        max_lag_frames: Maximum lag for cross-correlation
        alpha: Significance level
        effect_size_measure: Effect size measure to compute
        posthoc_method: Multiple comparison correction method
        error_representation: Error bar type for plots
        color_palette: Color scheme name
        export_formats: List of figure export formats
        figure_width: Figure width in inches
        figure_height: Figure height in inches
        figure_dpi: Figure resolution
        animation_format: 'mp4', 'html', or 'both'
        animation_fps: Animation frame rate
        animation_speed: Playback speed multiplier
        output_dir: Base output directory
        create_timestamped_folder: Create timestamped subfolder for each run
        save_intermediate_results: Save intermediate computation results
    
    Example:
        >>> config = Config()  # Use defaults
        >>> config = Config(outlier_method='zscore', alpha=0.01)
        >>> config.save('my_config.yaml')
        >>> loaded_config = Config.load('my_config.yaml')
    """
    
    # Data selection
    data_path: str = DEFAULTS['data_path']
    participants: Optional[List[str]] = None
    sizes: List[int] = field(default_factory=lambda: DEFAULTS['sizes'].copy())
    conditions: List[str] = field(default_factory=lambda: DEFAULTS['conditions'].copy())
    
    # Preprocessing
    velocity_method: Literal['difference', 'savgol'] = DEFAULTS['velocity_method']
    smooth_window: int = DEFAULTS['smooth_window']
    outlier_method: Literal['none', 'iqr', 'zscore', 'mad'] = DEFAULTS['outlier_method']
    outlier_threshold: float = DEFAULTS['outlier_threshold']
    missing_data_method: Literal['interpolate', 'drop', 'ffill', 'mean'] = DEFAULTS['missing_data_method']
    
    # Analysis window
    time_start_ms: int = DEFAULTS['time_start_ms']
    time_end_ms: Optional[int] = DEFAULTS['time_end_ms']
    
    # Cross-correlation
    normalize_xcorr: bool = DEFAULTS['normalize_xcorr']
    max_lag_frames: int = DEFAULTS['max_lag_frames']
    
    # Statistics
    alpha: float = DEFAULTS['alpha']
    effect_size_measure: str = DEFAULTS['effect_size_measure']
    posthoc_method: Literal['bonferroni', 'tukey', 'holm', 'fdr', 'none'] = DEFAULTS['posthoc_method']
    
    # Visualization
    error_representation: Literal['se', 'sd', 'ci95'] = DEFAULTS['error_representation']
    color_palette: str = DEFAULTS['color_palette']
    export_formats: List[str] = field(default_factory=lambda: DEFAULTS['export_formats'].copy())
    figure_width: float = DEFAULTS['figure_width']
    figure_height: float = DEFAULTS['figure_height']
    figure_dpi: int = DEFAULTS['figure_dpi']
    
    # Animation
    animation_format: Literal['mp4', 'html', 'both'] = DEFAULTS['animation_format']
    animation_fps: int = DEFAULTS['animation_fps']
    animation_speed: float = DEFAULTS['animation_speed']
    
    # Output
    output_dir: str = DEFAULTS['output_dir']
    create_timestamped_folder: bool = DEFAULTS['create_timestamped_folder']
    save_intermediate_results: bool = DEFAULTS['save_intermediate_results']
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate all configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate smooth_window is odd
        if self.smooth_window % 2 == 0:
            raise ValueError(f"smooth_window must be odd, got {self.smooth_window}")
        
        # Validate smooth_window range
        if not (3 <= self.smooth_window <= 21):
            raise ValueError(f"smooth_window must be between 3 and 21, got {self.smooth_window}")
        
        # Validate outlier_threshold
        if not (1.0 <= self.outlier_threshold <= 5.0):
            raise ValueError(f"outlier_threshold must be between 1.0 and 5.0, got {self.outlier_threshold}")
        
        # Validate alpha
        if not (0.001 <= self.alpha <= 0.10):
            raise ValueError(f"alpha must be between 0.001 and 0.10, got {self.alpha}")
        
        # Validate sizes
        for size in self.sizes:
            if size not in SD_VALUES:
                raise ValueError(f"Invalid size {size}, must be one of {SD_VALUES}")
        
        # Validate conditions
        for cond in self.conditions:
            if cond not in ['dynamic', 'static']:
                raise ValueError(f"Invalid condition '{cond}', must be 'dynamic' or 'static'")
        
        # Validate time window
        if self.time_end_ms is not None and self.time_start_ms >= self.time_end_ms:
            raise ValueError(f"time_start_ms ({self.time_start_ms}) must be less than time_end_ms ({self.time_end_ms})")
        
        # Validate animation_fps
        if not (10 <= self.animation_fps <= 60):
            raise ValueError(f"animation_fps must be between 10 and 60, got {self.animation_fps}")
        
        # Validate animation_speed
        if not (0.25 <= self.animation_speed <= 4.0):
            raise ValueError(f"animation_speed must be between 0.25 and 4.0, got {self.animation_speed}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            'data_path': self.data_path,
            'participants': self.participants,
            'sizes': self.sizes,
            'conditions': self.conditions,
            'velocity_method': self.velocity_method,
            'smooth_window': self.smooth_window,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'missing_data_method': self.missing_data_method,
            'time_start_ms': self.time_start_ms,
            'time_end_ms': self.time_end_ms,
            'normalize_xcorr': self.normalize_xcorr,
            'max_lag_frames': self.max_lag_frames,
            'alpha': self.alpha,
            'effect_size_measure': self.effect_size_measure,
            'posthoc_method': self.posthoc_method,
            'error_representation': self.error_representation,
            'color_palette': self.color_palette,
            'export_formats': self.export_formats,
            'figure_width': self.figure_width,
            'figure_height': self.figure_height,
            'figure_dpi': self.figure_dpi,
            'animation_format': self.animation_format,
            'animation_fps': self.animation_fps,
            'animation_speed': self.animation_speed,
            'output_dir': self.output_dir,
            'create_timestamped_folder': self.create_timestamped_folder,
            'save_intermediate_results': self.save_intermediate_results,
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save YAML file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Config instance with loaded parameters
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @staticmethod
    def get_parameter_info(param_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a parameter for UI display.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Dictionary with parameter info (description, options, pros/cons, etc.)
        """
        return PARAMETER_INFO.get(param_name, {})
    
    @staticmethod
    def get_all_parameter_info() -> Dict[str, Dict[str, Any]]:
        """
        Get information about all parameters.
        
        Returns:
            Dictionary mapping parameter names to their info
        """
        return PARAMETER_INFO.copy()
    
    def get_time_window_frames(self) -> tuple:
        """
        Convert time window from milliseconds to frames.
        
        Returns:
            Tuple of (start_frame, end_frame)
        """
        start_frame = int(self.time_start_ms / FRAME_DURATION_MS)
        
        if self.time_end_ms is None:
            end_frame = TOTAL_FRAMES
        else:
            end_frame = int(self.time_end_ms / FRAME_DURATION_MS)
        
        return start_frame, end_frame
    
    def get_recommended_value(self, param_name: str) -> Any:
        """
        Get the recommended value for a parameter.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Recommended value or None if no recommendation
        """
        info = PARAMETER_INFO.get(param_name, {})
        return info.get('recommended')


def get_recommendation_for_data(param_name: str, data_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get dynamic recommendation based on loaded data characteristics.
    
    This function analyzes data statistics to provide context-aware
    recommendations that may differ from static defaults.
    
    Args:
        param_name: Name of the parameter
        data_stats: Dictionary with data statistics (n_participants, n_files, etc.)
        
    Returns:
        Dictionary with 'value' and 'reason' keys
    """
    recommendations = {}
    
    n_participants = data_stats.get('n_participants', 0)
    n_files = data_stats.get('n_files', 0)
    
    if param_name == 'outlier_threshold':
        if n_participants < 10:
            recommendations = {
                'value': 3.0,
                'reason': f'With only {n_participants} participants, using conservative threshold (3.0) to preserve data'
            }
        else:
            recommendations = {
                'value': 2.5,
                'reason': 'Standard threshold (2.5) appropriate for adequate sample size'
            }
    
    elif param_name == 'posthoc_method':
        if n_participants < 20:
            recommendations = {
                'value': 'holm',
                'reason': f'With {n_participants} participants, Holm method provides good power while controlling error'
            }
        else:
            recommendations = {
                'value': 'bonferroni',
                'reason': 'Bonferroni appropriate for larger samples'
            }
    
    elif param_name == 'alpha':
        if n_participants < 15:
            recommendations = {
                'value': 0.05,
                'reason': 'Standard alpha (0.05) recommended; stricter levels may lack power'
            }
        else:
            recommendations = {
                'value': 0.05,
                'reason': 'Standard alpha (0.05) provides good balance'
            }
    
    # Default: return static recommendation
    if not recommendations:
        info = PARAMETER_INFO.get(param_name, {})
        recommendations = {
            'value': info.get('recommended', DEFAULTS.get(param_name)),
            'reason': info.get('recommendation_logic', 'Default recommendation')
        }
    
    return recommendations
